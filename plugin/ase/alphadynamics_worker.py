"""Isolated AD+GBn2 worker; does not modify the deployed AD packages."""
import hashlib
import json
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def gbn2_parameters(symbols, xyz, metadata):
  """mbondi3/GBn2 parameters with exact residue and atom identities.

  OpenMM supplies standard residue bonds. Capping/nonstandard hydrogen
  partners missing from that table are assigned inside their own residue;
  ambiguous or detached hydrogens are rejected. No inter-residue nearest
  heavy atom assignment is used.
  """
  from openmm import app
  from openmm.app.internal.customgbforces import GBSAGBn2Force
  top = app.Topology()
  atoms, residues, chains = [], {}, {}
  for symbol, m in zip(symbols, metadata):
    chain = chains.setdefault(m["chain"], None)
    if chain is None:
      chain = chains[m["chain"]] = top.addChain(m["chain"])
    key = (m["chain"], m["residue_index"])
    if key not in residues:
      residues[key] = top.addResidue(m["resname"], chain, id=m["resid"])
    atoms.append(top.addAtom(m["name"], app.element.Element.getBySymbol(symbol),
      residues[key]))
  top.createStandardBonds()
  bonded = {i: [] for i in range(len(atoms))}
  for a, b in top.bonds():
    bonded[a.index].append(b.index)
    bonded[b.index].append(a.index)
  parents = {}
  for i, symbol in enumerate(symbols):
    if symbol != "H": continue
    possible = [j for j in bonded[i] if symbols[j] != "H"]
    if not possible:
      m = metadata[i]
      possible = [j for j, s in enumerate(symbols) if s != "H" and
        metadata[j]["chain"] == m["chain"] and
        metadata[j]["residue_index"] == m["residue_index"] and
        np.linalg.norm(xyz[i]-xyz[j]) <= 1.5]
      possible.sort(key=lambda j: np.linalg.norm(xyz[i]-xyz[j]))
      if len(possible) > 1 and (np.linalg.norm(xyz[i]-xyz[possible[1]]) -
          np.linalg.norm(xyz[i]-xyz[possible[0]]) < 0.15):
        raise ValueError("Ambiguous hydrogen parent: %s" % m)
      if possible:
        possible = possible[:1]
        top.addBond(atoms[i], atoms[possible[0]])
    if len(possible) != 1 or np.linalg.norm(xyz[i]-xyz[possible[0]]) > 1.7:
      raise ValueError("Invalid hydrogen parent: %s" % metadata[i])
    parents[str(i)] = int(possible[0])
  standard = np.asarray(GBSAGBn2Force.getStandardParameters(top), dtype=float)
  return standard[:, 0]*10, standard[:, 1], standard[:, 2:], parents


def run(fd, toolkit):
  toolkit = Path(toolkit)
  sys.path.insert(0, str(toolkit))
  # The user's AD deployment keeps its optional Python deps here.
  for directory in [toolkit / "ADpython/.deps"]:
    if directory.is_dir(): sys.path.append(str(directory))
  conn = Connection(fd)
  engine = None
  try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from AD_singlepoint import SinglePoint
    from AD_singlepoint_implicit import ImplicitSinglePoint
    from AD_singlepoint_implicit.singlepoint_implicit.solvent import OBC2Solvent
    vacuum = SinglePoint(response="total")
    conn.send(dict(ready=True))
    while True:
      try: request = conn.recv()
      except EOFError: break
      try:
        symbols = request["symbols"]
        xyz = np.asarray(request["positions"], dtype=float)
        directory = Path(request["directory"])
        if engine is None:
          radii, scales, extra, parents = gbn2_parameters(
            symbols, xyz, request["metadata"])
          solvent = OBC2Solvent(symbols, np.zeros(len(symbols)), radii, scales,
            extra_parameters=extra, model="GBn2", platform="CUDA",
            precision="double", charge_potential_method="analytic",
            response_kernel="triton",
            atom_names=[m["name"] for m in request["metadata"]],
            source="Q|R hierarchy identity; OpenMM GBn2/mbondi3")
          engine = ImplicitSinglePoint(solvent, vacuum_engine=vacuum,
            charge_mode="aux", aux_response="total",
            aux_evaluation="single_forward")
          np.savez_compressed(directory / "gbn2_parameters.npz", radii_A=radii,
            scales=scales, extra_parameters=extra)
          (directory / "identity.json").write_text(json.dumps(dict(
            atoms=request["metadata"], hydrogen_parents=parents), indent=2))
          sources = {}
          # Hash imported implementation files, not unrelated report archives.
          for name, module in list(sys.modules.items()):
            if name.startswith(("AD_singlepoint", "ADpython")):
              file = getattr(module, "__file__", None)
              if file and Path(file).is_file():
                sources[str(Path(file).resolve())] = hashlib.sha256(
                  Path(file).read_bytes()).hexdigest()
          (directory / "provenance.json").write_text(json.dumps(dict(
            python=sys.executable, torch=torch.__version__,
            toolkit=str(toolkit.resolve()), sources=sources,
            vacuum_response="total", solvent="GBn2", charge_mode="aux",
            aux_response="total", aux_evaluation="single_forward"), indent=2))
        started = time.perf_counter()
        result = engine.compute(symbols, xyz, target_charge=request["charge"],
          atom_names=[m["name"] for m in request["metadata"]])
        if result["aux_response"] != "total" or result["aux_evaluation"] != "single_forward":
          raise ValueError("AD did not use complete single-forward AUX response")
        solvent_meta = result["solvent"]
        if solvent_meta["platform"] != "CUDA" or solvent_meta["precision"] != "double":
          raise ValueError("Unexpected solvent backend/precision")
        conn.send(dict(result={key: result[key] for key in
          ("pe", "forces", "units", "aux_response")}, metadata=dict(
          natoms=len(symbols), charge=request["charge"],
          solvent=solvent_meta, solvation_energy_eV=result["solvation_energy"],
          aux_evaluation=result["aux_evaluation"],
          force_convention=result["force_convention"],
          compute_seconds=time.perf_counter()-started)))
      except Exception:
        np.savez_compressed(Path(request["directory"]) / "failed_request.npz",
          symbols=request["symbols"], positions=request["positions"],
          charge=request["charge"])
        conn.send(dict(error=traceback.format_exc()))
  except Exception:
    try: conn.send(dict(error=traceback.format_exc()))
    except Exception: pass
  finally:
    if engine is not None: engine.close()
    conn.close()


if __name__ == "__main__":
  run(int(sys.argv[1]), sys.argv[2])
