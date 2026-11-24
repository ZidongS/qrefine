"""AlphaDynamics mechanical GBn2 interface. ASE-side values are eV and eV/A.

The AD runtime is a separate process, so its torch/OpenMM libraries do not
replace libraries loaded by Phenix. No AIMNet2 calculator is monkey-patched.
"""
from __future__ import absolute_import, division, print_function
import atexit
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Connection
import numpy as np
from ase.calculators.calculator import Calculator


def available(params):
  p = params.quantum.alphadynamics
  runtime = p.python or os.environ.get("ALPHADYNAMICS_PYTHON")
  toolkit = p.toolkit or os.environ.get("ALPHADYNAMICS_TOOLKIT")
  return bool(runtime and shutil.which(runtime) and toolkit and
    (Path(toolkit) / "AD_singlepoint_implicit").is_dir())


def validate_result(result, n_atoms):
  """Check the public AD contract without scaling or changing force signs."""
  units = result.get("units", {})
  if units.get("energy") != "eV" or units.get("force") != "eV/Angstrom":
    raise ValueError("AD must return explicit eV and eV/Angstrom units")
  energy = float(result["pe"])
  forces = np.asarray(result["forces"], dtype=np.float64)
  if forces.shape != (n_atoms, 3):
    raise ValueError("AD force shape does not match quantum model")
  if not np.isfinite(energy) or not np.isfinite(forces).all():
    raise ValueError("Nonfinite AD energy or forces")
  if result.get("aux_response") != "total":
    raise ValueError("Full AUX charge response is required")
  return energy, forces


class AlphaDynamicsCalculator(Calculator):
  def __init__(self, params):
    super(AlphaDynamicsCalculator, self).__init__()
    self.settings = params
    self.process = self.conn = None
    self.identity = None
    self.calls = 0
    self.audit_directory = None
    atexit.register(self.close)

  def _start(self):
    p = self.settings
    runtime = p.python or os.environ.get("ALPHADYNAMICS_PYTHON")
    toolkit = p.toolkit or os.environ.get("ALPHADYNAMICS_TOOLKIT")
    if not runtime or not toolkit:
      raise ValueError("Set quantum.alphadynamics.python/toolkit or "
        "ALPHADYNAMICS_PYTHON/ALPHADYNAMICS_TOOLKIT")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LIBTBX_BUILD", "CONDA_PREFIX"):
      env.pop(key, None)
    env["PATH"] = str(Path(runtime).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    parent, child = socket.socketpair()
    try:
      self.process = subprocess.Popen([runtime,
        str(Path(__file__).with_name("alphadynamics_worker.py")),
        str(child.fileno()), str(Path(toolkit).resolve())],
        pass_fds=(child.fileno(),), env=env)
    except Exception:
      parent.close()
      raise
    finally:
      child.close()
    self.conn = Connection(parent.detach())
    self._receive()

  def _receive(self):
    if not self.conn.poll(self.settings.timeout):
      self.close()
      raise RuntimeError("AlphaDynamics worker timed out")
    reply = self.conn.recv()
    if "error" in reply:
      self.close()
      raise RuntimeError(reply["error"])
    return reply

  def run_qr(self, atoms, coordinates, charge, pointcharges, define_str=None):
    if pointcharges is not None:
      raise ValueError("AD+GBn2 supports mechanical embedding only")
    metadata = atoms.info.get("qrefine_atom_metadata")
    if metadata is None or len(metadata) != len(atoms):
      raise ValueError("AD+GBn2 requires atom/residue identity from Q|R hierarchy")
    identity = (atoms.get_chemical_symbols(), metadata)
    if self.identity is None:
      self.identity = copy.deepcopy(identity)
    elif identity != self.identity:
      raise ValueError("Quantum model atom identity changed within a calculator")
    xyz = np.asarray(atoms.positions, dtype=float)
    if not np.isfinite(xyz).all():
      raise ValueError("Nonfinite quantum coordinates")
    if self.audit_directory is None:
      self.audit_directory = Path(str(self.label) + ".alphadynamics")
      self.audit_directory.mkdir(parents=True, exist_ok=True)
    # Fail before inference on invalid input/trials; the stock controller
    # does not implement collision-aware backtracking. Never silently accept.
    if self.settings.minimum_distance > 0:
      from scipy.spatial import cKDTree
      pairs = cKDTree(xyz).query_pairs(self.settings.minimum_distance,
        output_type="ndarray")
      if len(pairs):
        i, j = min(pairs, key=lambda ij: np.linalg.norm(xyz[ij[0]]-xyz[ij[1]]))
        np.savez_compressed(self.audit_directory / "short_contact_request.npz",
          positions=xyz, symbols=identity[0], pairs=pairs)
        (self.audit_directory / "short_contact_identity.json").write_text(
          json.dumps(metadata, indent=2))
        raise ValueError("AD_SHORT_CONTACT " + json.dumps(dict(
          indices=[int(i), int(j)], atoms=[metadata[i], metadata[j]],
          distance_A=float(np.linalg.norm(xyz[i]-xyz[j])))))
    if self.conn is None:
      self._start()
    started = time.perf_counter()
    self.conn.send(dict(symbols=identity[0], positions=xyz,
      charge=charge, metadata=metadata, directory=str(self.audit_directory.resolve())))
    reply = self._receive()
    self.energy_free, self.forces = validate_result(reply["result"], len(atoms))
    self.calls += 1
    record = dict(call=self.calls, energy_eV=self.energy_free,
      max_force_eV_A=float(np.abs(self.forces).max()),
      seconds=time.perf_counter()-started, **reply["metadata"])
    with (self.audit_directory / "calls.jsonl").open("a") as stream:
      stream.write(json.dumps(record) + "\n")
    if self.calls == 1:
      print("ALPHADYNAMICS_FIRST_CALL " + json.dumps(record), flush=True)

  def close(self):
    if self.conn is not None:
      try: self.conn.close()
      except Exception: pass
      self.conn = None
    if self.process is not None and self.process.poll() is None:
      self.process.terminate()
      try: self.process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait()
