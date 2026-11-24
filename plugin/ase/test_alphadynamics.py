"""Fast adapter contract tests; no model weights, GPU or cctbx required.

Run: python plugin/ase/test_alphadynamics.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from ase import Atoms


def load(name):
  spec = importlib.util.spec_from_file_location(name,
    Path(__file__).with_name(name + ".py"))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


adapter = load("alphadynamics_qr")
worker = load("alphadynamics_worker")


def identity(names, residue="GLY", chain="A", index=0):
  return [dict(name=name, resname=residue, chain=chain, resid="1",
    residue_index=index, altloc="") for name in names]


class HarmonicConnection:
  """Known potential exposes unit/sign/order errors through the real bridge."""
  def send(self, request): self.request = request
  def poll(self, timeout): return True
  def recv(self):
    x = self.request["positions"]
    return dict(result=dict(pe=float(np.sum(x*x)/2), forces=-x,
      units=dict(energy="eV", force="eV/Angstrom"), aux_response="total"),
      metadata={})
  def close(self): pass


class AdapterTests(unittest.TestCase):
  def calculator(self, directory):
    calc = adapter.AlphaDynamicsCalculator(SimpleNamespace(
      minimum_distance=0.7, timeout=5, python=None, toolkit=None))
    calc.set_label(str(Path(directory) / "quantum"))
    calc.conn = HarmonicConnection()
    return calc

  def test_native_units_and_force_finite_difference(self):
    with tempfile.TemporaryDirectory() as directory:
      calc = self.calculator(directory)
      atoms = Atoms("NH", positions=[[2., 0, 0], [3., 0, 0]])
      atoms.info["qrefine_atom_metadata"] = identity(["N", "H"])
      calc.run_qr(atoms, None, 0, None)
      self.assertEqual(calc.energy_free, 6.5) # must not become 0.2819 or 149.89
      np.testing.assert_array_equal(calc.forces, -atoms.positions)
      force = calc.forces[0, 0]
      eps = 1e-5
      atoms.positions[0, 0] += eps
      calc.run_qr(atoms, None, 0, None)
      ep = calc.energy_free
      atoms.positions[0, 0] -= 2*eps
      calc.run_qr(atoms, None, 0, None)
      self.assertAlmostEqual(force, -(ep-calc.energy_free)/(2*eps), places=8)
      atoms.info["qrefine_atom_metadata"].reverse()
      with self.assertRaisesRegex(ValueError, "identity changed"):
        calc.run_qr(atoms, None, 0, None)
      calc.close()

  def test_reject_invalid_result_contract(self):
    result = dict(pe=1., forces=np.zeros((2, 3)),
      units=dict(energy="eV", force="eV/Angstrom"), aux_response="total")
    for field, value in [("pe", np.nan), ("forces", np.zeros((3, 3))),
        ("units", dict(energy="kcal/mol", force="kcal/mol/A")),
        ("aux_response", "fixed")]:
      with self.subTest(field=field):
        with self.assertRaises(ValueError):
          adapter.validate_result(dict(result, **{field: value}), 2)

  def test_short_contact_saved_before_inference(self):
    with tempfile.TemporaryDirectory() as directory:
      calc = self.calculator(directory)
      atoms = Atoms("NH", positions=[[0, 0, 0], [0.4, 0, 0]])
      atoms.info["qrefine_atom_metadata"] = identity(["N", "H"])
      with self.assertRaisesRegex(ValueError, "AD_SHORT_CONTACT"):
        calc.run_qr(atoms, None, 0, None)
      self.assertTrue((calc.audit_directory / "short_contact_request.npz").is_file())
      self.assertFalse(hasattr(calc.conn, "request"))
      calc.close()

  def test_gbn2_chain_identity_and_hydrogen_partners(self):
    symbols = ["N", "H", "N", "H"]
    metadata = identity(["NH1", "HH11"], "ARG", "A") + identity(
      ["NH2", "HH21"], "ARG", "B", 1)
    xyz = np.array([[0.,0,0], [1.,0,0], [10.,0,0], [11.,0,0]])
    radii, scales, extra, parents = worker.gbn2_parameters(symbols, xyz, metadata)
    self.assertEqual(parents, {"1": 0, "3": 2})
    self.assertEqual(extra.shape, (4, 3))
    self.assertTrue(np.isfinite(radii).all() and np.isfinite(scales).all())
    # mbondi3 ARG guanidinium hydrogen radius is 1.17 A.
    np.testing.assert_allclose(radii[[1, 3]], [1.17, 1.17])
    xyz[1] = [5., 0, 0]
    with self.assertRaisesRegex(ValueError, "Invalid hydrogen parent"):
      worker.gbn2_parameters(symbols, xyz, metadata)


if __name__ == "__main__": unittest.main()
