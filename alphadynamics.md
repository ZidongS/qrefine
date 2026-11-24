# AlphaDynamics + GBn2

Development baseline: Q|R `d88d575261490b49fd1bd82451332cd898c50ade`.

This engine uses the user's deployed `AD_singlepoint.SinglePoint(response="total")`
and `AD_singlepoint_implicit.ImplicitSinglePoint` with mechanical GBn2,
AUX charges, full AUX response and the single-forward charge-potential API.
It does not change the AD models, cutoffs, force implementation or Q|R controller.
The user must supply the AD runtime and toolkit; they are not bundled here.

```
export ALPHADYNAMICS_PYTHON=/path/to/AD/env/bin/python
export ALPHADYNAMICS_TOOLKIT=/path/to/zidong_analysis_toolkit
qr.refine prepared_model.pdb data.mtz quantum.engine_name=alphadynamics
```

The same settings can be provided through `quantum.alphadynamics.python` and
`quantum.alphadynamics.toolkit`. The AD interpreter needs torch, OpenMM, scipy,
Triton and the deployed AD dependencies. It runs in a subprocess, independently
of the Phenix/cctbx interpreter. `CUDA_VISIBLE_DEVICES` is inherited from Slurm;
this adapter does not request or select a second GPU.

Use the current upstream Reduce2 QM-ready preparation and compatible cctbx
dependencies. Old hydrogenated inputs are not automatically treated as QM-ready.
Current AQuaRef auto-presets are shared with AIMNet2, including their final
effective settings; record those settings rather than assuming old CLI options
have the same meaning. Initial support is for whole-model/symmetry expansion
and conformer routes, not clustered fragments.

## Units and atom identity

`energy_free` is native eV and `forces` is native eV/Angstrom, directly from AD.
Q|R's existing `from_qm.target_and_gradients` performs the sole conversion to
kcal/mol, negates forces to gradients, and applies its usual master selection.
There is no AIMNet2-specific scaling or model substitution.

Names, residue names, chain IDs, insertion/residue IDs and residue indices come
directly from the hierarchy passed to the calculator. No rounded PDB or atom
serial parsing is used. OpenMM creates standard residue bonds for mbondi3;
missing capping/nonstandard H partners are assigned only within their own residue.
Ambiguous or detached H assignments fail. Parameters and identities are fixed
at calculator initialization, then coordinates and the target charge are updated.
The solvent uses CUDA double precision and analytic/Triton charge potentials.

## Diagnostics and limits

Each calculator writes `<label>.alphadynamics/` with identities, hydrogen parent
indices, solvent parameters, imported AD implementation hashes and per-call
timings/energies. Failed worker requests and short-contact inputs are retained.
Full runtime/model provenance must additionally be frozen in each experiment's
manifest; imported source hashes alone are not model-weight hashes.

The default minimum quantum-cluster separation is 0.7 Angstrom. A shorter
contact stops the case before inference. This is a fail-fast guard, **not** a new
line-search/backtracking algorithm: the upstream optimizer is unchanged. A stopped
case is not a successful refinement and must not be silently counted as one.
All-atom MolProbity validation remains an independent output check.

The current upstream expansion still supplies master gradients and zero energy;
this interface preserves that convention. A single-point finite-difference check
does not validate the derivative of a moving expanded cluster's total energy.
Upstream X-ray weight search uses Rfree for selection, so its final Rfree is not
a wholly independent holdout metric. AIMNet2's legacy scale needs a separately
audited baseline before model-ranking claims.

No historical checkout, source compatibility copy, result directory or shared
environment should be overwritten for this development. Run from an isolated
checkout and run directory with pinned Q|R, cctbx, AD sources and weight hashes.
