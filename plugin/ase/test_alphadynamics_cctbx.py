"""GPU gate: real worker, Q|R unit/sign bridge, presets and finite differences."""
import json
from pathlib import Path
import sys
import numpy as np
import ase.units
import iotbx.pdb
from iotbx import phil
from scitbx.array_family import flex
from qrefine import qr, api
from qrefine.restraints import from_qm

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
p=qr.get_default_params();p.quantum.engine_name='alphadynamics'
api.check_engine(p)
for group,kw in [('cryoem',dict(has_map=True)),('xray_high',dict(has_data=True,d_min=1.0)),('xray_low',dict(has_data=True,d_min=3.5))]:
 a=qr.get_default_params();b=qr.get_default_params()
 a.quantum.engine_name='alphadynamics';b.quantum.engine_name='aimnet2'
 api.finalize_params(a,n_conformers=0,**kw);api.finalize_params(b,n_conformers=0,**kw)
 for name in ['expansion','restraints']:
  assert getattr(a,name)==getattr(b,name),(group,name)
 assert vars(a.refine)==vars(b.refine),(group,'refine')
print('PRESET_PARITY_OK',flush=True)
pdb='''CRYST1   25.000   25.000   25.000  90.00  90.00  90.00 P 1
HETATM    1  O   HOH A   1       5.000   5.000   5.000  1.00 20.00           O
HETATM    2  H1  HOH A   1       5.980   5.000   5.000  1.00 20.00           H
HETATM    3  H2  HOH A   1       4.750   5.950   5.000  1.00 20.00           H
END
'''
inp=iotbx.pdb.input(source_info=None,lines=pdb)
ph=inp.construct_hierarchy(sort_atoms=False)
x=ph.atoms().extract_xyz()
manager=from_qm(pdb_hierarchy=ph,charge=0,qm_engine_name='alphadynamics',
 crystal_symmetry=inp.crystal_symmetry(),file_name=str(out/'water.pdb'),
 alphadynamics_params=p.quantum.alphadynamics)
try:
 e,g=manager.target_and_gradients(x)
 scale=ase.units.mol/ase.units.kcal
 assert np.isclose(e,manager.qm_engine.energy_free*scale)
 np.testing.assert_allclose(np.array(g),-manager.qm_engine.forces*scale,rtol=1e-12,atol=1e-12)
 f=-np.array(g)/scale
 records=[]
 for eps in [.002,.001]:
  xp=x.deep_copy();xm=x.deep_copy()
  a=list(xp[1]);a[0]+=eps;xp[1]=a
  a=list(xm[1]);a[0]-=eps;xm[1]=a
  ep,_=manager.target_and_gradients(xp);em,_=manager.target_and_gradients(xm)
  fd=-(ep-em)/(2*eps*scale)
  records.append(dict(epsilon_A=eps,force_eV_A=float(f[1,0]),finite_difference_eV_A=fd,error_eV_A=abs(fd-f[1,0])))
 assert min(r['error_eV_A'] for r in records)<max(.1,abs(f[1,0])*.05),records
 result=dict(state='passed',energy_eV=e/scale,conversion=scale,finite_differences=records)
 (out/'gate.json').write_text(json.dumps(result,indent=2))
 print('GPU_GATE_PASSED',json.dumps(result),flush=True)
finally:manager.qm_engine.close()
