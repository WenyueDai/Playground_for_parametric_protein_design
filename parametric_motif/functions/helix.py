import pyrosetta
from utils import _pose_from_seq, _set_phi_psi_omega, _relax, _ensure_init
import datetime
import os
from pathlib import Path

_ensure_init()

PHI_ALPHA, PSI_ALPHA = -57.8, -47.0

def build_alpha(phi=PHI_ALPHA, psi=PSI_ALPHA, length=12, aa="A", relax_after=True, dump_pdb=False):
    pose = _pose_from_seq(aa * length)
    _set_phi_psi_omega(pose, 1, length, phi, psi, 180.0)
    if relax_after:
        _relax(pose)
    if dump_pdb:
        output_folder = Path.cwd() / "output"
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_helix.pdb"
        pose.dump_pdb(str(output_folder / filename))

    return pose

if __name__ == "__main__":
    helix = build_alpha(-57.8, -47.0, 12)