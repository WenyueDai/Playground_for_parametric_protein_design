"""
Note: MakeBundle's ResidueTypeSet is centroid or parametric only.
Cannot build fullatom pose directly.

Solution:
After building coiled-coil, convert it to fa_standard full-atom pose
by PDB round-trip (safest & most robust way).
"""

import math
import numpy as np
import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pathlib import Path
import datetime

from utils import _rotmat, _transform_pose_inplace, _relax, _ensure_init
from helix import build_alpha, PHI_ALPHA, PSI_ALPHA

_ensure_init()

def _convert_to_fa_standard(pose):
    """
    Safest possible method to convert a non-fullatom pose to full-atom:
    PDB round-trip.

    This avoids all variant / residue mismatch issues.
    """
    tmp = "tmp_fullatom_convert.pdb"
    pose.dump_pdb(tmp)
    full_pose = rosetta.core.import_pose.pose_from_file(tmp)
    return full_pose


def build_coiled_coil(
                      CC_N_HELICES: int,
                      CC_LENGTHS: list =None,
                      CC_PHASES_DEG: list =None,
                      CC_INVERT: list =None, 
                      AA: str ="ALA",
                      CC_R0: float =8.0,
                      CC_OMEGA0: float =-1.0,
                      CC_Z0: float =1.5,
                      relax_after: bool = True,
                      dump_pdb: bool = True):
    """
    Build a parametric coiled-coil using MakeBundle + convert to full-atom.
    """

    n = CC_N_HELICES 
    lengths = CC_LENGTHS if CC_LENGTHS else [28] * n
    phases  = CC_PHASES_DEG if CC_PHASES_DEG else [i * 360.0 / n for i in range(n)] 
    invert  = CC_INVERT if CC_INVERT else [False] * n 
    
    if not (len(lengths) == len(phases) == len(invert) == n):
        raise ValueError("CC_LENGTHS / CC_PHASES_DEG / CC_INVERT length must equal CC_N_HELICES")

    # --- Build MakeBundle XML ---
    helix_lines = []
    for L, ph, inv in zip(lengths, phases, invert):
        inv_str = "true" if inv else "false"
        helix_lines.append(
            f'<Helix helix_length="{L}" residue_name="{AA}" '
            f'r0="{CC_R0:.3f}" omega0="{CC_OMEGA0:.3f}" delta_omega0="{ph:.3f}" '
            f'z1="{CC_Z0:.3f}" invert="{inv_str}" />'
        )
    helix_block = "\n        ".join(helix_lines)

    xml = f"""
    <ROSETTASCRIPTS>
    <MOVERS>
        <MakeBundle name="mb" use_degrees="true" reset="true"
                    set_bondlengths="true" set_bondangles="true" set_dihedrals="true">
            {helix_block}
        </MakeBundle>
    </MOVERS>
    <PROTOCOLS>
        <Add mover_name="mb"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """.strip()

    # --- Build Pose (MakeBundle) ---
    pose = rosetta.core.pose.Pose()
    try:
        objs = XmlObjects.create_from_string(xml)
        mb = objs.get_mover("mb")
        mb.apply(pose)
    except Exception:
        pose = rosetta.core.pose.Pose()

    if pose.size() == 0:
        raise RuntimeError("Failed to build coiled-coil structure with the given parameters.")

    # --- Convert to full-atom (key step!) ---
    pose = _convert_to_fa_standard(pose)

    # --- Optional relax (full-atom relax) ---
    if relax_after:
        _relax(pose)

    # --- Dump final full-atom coiled-coil ---
    if dump_pdb:
        output_dir = Path.cwd() / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pose.dump_pdb(str(output_dir / f"{timestamp}_coiled_coil_fa.pdb"))

    return pose


if __name__ == "__main__":
    cc = build_coiled_coil(
        CC_N_HELICES=4,
        CC_LENGTHS=[35, 35, 35, 35],
        CC_PHASES_DEG=[0, 90, 180, 270],
        CC_INVERT=[False, True, False, True],
        CC_R0=9.5,
        AA="ALA",
        CC_OMEGA0=+0.6,
        CC_Z0=1.48,
        relax_after=False,
        dump_pdb=True
    )
    print(f"Built FULL-ATOM coiled-coil with {cc.size()} residues.")
    
    # clean up temporary file
    tmp_file = Path("tmp_fullatom_convert.pdb")
    if tmp_file.exists():
        tmp_file.unlink()
        
