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
    Using RosettaScripts to build a parametric coiled-coil structure.
    CC_N_HELICES: number of helices in the coiled-coil bundle
    CC_LENGTHS: list of lengths (in residues) for each helix; if None
    CC_PHASES_DEG: list of phase offsets (in degrees) for each helix; 
    CC_INVERT: list of booleans indicating if each helix is inverted;
    CC_R0: radius of the coiled-coil (distance from bundle axis to helix axis)
    CC_OMEGA0: base omega dihedral angle (deg/residue) for helices in the bundle
    CC_Z0: rise per residue along the helix axis (in Å)
    
    RHCC (PDB: 1RHZ)
    build_coiled_coil(
    AA="ALA",
    CC_N_HELICES=4,
    CC_LENGTHS=[35, 35, 35, 35],
    CC_PHASES_DEG=[0, 90, 180, 270],
    CC_INVERT=[False]*4,
    CC_R0=9.5,
    CC_OMEGA0=+0.6,   # right-handed CC but positive twist
    CC_Z0=1.48
    )
    
    3-HELIX BARREL (PDB: 1E0Z)
    build_coiled_coil(
    AA="ALA",
    CC_N_HELICES=3,
    CC_LENGTHS=[28, 28, 28],
    CC_PHASES_DEG=[0, 120, 240],
    CC_INVERT=[False, False, False],
    CC_R0=9.0,
    CC_OMEGA0=-1.2,
    CC_Z0=1.47
    )
    """

    n = CC_N_HELICES 
    lengths = CC_LENGTHS if CC_LENGTHS else [28] * n
    phases  = CC_PHASES_DEG if CC_PHASES_DEG else [i * 360.0 / n for i in range(n)] 
    invert  = CC_INVERT if CC_INVERT else [False] * n 
    
    if not (len(lengths) == len(phases) == len(invert) == n):
        raise ValueError("CC_LENGTHS / CC_PHASES_DEG / CC_INVERT length must equal CC_N_HELICES")

    # omega0 between -3 to +3 deg/residue; z1 ~ 1.5 Å/residue
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

    pose = rosetta.core.pose.Pose()
    try:
        objs = XmlObjects.create_from_string(xml)
        mb = objs.get_mover("mb")
        mb.apply(pose)
    except Exception:
        pose = rosetta.core.pose.Pose()

    # Check if pose was built successfully
    if pose.size() == 0:
        raise RuntimeError("Failed to build coiled-coil structure with the given parameters.")
    
    if relax_after:
        _relax(pose)
        
    if dump_pdb:
        # Create output directory if it doesn't exist in the upper level of current working directory
        output_dir = Path.cwd() / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pose.dump_pdb(str(output_dir / f"{timestamp}_coiled_coil.pdb"))
    return pose


if __name__ == "__main__":
    cc = build_coiled_coil(
        CC_N_HELICES=4,
        CC_LENGTHS=[35, 35, 35, 35],
        CC_PHASES_DEG=[0, 90, 180, 270],
        CC_INVERT=[False, True, False, True],
        CC_R0=9.5,
        AA="ALA",
        CC_OMEGA0=+0.6,   # right-handed CC but positive twist
        CC_Z0=1.48,
        relax_after=True,
        dump_pdb=True
        )
    print(f"Built coiled-coil with {cc.size()} residues.")