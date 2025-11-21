import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.protocols.loops import Loop
from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import CCDLoopClosureMover
from pyrosetta.rosetta.protocols.idealize import IdealizeMover
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.core.kinematics import FoldTree, MoveMap
from pyrosetta.rosetta.utility import vector1_unsigned_long

from coiled_coil import build_coiled_coil  
from utils import _pose_from_seq, _relax, _ensure_init
from pathlib import Path
import os
import datetime

_ensure_init()

# Comvert 3-letter amino acid codes to single-letter codes
three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y"
}

def three_to_one_code(three_letter_code):
    return three_to_one.get(three_letter_code.upper(), "A")  # Default to Alanine if not found

def build_HLHLH_from_CC(
        CC_LENGTHS = [10, 10, 10],
        LOOP_LENGTHS = [5, 5],
        CC_PHASES_DEG = [0, 120, 240],
        CC_INVERT = [False, True, False],
        CC_R0 = 8.0,
        AA_HELIX = "ALA",
        AA_LOOP = "GLY",
        relax = True,
        dump_pdb = True
    ):
    """
    Build HLHLH by:
      1) Generating 3 helices via your coiled-coil parametric model
      2) Reassembling them into a single-chain H-L-H-L-H
      3) Running loop closure (CCD)
      4) Idealizing and relaxing
    """

    H1, H2, H3 = CC_LENGTHS
    L1, L2 = LOOP_LENGTHS

    # --- 1. Build 3 helices as a coiled-coil ---
    try:
        cc = build_coiled_coil(
            CC_N_HELICES=3,
            CC_LENGTHS=list(CC_LENGTHS),
            CC_PHASES_DEG=list(CC_PHASES_DEG),
            CC_INVERT=list(CC_INVERT),
            AA=AA_HELIX,
            CC_R0=CC_R0,
            relax_after=False,
            dump_pdb=False
        )
    except Exception as e:
        print("Error building coiled-coil:", e)
        return None

    # --- 2. Extract the helices individually ---
    helix_poses = []
    start = 1
    for L in CC_LENGTHS:
        sub = Pose()
        idx = vector1_unsigned_long()
        
        for k in range(start, start + L):
            idx.append(k)

        rosetta.core.pose.pdbslice(sub, cc, idx)
        
        helix_poses.append(sub)
        start += L

    # --- 3. Build final single-chain sequence ---
    
    full_seq = (
        three_to_one_code(AA_HELIX) * H1 +
        three_to_one_code(AA_LOOP)  * L1 +
        three_to_one_code(AA_HELIX) * H2 +
        three_to_one_code(AA_LOOP)  * L2 +
        three_to_one_code(AA_HELIX) * H3
    )
    pose = _pose_from_seq(full_seq)

    # --- 4. Superimpose helices back onto the new single-chain pose ---
    # Mapping:
    # H1: 1..H1
    # H2: H1+L1+1 .. H1+L1+H2
    # H3: H1+L1+H2+L2+1 .. end

    helix_ranges = [
        (1, H1),
        (H1 + L1 + 1, H1 + L1 + H2),
        (H1 + L1 + H2 + L2 + 1, H1 + L1 + H2 + L2 + H3),
    ]

    for (a, b), hpose in zip(helix_ranges, helix_poses):
        for i, j in enumerate(range(a, b+1), start=1):
            pose.set_xyz(
                rosetta.core.id.AtomID( pose.residue(j).atom_index("CA"), j ),
                hpose.residue(i).xyz("CA")
            )
            pose.set_xyz(
                rosetta.core.id.AtomID( pose.residue(j).atom_index("N"), j ),
                hpose.residue(i).xyz("N")
            )
            pose.set_xyz(
                rosetta.core.id.AtomID( pose.residue(j).atom_index("C"), j ),
                hpose.residue(i).xyz("C")
            )
            pose.set_xyz(
                rosetta.core.id.AtomID( pose.residue(j).atom_index("O"), j ),
                hpose.residue(i).xyz("O")
            )

    # --- 5. Loop closure ---
    N = pose.size()

    def close_loop(start, end):
        cut = start + (end - start)//2
        ft = FoldTree()
        ft.simple_tree(N)
        pose.fold_tree(ft)

        rosetta.core.pose.add_variant_type_to_pose_residue(pose, VariantType.CUTPOINT_LOWER, cut)
        rosetta.core.pose.add_variant_type_to_pose_residue(pose, VariantType.CUTPOINT_UPPER, cut+1)

        loop = Loop(start, end, cut)
        mm = MoveMap()
        for i in range(start, end+1):
            mm.set_bb(i, True)

        CCDLoopClosureMover(loop, mm).apply(pose)
        IdealizeMover().apply(pose)

        rosetta.core.pose.remove_variant_type_from_pose_residue(pose, VariantType.CUTPOINT_LOWER, cut)
        rosetta.core.pose.remove_variant_type_from_pose_residue(pose, VariantType.CUTPOINT_UPPER, cut+1)

    # Loop 1
    close_loop(H1+1, H1+L1)

    # Loop 2
    close_loop(H1+L1+H2+1, H1+L1+H2+L2)

    # --- 6. Relax if desired ---
    if relax:
        _relax(pose)

    if dump_pdb:
        # Dump to output folder
        output_folder = Path.cwd() / "output"
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pose.dump_pdb(str(output_folder / f"{timestamp}_hlhlh.pdb"))

    return pose

if __name__ == "__main__":
    hlhlh = build_HLHLH_from_CC(
        CC_LENGTHS = [10, 10, 10],
        LOOP_LENGTHS = [5, 5],
        CC_PHASES_DEG = [0, 120, 240],
        CC_INVERT = [False, True, False],
        CC_R0 = 8.0,
        AA_HELIX = "ALA",
        AA_LOOP = "GLY",
        relax = False,
        dump_pdb = True
    )