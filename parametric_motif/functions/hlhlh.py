'''
Comparing to coiled coil, building HLHLH with loops involves additional steps:
1. loop
loop(start=30, stop=40, cut=35)
2. loops
loops = Loops()
loops.add_loop(Loop(30, 40, 35))
3. fold_tree_from_loops(pose, loops, fold_tree)
It provide jump at cutpoint, and set upstream/downstream anchor, build fold tree.
4. add_cutpoint_variants(pose)
Add CUTPOINT_LOWER and CUTPOINT_UPPER variants at cutpoints in pose.
So that Rosetta know the chain can temporarily break at cutpoints, to allow KIC/CCD kinematics.
But after loop modeling, we need to manually clean up the cutpoints (remove variants, rebuild peptide bond).

'''

from pyrosetta import rosetta
from pyrosetta.rosetta.protocols.loops import (
    Loop, Loops, fold_tree_from_loops, add_cutpoint_variants
)
from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC
from coiled_coil import build_coiled_coil
from utils import _relax, _ensure_init, _append_residue
from pathlib import Path
import datetime
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def build_coiled_coil_with_loops(
        CC_N_HELICES: int,
        CC_LENGTHS: list,
        CC_PHASES_DEG: list,
        CC_INVERT: list,
        LOOP_LENGTHS: list,
        AA: str = "ALA",
        LOOP_AA: str = "GLY",
        CC_R0: float = 8.0,
        CC_OMEGA0: float = -1.0,
        CC_Z0: float = 1.5,
        relax_after: bool = False,
        dump_pdb: bool = True
):
    """
    Build HLHLH but do NOT try to fix cutpoints inside Rosetta.
    Instead, repair loop continuity with PDBFixer afterwards.
    """

    n = CC_N_HELICES
    assert len(LOOP_LENGTHS) == n - 1

    # === STEP 1: Build individual helices using MakeBundle ===
    # This will create a multi-chain pose (chain per helix)
    cc_pose = build_coiled_coil(
        CC_N_HELICES=n,
        CC_LENGTHS=CC_LENGTHS,
        CC_PHASES_DEG=CC_PHASES_DEG,
        CC_INVERT=CC_INVERT,
        AA=AA,
        CC_R0=CC_R0,
        CC_OMEGA0=CC_OMEGA0,
        CC_Z0=CC_Z0,
        relax_after=False,
        dump_pdb=False,
    )

    conf = cc_pose.conformation()
    
    # Get helix range for each chain
    # cc_pose.num_chains() = 4
    # conf.chain_begin(1) = 1
    # conf.chain_end(1) = 10
    
    helix_ranges = [
        (conf.chain_begin(i), conf.chain_end(i))
        for i in range(1, cc_pose.num_chains() + 1)
    ]

    # === STEP 2: Merge helices + insert loops ===
    # We will do the following steps:
    # apend helix residues to new pose
    # add loop residues to new pose
    # set initial coordinates for loops
    # set up FoldTree
    # perform KIC refinement
    # use PDBFixer to repair chain continuity
    new_pose = rosetta.core.pose.Pose()
    # We need to make sure the residue type for the newly created loop will be the same with cc_pose
    rts = cc_pose.residue_type_set_for_pose()
    # Mapping from old residue index to new residue index
    old_to_new = {}
    # Mapping from old loop residue index to new loop residue index
    loop_regions = []

    for i, (start, end) in enumerate(helix_ranges):
        # Copy helix residues
        for old_r in range(start, end + 1):
            # Since the coiled coil made from MakeBundle has been full-atomed, so the N, C terminus have been added with
            # NtermProteinFull and CtermProteinFull variants. We need to remove them before adding cutpoint variants,
            res = cc_pose.residue(old_r)
            res_clean = rosetta.core.conformation.ResidueFactory.create_residue(
                rosetta.core.pose.get_restype_for_pose(new_pose, res.name3())
            )
            _append_residue(new_pose, res_clean)
            old_to_new[old_r] = new_pose.size()

        # Insert loop
        if i < n - 1:
            L = LOOP_LENGTHS[i]
            loop_start = new_pose.size() + 1
            loop_type = rts.name_map(LOOP_AA)
            for _ in range(L):
                loop_res = rosetta.core.conformation.ResidueFactory.create_residue(loop_type)
                _append_residue(new_pose, loop_res)
            loop_end = new_pose.size()
            cut = (loop_start + loop_end) // 2
            loop_regions.append((loop_start, loop_end, cut))

    # === STEP 3: Copy helix coordinates (safe version) ===
    for old_r, new_r in old_to_new.items():
        old_res = cc_pose.residue(old_r)
        new_res = new_pose.residue(new_r)

        # Only iterate over atoms in new_res to avoid out-of-bound errors
        for i in range(1, new_res.natoms() + 1):
            atom_name = new_res.atom_name(i).strip()  # Rosetta name has spaces
            if not old_res.has(atom_name):
                # Some atoms (especially H) may not exist in old_res, skip them
                continue

            old_idx = old_res.atom_index(atom_name)
            old_id = rosetta.core.id.AtomID(old_idx, old_r)
            new_id = rosetta.core.id.AtomID(i, new_r)

            new_pose.set_xyz(new_id, cc_pose.xyz(old_id))

    # === STEP 4: Simple straight-line loop placement ===
    for loop_start, loop_end, cut in loop_regions:
        up = loop_start - 1
        dn = loop_end + 1
        up_CA = new_pose.residue(up).xyz("CA")
        dn_CA = new_pose.residue(dn).xyz("CA")

        L = loop_end - loop_start + 1
        dx = dn_CA.x - up_CA.x
        dy = dn_CA.y - up_CA.y
        dz = dn_CA.z - up_CA.z

        for i, resi in enumerate(range(loop_start, loop_end + 1), start=1):
            t = float(i) / float(L + 1)
            target = rosetta.numeric.xyzVector_double_t(
                up_CA.x + dx * t,
                up_CA.y + dy * t,
                up_CA.z + dz * t,
            )
            cur_CA = new_pose.residue(resi).xyz("CA")
            shift = target - cur_CA

            for atom_idx in range(1, new_pose.residue(resi).natoms() + 1):
                aid = rosetta.core.id.AtomID(atom_idx, resi)
                new_pose.set_xyz(aid, new_pose.xyz(aid) + shift)

    # === STEP 5: FoldTree for KIC ===
    loops = Loops()
    for (start, stop, cut) in loop_regions:
        loops.add_loop(Loop(start, stop, cut))
        print(f"Loop defined: start={start}, stop={stop}, cut={cut}")

    ft = rosetta.core.kinematics.FoldTree()
    fold_tree_from_loops(new_pose, loops, ft)
    new_pose.fold_tree(ft)

    add_cutpoint_variants(new_pose)

    # === STEP 6: KIC refinement ===
    kic = LoopMover_Refine_KIC(loops)
    kic.apply(new_pose)

    # === STEP 7: optional relax ===
    if relax_after:
        _relax(new_pose)

    # === STEP 8: Output raw PDB ===
    if dump_pdb:
        outdir = Path.cwd() / "output"
        outdir.mkdir(exist_ok=True)
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = outdir / f"{tag}_HLHLH_raw.pdb"
        new_pose.dump_pdb(str(raw_path))

        print(f"[Rosetta] wrote: {raw_path}")

        # === STEP 9: FIX DISCONTINUOUS LOOPS USING PDBFIXER ===
        print("[PDBFixer] repairing chain continuity...")

        fixer = PDBFixer(filename=str(raw_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingHydrogens()

        fixed_path = outdir / f"{tag}_HLHLH_fixed.pdb"
        with open(fixed_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        print(f"[PDBFixer] wrote fixed structure: {fixed_path}")

    return new_pose


if __name__ == "__main__":
    _ensure_init()
    build_coiled_coil_with_loops(
        CC_N_HELICES=4,
        CC_LENGTHS=[15, 15, 15, 15],
        CC_PHASES_DEG=[0, 90, 180, 270],
        CC_INVERT=[True, False, True, False],
        LOOP_LENGTHS=[7, 7, 7],
        AA="ALA",
        LOOP_AA="GLY",
        CC_R0=7.0,
        CC_OMEGA0=+0.6,
        CC_Z0=1.48,
        relax_after=False,
        dump_pdb=True,
    )
