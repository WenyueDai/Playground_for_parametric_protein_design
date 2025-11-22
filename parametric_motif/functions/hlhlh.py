from pyrosetta import rosetta
from pyrosetta.rosetta.protocols.loops import Loop, Loops, fold_tree_from_loops, add_cutpoint_variants
from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_KIC
from coiled_coil import build_coiled_coil  
from utils import _relax, _ensure_init
from pathlib import Path
import datetime

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
    Build coiled-coil with loops between helices, using:
      - fold_tree_from_loops()
      - add_cutpoint_variants()
      - LoopMover_Refine_KIC()
    Compatible with PyRosetta 2025 API.
    """

    n = CC_N_HELICES
    assert len(LOOP_LENGTHS) == n - 1

    # === Step 1: Build helices separately using MakeBundle ===
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
    helix_ranges = [
        (conf.chain_begin(i), conf.chain_end(i))
        for i in range(1, cc_pose.num_chains() + 1)
    ]

    # === Step 2: Create single-chain pose with helices + loops ===
    new_pose = rosetta.core.pose.Pose()
    rts = cc_pose.residue_type_set_for_pose()

    # Map old→new CA coordinates
    old_to_new = {}
    loop_regions = []

    def append_residue(pose, res):
        if pose.size() == 0:
            pose.append_residue_by_bond(res, True)
        else:
            pose.append_residue_by_bond(res, False)

    for i, (start, end) in enumerate(helix_ranges):
        # copy helix
        for old_r in range(start, end + 1):
            res = cc_pose.residue(old_r)
            append_residue(new_pose, res)
            old_to_new[old_r] = new_pose.size()

        # insert loop
        if i < n - 1:
            L = LOOP_LENGTHS[i]
            loop_start = new_pose.size() + 1
            loop_type = rts.name_map(LOOP_AA)
            for _ in range(L):
                loop_res = rosetta.core.conformation.ResidueFactory.create_residue(loop_type)
                append_residue(new_pose, loop_res)
            loop_end = new_pose.size()
            cut = (loop_start + loop_end) // 2
            loop_regions.append((loop_start, loop_end, cut))

    # === Step 3: Copy Cartesian coordinates for helix residues ===
    for old_r, new_r in old_to_new.items():
        old_res = cc_pose.residue(old_r)
        for atom_idx in range(1, old_res.natoms() + 1):
            old_id = rosetta.core.id.AtomID(atom_idx, old_r)
            new_id = rosetta.core.id.AtomID(atom_idx, new_r)
            new_pose.set_xyz(new_id, cc_pose.xyz(old_id))

    # === Step 4: Pre-position all loop residues (linear interpolation) ===
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

    # === Step 5: Apply PROPER FoldTree for loops ===
    loops = Loops()
    for (start, stop, cut) in loop_regions:
        loops.add_loop(Loop(start, stop, cut))

    ft = rosetta.core.kinematics.FoldTree()
    fold_tree_from_loops(new_pose, loops, ft)
    new_pose.fold_tree(ft)
    add_cutpoint_variants(new_pose)

    # === Step 6: KIC refinement (correct API) ===
    kic = LoopMover_Refine_KIC(loops)
    kic.apply(new_pose)

    # === Step 7: Relax ===
    if relax_after:
        _relax(new_pose)

    # === Step 8: Save ===
    if dump_pdb:
        outdir = Path.cwd() / "output"
        outdir.mkdir(exist_ok=True)
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_pose.dump_pdb(str(outdir / f"{tag}_HLHLH.pdb"))

    return new_pose

if __name__ == "__main__":
    _ensure_init()
    cc_hlhlh = build_coiled_coil_with_loops(
        CC_N_HELICES=4,
        CC_LENGTHS=[35, 35, 35, 35],
        CC_PHASES_DEG=[0, 90, 180, 270],
        CC_INVERT=[True, False, True, False],
        LOOP_LENGTHS=[5, 5, 5],  # 3 loops between 4 helices
        AA="ALA",
        LOOP_AA="GLY",
        CC_R0=9.5,
        CC_OMEGA0=+0.6,
        CC_Z0=1.48,
        relax_after=False,
        dump_pdb=True,
    )
    print(f"Built HLHLH coiled-coil with {cc_hlhlh.size()} residues.")
