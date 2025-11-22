from pyrosetta import rosetta, init
from pyrosetta import Pose, pose_from_pdb
from pyrosetta.rosetta.core import conformation, pose
from utils import _append_residue, _ensure_init
import numpy as np

_ensure_init()

# ------------------------------
# xyz → numpy
# ------------------------------
def vec_to_np(v):
    return np.array([v.x, v.y, v.z])

# ------------------------------
# Kabsch alignment
# ------------------------------
def superimpose_coords(X, Y):
    # Center the points
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    # Compute covariance matrix
    C = np.dot(Xc.T, Yc)
    # Singular Value Decomposition
    V, S, Wt = np.linalg.svd(C)
    # Compute rotation matrix
    R = np.dot(Wt.T, V.T)
    # Special reflection case
    if np.linalg.det(R) < 0:
        Wt[-1, :] *= -1
        R = np.dot(Wt.T, V.T)
    # Compute translation
    t = Y.mean(axis=0) - np.dot(X.mean(axis=0), R)
    return R, t

# ------------------------------
# Apply R, t to a Pose
# ------------------------------
def transform_pose(p, R, t):
    for i in range(1, p.size() + 1):
        res = p.residue(i)
        for j in range(1, res.natoms() + 1):
            old = p.xyz(rosetta.core.id.AtomID(j, i))
            xyz = np.array([old.x, old.y, old.z])
            new_xyz = np.dot(xyz, R) + t
            p.set_xyz(
                rosetta.core.id.AtomID(j, i),
                rosetta.numeric.xyzVector_double_t(*new_xyz)
            )

# ------------------------------
# Copy coordinates by atom name (safe)
# ------------------------------
def copy_coords_by_name(src_pose, src_resi, dst_pose, dst_resi):
    src_res = src_pose.residue(src_resi)
    dst_res = dst_pose.residue(dst_resi)
    for j in range(1, dst_res.natoms() + 1):
        aname = dst_res.atom_name(j).strip()
        if not src_res.has(aname):
            continue
        src_idx = src_res.atom_index(aname)
        src_xyz = src_pose.xyz(rosetta.core.id.AtomID(src_idx, src_resi))
        dst_pose.set_xyz(
            rosetta.core.id.AtomID(j, dst_resi),
            src_xyz
        )

# ------------------------------
# Main: graft motif with C/N alignment + optional loops
# ------------------------------
def graft_motif(
        big_pdb,
        motif_pdb,
        loop_start,
        loop_end,
        N_linker_len: int = 0,
        C_linker_len: int = 0,
        loop_AA: str = "GLY"
):
    """
    big_pdb: host protein
    motif_pdb: fragment to insert
    loop_start, loop_end: host loop [loop_start, loop_end] to replace
    N_linker_len, C_linker_len: optional loops on both sides of motif
    """

    big = pose_from_pdb(big_pdb)
    motif = pose_from_pdb(motif_pdb)

    print(f"Host size = {big.size()}, motif size = {motif.size()}")
    print(f"Replace host[{loop_start}:{loop_end}] with motif, "
          f"N_linker={N_linker_len}, C_linker={C_linker_len}")

    # ---- 1) Build alignment based on C_up & N_down and motif N/C ----
    up_res  = big.residue(loop_start - 1)
    down_res = big.residue(loop_end + 1)

    C_up   = vec_to_np(up_res.xyz("C"))
    N_down = vec_to_np(down_res.xyz("N"))

    N_motif = vec_to_np(motif.residue(1).xyz("N"))
    C_motif = vec_to_np(motif.residue(motif.size()).xyz("C"))

    X = np.vstack([N_motif, C_motif])   # motif endpoints
    Y = np.vstack([C_up, N_down])       # host endpoints

    R, t = superimpose_coords(X, Y)
    transform_pose(motif, R, t)

    # ---- 2) Build new_pose sequence: hostN – Nlinker – motif – Clinker – hostC ----
    new = Pose()
    rts = big.residue_type_set_for_pose()

    old_big_to_new = {}
    old_motif_to_new = {}

    # host N-part
    for i in range(1, loop_start):
        res = big.residue(i)
        clean = conformation.ResidueFactory.create_residue(
            pose.get_restype_for_pose(new, res.name3())
        )
        _append_residue(new, clean)
        old_big_to_new[i] = new.size()

    # N-linker residues
    loop_type = rts.name_map(loop_AA)
    N_linker_start = None
    N_linker_end = None
    if N_linker_len > 0:
        N_linker_start = new.size() + 1
        for _ in range(N_linker_len):
            link_res = conformation.ResidueFactory.create_residue(loop_type)
            _append_residue(new, link_res)
        N_linker_end = new.size()

    # motif residues
    motif_start_new = new.size() + 1
    for i in range(1, motif.size() + 1):
        res = motif.residue(i)
        clean = conformation.ResidueFactory.create_residue(
            pose.get_restype_for_pose(new, res.name3())
        )
        _append_residue(new, clean)
        old_motif_to_new[i] = new.size()
    motif_end_new = new.size()

    # C-linker residues
    C_linker_start = None
    C_linker_end = None
    if C_linker_len > 0:
        C_linker_start = new.size() + 1
        for _ in range(C_linker_len):
            link_res = conformation.ResidueFactory.create_residue(loop_type)
            _append_residue(new, link_res)
        C_linker_end = new.size()

    # host C-part
    for i in range(loop_end + 1, big.size() + 1):
        res = big.residue(i)
        clean = conformation.ResidueFactory.create_residue(
            pose.get_restype_for_pose(new, res.name3())
        )
        _append_residue(new, clean)
        old_big_to_new[i] = new.size()

    # ---- 3) Copy coordinates for host parts & motif (真正用上对齐后的 motif 坐标) ----
    # host coords
    for old_i, new_i in old_big_to_new.items():
        copy_coords_by_name(big, old_i, new, new_i)

    # motif coords
    for old_i, new_i in old_motif_to_new.items():
        copy_coords_by_name(motif, old_i, new, new_i)

    # ---- 4) Initialize linker coordinates by CA interpolation (straight line) ----
    def interp_linker(start_resi, end_resi, first_link, last_link):
        if first_link is None:
            return
        up_CA = new.residue(start_resi).xyz("CA")
        dn_CA = new.residue(end_resi).xyz("CA")
        up = np.array([up_CA.x, up_CA.y, up_CA.z])
        dn = np.array([dn_CA.x, dn_CA.y, dn_CA.z])

        L = last_link - first_link + 1
        dx, dy, dz = dn - up

        for k, resi in enumerate(range(first_link, last_link + 1), start=1):
            t = float(k) / float(L + 1)
            target = np.array([
                up[0] + dx * t,
                up[1] + dy * t,
                up[2] + dz * t,
            ])
            cur_CA = new.residue(resi).xyz("CA")
            cur = np.array([cur_CA.x, cur_CA.y, cur_CA.z])
            shift = target - cur

            for atom_idx in range(1, new.residue(resi).natoms() + 1):
                aid = rosetta.core.id.AtomID(atom_idx, resi)
                old_xyz = new.xyz(aid)
                moved = np.array([old_xyz.x, old_xyz.y, old_xyz.z]) + shift
                new.set_xyz(
                    aid,
                    rosetta.numeric.xyzVector_double_t(*moved)
                )

    # N-linker: between host_up and motif_start
    if N_linker_len > 0:
        up_new = old_big_to_new[loop_start - 1]
        interp_linker(up_new, motif_start_new, N_linker_start, N_linker_end)

    # C-linker: between motif_end and host_down
    if C_linker_len > 0:
        down_new = old_big_to_new[loop_end + 1]
        interp_linker(motif_end_new, down_new, C_linker_start, C_linker_end)

    # ---- 5) (optionally) you can now define Loops and run KIC here ----
    # 例如：
    # loops = Loops()
    # if N_linker_len > 0:
    #     loops.add_loop(Loop(up_new + 1, motif_start_new - 1, (up_new + motif_start_new)//2))
    # if C_linker_len > 0:
    #     loops.add_loop(Loop(motif_end_new + 1, down_new - 1, (motif_end_new + down_new)//2))
    # kic = LoopMover_Refine_KIC(loops)
    # kic.apply(new)

    # ---- 6) 输出 ----
    new.dump_pdb("graft_CN_aligned_with_loops.pdb")
    print("Saved: graft_CN_aligned_with_loops.pdb")
    return new


if __name__ == "__main__":
    graft_motif(
        "/home/eva/20251031_parametric_design_playground/input/1CRN.pdb",
        "/home/eva/20251031_parametric_design_playground/input/hth_motif.pdb",
        loop_start=20,
        loop_end=25,
        N_linker_len=2,
        C_linker_len=2,
        loop_AA="GLY",
    )
