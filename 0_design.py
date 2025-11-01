"""
  conda activate pyrosetta
  python sym_helix_extended_pyrosetta.py config_helix.json
  outs/helix_output_CA_only__symN-5__R-18.0__tilt-15.0__roll-20.0.pdb
  
# What this script does:
    - Build ideal alpha-helix or load user PDB as seed
    - Apply rigid-body transforms to align helix axis
    1. local alignment / roll
    2. Cn symmetry placement (+ tilt)
    3. Dn dihedral extension 
    4. Ring stacking / expansion
    - Output multi-chain CA-only for downstream visualization / MPNN
"""

import os, math, json, string
from typing import List, Tuple
import numpy as np
import pyrosetta
from pyrosetta import rosetta

# =========================================================
# Default parameters
# =========================================================
# For demo: single helix
helix_len = 28 # number of residues in the helix
phi_deg = -57.8 # typical alpha-helix phi angle
psi_deg = -47.0 # typical alpha-helix psi angle
omega_deg = 180.0 # typical alpha-helix omega angle
helix_axis = "0,0,1" # direction vector of helix axis
helix_start = "0,0,0" # starting point of helix

# --- Otherwise, use your own pdb ---
input_pdb_path = ""      # empty string means "disabled"
input_chain = ""         # "A"; empty means "use whole pose or infer"
input_pdb_range = ""     # "5-42" in PDB numbering within the chosen chain; empty means "full chain"
# --- Imported-PDB axis controls (so imported PDBs can be aimed like ideal helices) ---
input_axis_mode = "auto_pca"      # "auto_pca" | "auto_ends" | "explicit"
input_axis_hint = "0,0,1"         # used only if input_axis_mode == "explicit"
input_target_axis = "0,0,1"       # where to point the imported fragment’s long axis
input_pre_roll_deg = 0.0          # roll around the (aligned) target axis
input_center_to = "none"          # "none" | "sym_center" | "x,y,z"
input_flip_if_opposite = True     # flip source axis if ~180° to target (for minimal rotation)


# --- Local alignment parameters ---
local_align_axis = "1,0,0" # e.g., "1,0,0"
local_align_deg = 0.0 # e.g., 30.0, rotate about local_align_axis
local_roll_deg = 0.0 # e.g., 15.0, similar to the helical rotation in crick parameters


# --- Cn symmetry parameters ---
sym_n = 0 # number of symmetry units; 0 means no symmetry
sym_axis = "0,0,1" # symmetry axis direction, default Z
sym_center = "0,0,0" # symmetry center point
sym_radius = 15.0 # radius from symmetry axis to helix center
sym_start_angle = 0.0 # starting angle for symmetry placement (azimuth of the first helix/pdb)
global_tilt_deg = 0.0 # tilt each helix away from sym_axis by this many degrees (lean in/out)

# --- Dihedral extension (C_n -> D_n) ---
dihedral_enable = False # create partner ring by a 180° rotation about dihedral_axis through sym_center
dihedral_axis = "1,0,0"         # a C2 axis perpendicular to sym_axis
dihedral_post_shift_z   = 0.0      # translate/shift along sym_axis (Å)
dihedral_post_delta_r   = 0.0      # push outward radially from sym_center (Å)
dihedral_twist_deg = 0.0        # extra twist about sym_axis immediately after reflection
# Not commonly used but nice to have
dihedral_post_yaw_deg   = 0.0      # rotate around the dihedral_axis itself

# --- Ring stacking / expansion ---
ring_stack_copies = 0           # number of extra rings to add
ring_delta_radius = 0.0         # outward shift per ring (Å)
ring_delta_z = 0.0              # axial shift per ring (Å)
ring_delta_twist_deg = 0.0      # extra azimuth per ring (deg)

resname = "ALA"     # residue name to display in PDB
out_pdb = "outs/helix_output_CA_only.pdb" # output PDB file path

# =========================================================
# Basic 3D operations
# =========================================================

def unit(v): 
    # 3D vector operations
    # unit([3, 4, 0]) -> array([0.6, 0.8, 0. ])
    # build unit axes for stable rotation and projections
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    return v if n==0 else v/n

def rotmat_axis_angle(axis,angle_deg):
    """
    Purpose: build a 3x3 rotation matrix that rotates vectors by angle_deg degrees about 
    the given axis (right hand rule). Method: implements Rodrigues' rotation formula.
    - Every tilt/roll/twist in the script use this function to build rotation matrices. (A 3x3 matrix
    that defines how to rotate a vector in 3D space.)
    """
    # normalize the axis vector
    a=unit(np.array(axis,float))
    # compute degrees to radians, cos, sin
    th=math.radians(angle_deg)
    # pre-compute cos, sin
    c,s=math.cos(th),math.sin(th)
    x,y,z=a
    # Rodrigues' rotation formula 
    # # assume rotmat_axis_angle is defined in the module
    # R = rotmat_axis_angle([0,0,1], 90)            # rotate 90° about +Z
    # v = np.array([1.0, 0.0, 0.0])
    # v_rot = R @ v
    # print(np.round(v_rot, 6))  # -> [0.0, 1.0, 0.0] (approximately)
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])

def apply_rt(X,R=None,t=None):
    # Apply rotation and translation to 3D points
    # X: Nx3 array of points
    # R: 3x3 rotation matrix
    # t: 3D translation vector
    # X = np.array([[1, 0, 0]]) # one point at (1,0,0)
    # R = np.array([[0,-1,0],[1,0,0],[0,0,1]]) # rotate 90° around Z
    # t = np.array([0, 0, 1]) # shift up by 1 Å
    # print(X @ R.T + t) [[0, 1, 1]]
    X=np.asarray(X,float)
    R=np.eye(3) if R is None else R
    t=np.zeros(3) if t is None else t
    return (X@R.T)+t

def v3(s):
    # Convert comma-separated string (from json or global) to 3D vector
    return np.array([float(x) for x in s.split(",")],float)

def load_config_json(path:str):
    if not os.path.exists(path): 
        print(f"[WARN] JSON config {path} not exist, use default."); return {}
    with open(path,"r") as f: 
        data=json.load(f)
    print(f"[OK] Read JSON config: {path}")
    return data

# =========================================================
# PyRosetta 操作
# =========================================================
def init_pyrosetta(): 
    # mute start messages
    pyrosetta.init("-mute all")

def build_ideal_helix_pose(n_res:int,aa="A"):
    seq = aa * n_res
    # Set the full-atom residue types (not centroid or coarse-grained)
    cm = rosetta.core.chemical.ChemicalManager.get_instance()
    fa_rts = cm.residue_type_set("fa_standard")
    # Make empty pose and build from sequence
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, fa_rts)
    # Set ideal helix backbone torsions
    for i in range(1, n_res + 1):
        pose.set_phi(i, phi_deg)
        pose.set_psi(i, psi_deg)
        pose.set_omega(i, omega_deg)
    return pose

def pose_center_CA(pose):
    # Compute center of mass of CA atoms
    xs = []
    # Iterate over all residues, take the coordinates of CA atoms
    for i in range(1, pose.size() + 1):
        aid = rosetta.core.id.AtomID(pose.residue(i).atom_index("CA"), i)
        v = pose.xyz(aid)
        xs.append([v.x, v.y, v.z])
    # Return mean position for all CA atoms
    return np.mean(np.asarray(xs, dtype=float), axis=0)

def transform_pose(pose,R=None,t=None,about=None):
    # Apply rotation R and translation t to the pose about point 'about' default as origin
    R=np.eye(3) if R is None else R
    t=np.zeros(3) if t is None else t
    about=np.zeros(3) if about is None else about
    # For all the residue in pose
    for i in range(1,pose.size()+1):
        rsd=pose.residue(i)
        # For all the atom in residue
        for j in range(1,rsd.natoms()+1):
            # get atom id
            aid=rosetta.core.id.AtomID(j,i)
            # get atom xyz
            xyz=pose.xyz(aid)
            # apply rt to do the rotation and translation
            p=np.array([xyz.x,xyz.y,xyz.z])
            # p-about: shift the coordinate system to 'about' point
            # apply rotation R
            # move it back to the original coordinate system and add translation t
            # so that you can build the initial Cn around origin but rotate in a different center
            p2=apply_rt(p-about,R,about+t)
            # set new xyz to pose
            pose.set_xyz(aid,rosetta.numeric.xyzVector_double_t(*p2))

def extract_CA_coords(pose):
    # Extract CA atom coordinates as Nx3 numpy array
    # It is simple but very useful when we need the geomtry of the pose
    # e.g., compute helix axis, center, aligning, etc.
    arr=[]
    for i in range(1,pose.size()+1):
        a=pose.xyz(rosetta.core.id.AtomID(pose.residue(i).atom_index("CA"),i))
        arr.append([a.x,a.y,a.z])
    return np.array(arr)

def principal_axis_of_pose(pose):
    """Return the first principal component (unit vector) of CA coords."""
    X = extract_CA_coords(pose)
    # centered coordinates
    Xc = X - X.mean(axis=0)
    # NumPy’s SVD (Singular Value Decomposition) computes
    # U:an N X 3 orthonormal matrix - left singular vectors (direction of variation in data)
    # S: diagonal matrix of singular values (square roots of variance)
    # Vt: 3 X 3 orthonormal matrix - right singular vectors (directions in coordinate space, same as eigenvectors of C), rows are principal directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]  # principal direction (row vector)
    return unit(pc1)

def ends_axis_of_pose(pose):
    """Return unit vector from first CA to last CA (fallback if <2 residues)."""
    ca = extract_CA_coords(pose)
    if len(ca) >= 2:
        return unit(ca[-1] - ca[0])
    return np.array([0.0, 0.0, 1.0], float)

def align_pose_axis_to(pose, source_axis, target_axis, about_point):
    """Rigidly rotate pose so source_axis aligns with target_axis, pivoting at about_point."""
    a = unit(np.asarray(source_axis, float))
    b = unit(np.asarray(target_axis, float))
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        # parallel or anti-parallel
        if c > 0:  # already aligned
            return
        # 180°: rotate around any axis perpendicular to a
        # pick a stable axis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, a)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(a, ref))
        R = rotmat_axis_angle(axis, 180.0)
    else:
        axis = v / s
        angle = math.degrees(math.atan2(s, c))
        R = rotmat_axis_angle(axis, angle)
    transform_pose(pose, R=R, about=np.asarray(about_point, float))


def load_seed_pose_from_pdb(pdb_path: str, chain: str = "", pdb_range: str = "") -> rosetta.core.pose.Pose:
    """
    Load a PDB into a Pose. Optionally slice a single continuous segment by PDB chain/id range.
    - chain: PDB chain ID like "A". If empty, and a range is given, we try to infer the first residue's chain.
    - pdb_range: e.g., "5-42". If empty, take the whole (chain) pose.
    Returns a Pose; no torsions are modified. You can rigid-body it downstream as usual.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"[ERR] input_pdb_path not found: {pdb_path}")

    pose = rosetta.core.pose.Pose()
    rosetta.core.import_pose.pose_from_file(pose, pdb_path)

    # If no slicing requested, just return the whole pose
    if (not chain) and (not pdb_range):
        return pose

    pinfo = pose.pdb_info()
    if pinfo is None:
        # Shouldn't happen for standard PDBs, but be defensive
        return pose

    # Infer chain if needed when a range is provided
    start_i = 1
    end_i = pose.size()

    if pdb_range:
        try:
            start_pdb, end_pdb = pdb_range.replace(" ", "").split("-")
            start_pdb, end_pdb = int(start_pdb), int(end_pdb)
        except Exception:
            raise ValueError(f"[ERR] input_pdb_range should look like '5-42', got: {pdb_range}")

        # If chain not given, try to take chain of the first residue in the pose
        use_chain = chain if chain else pinfo.chain(1)

        # Map PDB numbering to pose indices; ' ' = blank insertion code
        start_i = pinfo.pdb2pose(use_chain, start_pdb, ' ')
        end_i   = pinfo.pdb2pose(use_chain, end_pdb,   ' ')
        if start_i <= 0 or end_i <= 0:
            raise ValueError(f"[ERR] Could not map PDB range {use_chain}:{pdb_range} to pose indices.")

        if start_i > end_i:
            start_i, end_i = end_i, start_i

        sub = rosetta.core.pose.Pose(pose, start_i, end_i)
        return sub

    # If only chain is provided (no range), extract that full chain span
    if chain:
        # Find first/last residue with that chain in PDBInfo
        idxs = [i for i in range(1, pose.size()+1) if pinfo.chain(i) == chain]
        if not idxs:
            raise ValueError(f"[ERR] Chain '{chain}' not found in PDB.")
        start_i, end_i = min(idxs), max(idxs)
        sub = rosetta.core.pose.Pose(pose, start_i, end_i)
        return sub

    return pose

# =========================================================
# Cn
# =========================================================
def c_n_symmetry_place_pose(seed,n,sym_axis_vec,center_vec,radius_xy,
                            start_angle_deg,global_tilt_deg,local_roll_deg):
    """_summary_
        Generate Cn symmetry copies of a seed Pose (e.g. helix or import pdb)
        arranged evenly around a defined symmetry axis. Each copy can have a
        local roll (twist about the axis) and/or a global tilt (lean away from
        the axis) applied to it.

    Args:
        seed (rosetta.core.pose.Pose): 
            The input Pose used as the repeating unit for symmetry placement.

        n (int): 
            Number of symmetric subunits (Cn order). If n=0, returns empty list
            so no symmetry is applied.

        sym_axis_vec (array-like or list of float): 
            Direction vector of the symmetry axis (e.g., [0,0,1] for Z-axis).

        center_vec (array-like or list of float): 
            3D coordinates of the symmetry axis origin (center of rotation, default: [0,0,0]).

        radius_xy (float): 
            Distance (Å) from the symmetry axis to the center of each subunit.

        start_angle_deg (float): 
            Starting angular offset (degrees) for the first subunit around the axis.

        global_tilt_deg (float): 
            Tilt angle (degrees) of each subunit away from the radial plane,
            producing “flower” or “umbrella”-like arrangements.

        local_roll_deg (float): 
            Additional rotation (degrees) applied to each subunit about the 
            symmetry axis before placement — similar to a helical rotation.

    Returns:
        list[rosetta.core.pose.Pose]: 
            List of Pose objects corresponding to the n placed subunits in the 
            generated Cₙ symmetric ring.
    """
    # Place n copies of seed pose in Cn symmetry
    sym_axis=unit(np.array(sym_axis_vec,float))
    # Compute orthonormal basis
    center=np.array(center_vec,float)
    # find a vector not parallel to sym_axis
    tmp=np.array([1,0,0]) if abs(np.dot([1,0,0],sym_axis))<0.9 else np.array([0,1,0])
    # Compute orthonormal basis
    x_dir=unit(np.cross(sym_axis,tmp))
    y_dir=unit(np.cross(sym_axis,x_dir))
    seed_center=pose_center_CA(seed)
    chains=[]
    base=seed.clone()
    # apply local roll to base
    if abs(local_roll_deg)>1e-6:
        # Compute rotation matrix for local roll
        R_roll=rotmat_axis_angle(sym_axis,local_roll_deg)
        # apply rotation about seed center
        transform_pose(base,R=R_roll,about=seed_center)
    for k in range(n):
        ang=start_angle_deg+360*k/n
        radial=math.cos(math.radians(ang))*x_dir+math.sin(math.radians(ang))*y_dir
        # compute placement point
        place=center+radius_xy*radial
        P=base.clone()
        c0=pose_center_CA(P)
        # apply translation to place
        transform_pose(P,t=place-c0)
        tangent=unit(np.cross(sym_axis,radial))
        # apply global tilt
        if abs(global_tilt_deg)>1e-6:
            R_tilt=rotmat_axis_angle(tangent,global_tilt_deg)
            transform_pose(P,R=R_tilt,about=place)
        chains.append(P)
    return chains

# =========================================================
# Rotation about arbitrary axis, reflection, dihedral, ring stacking
# =========================================================
def decompose_along_axis(vec, axis_dir, axis_point):
    """Return (axis_component, radial_component) of vec relative to a line with direction axis_dir through axis_point."""
    u = unit(np.asarray(axis_dir, float))
    v = np.asarray(vec, float) - np.asarray(axis_point, float)
    axis_comp = np.dot(v, u) * u
    radial_comp = v - axis_comp
    return axis_comp, radial_comp

def rotate_about_axis(pose, axis_dir, angle_deg, about_point):
    """Rigid rotate pose about an arbitrary axis (direction & point)."""
    R = rotmat_axis_angle(axis_dir, angle_deg)
    transform_pose(pose, R=R, about=np.asarray(about_point, float))

def make_dihedral_extension(
    chains,
    dihedral_axis_vec,
    center_point,
    dihedral_twist_deg=0.0,
    sym_axis_vec=None,
    # NEW post-reflection rigid-body controls:
    #post_twist_deg=0.0,    # extra rotate about sym_axis
    post_shift_z=0.0,      # extra translate along sym_axis
    post_delta_r=0.0,      # extra radial expansion from center
    post_yaw_deg=0.0       # extra rotate about dihedral_axis
):
    """
    Build Dn partner set from a Cn ring:
      1) 180° rotation about dihedral_axis through center_point (C2)
      2) optional interdigitation twist about sym_axis by dihedral_twist_deg
      3) NEW: optional extra rigid ops applied to reflected partners only:
         - translate along sym_axis by post_shift_z
         - translate radially outward by post_delta_r
         - rotate about dihedral_axis by post_yaw_deg
    Returns original chains + transformed partners.
    """
    out = list(chains)
    C2 = unit(v3(dihedral_axis_vec)) if isinstance(dihedral_axis_vec, str) else unit(dihedral_axis_vec)
    C  = np.asarray(center_point, float)
    U  = unit(v3(sym_axis_vec)) if (sym_axis_vec is not None and isinstance(sym_axis_vec, str)) else (unit(sym_axis_vec) if sym_axis_vec is not None else None)

    for p in chains:
        q = p.clone()

        # (1) reflect by 180° around dihedral axis
        rotate_about_axis(q, C2, 180.0, C)

        # (2) optional interdigitation twist about main symmetry axis
        if U is not None and abs(dihedral_twist_deg) > 1e-9:
            rotate_about_axis(q, U, dihedral_twist_deg, C)

        # (3a) NEW: rotate around sym_axis (post_twist)
        #if U is not None and abs(post_twist_deg) > 1e-9:
        #    rotate_about_axis(q, U, post_twist_deg, C)

        # (3b) NEW: rotate around dihedral axis (post_yaw)
        if abs(post_yaw_deg) > 1e-9:
            rotate_about_axis(q, C2, post_yaw_deg, C)

        # (3c) NEW: translate along sym_axis (post_shift_z)
        if U is not None and abs(post_shift_z) > 1e-12:
            transform_pose(q, t=U * post_shift_z)

        # (3d) NEW: push radially outward from center by post_delta_r
        if abs(post_delta_r) > 1e-12:
            c0 = pose_center_CA(q)
            # radial dir = component of (c0 - C) perpendicular to U (if U present), else just from C to c0
            if U is not None:
                v = c0 - C
                axis_comp = np.dot(v, U) * U
                radial = v - axis_comp
                if np.linalg.norm(radial) < 1e-12:
                    # pick any stable perpendicular direction
                    radial = unit(np.cross(U, np.array([1.0, 0.0, 0.0])))
                radial_dir = unit(radial)
            else:
                radial_dir = unit(c0 - C) if np.linalg.norm(c0 - C) > 1e-12 else np.array([1.0, 0.0, 0.0])
            transform_pose(q, t=radial_dir * post_delta_r)

        out.append(q)

    return out


def stack_rings(chains, copies, delta_radius, delta_z, delta_twist_deg, sym_axis_vec, sym_center_point):
    """
    Duplicate the current set of chains into additional 'rings'.
    Each new ring i (1..copies) is:
      - shifted outward by i*delta_radius along each chain's radial direction
      - shifted along the symmetry axis by i*delta_z
      - rotated by i*delta_twist_deg about the symmetry axis (about sym_center_point)
    Returns original chains + stacked copies.
    """
    if copies <= 0:
        return chains
    out = list(chains)
    u = unit(v3(sym_axis_vec)) if isinstance(sym_axis_vec, str) else unit(sym_axis_vec)
    C = np.asarray(v3(sym_center_point) if isinstance(sym_center_point, str) else sym_center_point, float)

    for i in range(1, copies + 1):
        for p in chains:
            q = p.clone()
            # current center and its radial direction
            c0 = pose_center_CA(q)
            axis_comp, radial_comp = decompose_along_axis(c0, u, C)
            radial_dir = unit(radial_comp) if np.linalg.norm(radial_comp) > 1e-12 else unit(np.cross(u, np.array([1.0,0.0,0.0])))
            # translate: outward + along axis
            t = radial_dir * (delta_radius * i) + u * (delta_z * i)
            transform_pose(q, t=t)
            # twist around the axis
            if abs(delta_twist_deg) > 1e-9:
                rotate_about_axis(q, u, delta_twist_deg * i, C)
            out.append(q)
    return out

# =========================================================
# Output
# =========================================================
def build_out_filename(base_name:str, params:dict) -> str:
    base, ext = os.path.splitext(base_name)
    tags = []
    if params.get("sym_n", 0) > 0:
        tags.append(f"symN-{params['sym_n']}")
        tags.append(f"R-{params['sym_radius']:.1f}")
    if abs(params.get("global_tilt_deg",0)) > 1e-6:
        tags.append(f"tilt-{params['global_tilt_deg']:.1f}")
    if abs(params.get("local_roll_deg",0)) > 1e-6:
        tags.append(f"roll-{params['local_roll_deg']:.1f}")
    tag_str = "__" + "__".join(tags) if tags else ""
    return f"{base}{tag_str}{ext}"

def write_pdb_ca_multichain(poses,out_pdb,resname_disp="ALA",start_resid=1):
    # Write multiple poses as CA-only chains in a single PDB file
    os.makedirs(os.path.dirname(out_pdb) or ".",exist_ok=True)
    chain_ids=list(string.ascii_uppercase+string.ascii_lowercase)
    lines=[]
    serial=1
    for ci,p in enumerate(poses):
        chain=chain_ids[ci%len(chain_ids)]
        resi=start_resid
        for i in range(1,p.size()+1):
            a=p.xyz(rosetta.core.id.AtomID(p.residue(i).atom_index("CA"),i))
            lines.append(f"ATOM  {serial:5d}  CA  {resname_disp:>3s} {chain:1s}{resi:4d}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.0:6.2f}{20.0:6.2f}          C \n")
            serial+=1
            resi+=1
        lines.append("TER\n")
    lines.append("END\n")
    with open(out_pdb,"w") as f:f.writelines(lines)
    print(f"[OK] Write {len(poses)} chains to {out_pdb}")

# =========================================================
# Main function
# =========================================================
def main(config_path="config_helix.json"):
    cfg = load_config_json(config_path)
    globals().update(cfg)
    init_pyrosetta()

    # ============================================================
    #Build seed: either import PDB or build an ideal helix
    # ============================================================
    if input_pdb_path:
        print(f"[INFO] Using user PDB as seed: {input_pdb_path} (chain='{input_chain}', range='{input_pdb_range}')")
        seed = load_seed_pose_from_pdb(input_pdb_path, chain=input_chain, pdb_range=input_pdb_range)

        # --- Optional: align the imported PDB's own axis to a desired target axis ---
        if 'input_axis_mode' in globals():
            # detect the input axis
            if input_axis_mode == "auto_pca":
                src_axis = principal_axis_of_pose(seed)
            elif input_axis_mode == "auto_ends":
                src_axis = ends_axis_of_pose(seed)
            elif input_axis_mode == "explicit":
                src_axis = unit(v3(input_axis_hint))
            else:
                print(f"[WARN] Unknown input_axis_mode='{input_axis_mode}', fallback to auto_ends.")
                src_axis = ends_axis_of_pose(seed)

            tgt_axis = unit(v3(input_target_axis)) if 'input_target_axis' in globals() else unit(v3(helix_axis))

            # optional flip if nearly opposite
            if 'input_flip_if_opposite' in globals() and input_flip_if_opposite:
                if np.dot(src_axis, tgt_axis) < 0:
                    src_axis = -src_axis

            c_seed = pose_center_CA(seed)
            align_pose_axis_to(seed, src_axis, tgt_axis, about_point=c_seed)

            # optional roll around the new axis
            if 'input_pre_roll_deg' in globals() and abs(input_pre_roll_deg) > 1e-6:
                R_roll = rotmat_axis_angle(tgt_axis, input_pre_roll_deg)
                transform_pose(seed, R=R_roll, about=c_seed)

            # optional recenter after alignment
            if 'input_center_to' in globals() and input_center_to:
                if input_center_to == "sym_center":
                    target_center = v3(sym_center)
                elif input_center_to == "none":
                    target_center = None
                else:
                    target_center = v3(input_center_to)
                if target_center is not None:
                    shift = np.asarray(target_center, float) - pose_center_CA(seed)
                    transform_pose(seed, t=shift)

    else:
        seed = build_ideal_helix_pose(helix_len, "A")

    # ============================================================
    # Align helix axis to target axis (for ideal helix case)
    # ============================================================
    ca = extract_CA_coords(seed)
    curr_axis = unit(ca[-1] - ca[0]) if len(ca) >= 2 else np.array([0, 0, 1])
    target = unit(v3(helix_axis))
    v = np.cross(curr_axis, target)
    s = np.linalg.norm(v)
    c = float(np.dot(curr_axis, target))
    R_align = np.eye(3) if s < 1e-8 and c > 0 else rotmat_axis_angle(v/s if s > 1e-8 else (1, 0, 0), math.degrees(math.atan2(s, c)))

    center = pose_center_CA(seed)
    transform_pose(seed, R=R_align, about=center)
    transform_pose(seed, t=v3(helix_start) - pose_center_CA(seed))

    # optional local alignment
    if local_align_axis and abs(local_align_deg) > 1e-6:
        R_local = rotmat_axis_angle(v3(local_align_axis), local_align_deg)
        transform_pose(seed, R=R_local, about=pose_center_CA(seed))

    # ============================================================
    # Symmetry, Dihedral, Stacking
    # ============================================================
    if sym_n > 0:
        chains = c_n_symmetry_place_pose(seed, sym_n, v3(sym_axis), v3(sym_center),
                                         sym_radius, sym_start_angle, global_tilt_deg, local_roll_deg)
    else:
        base = seed.clone()
        if abs(local_roll_deg) > 1e-6:
            Rr = rotmat_axis_angle(v3(sym_axis), local_roll_deg)
            transform_pose(base, R=Rr, about=pose_center_CA(base))
        chains = [base]

    # --- Dihedral extension (C_n -> D_n) ---
    if dihedral_enable:
        chains = make_dihedral_extension(
            chains,
            dihedral_axis_vec=dihedral_axis,
            center_point=v3(sym_center),
            dihedral_twist_deg=dihedral_twist_deg,
            sym_axis_vec=v3(sym_axis),
            #post_twist_deg=dihedral_post_twist_deg,
            post_shift_z=dihedral_post_shift_z,
            post_delta_r=dihedral_post_delta_r,
            post_yaw_deg=dihedral_post_yaw_deg
        )

    # --- Ring stacking / expansion ---
    if ring_stack_copies > 0:
        chains = stack_rings(
            chains,
            copies=ring_stack_copies,
            delta_radius=ring_delta_radius,
            delta_z=ring_delta_z,
            delta_twist_deg=ring_delta_twist_deg,
            sym_axis_vec=v3(sym_axis),
            sym_center_point=v3(sym_center)
        )


    # ============================================================
    #  Output
    # ============================================================
    params = dict(sym_n=sym_n, sym_radius=sym_radius, global_tilt_deg=global_tilt_deg,
                  local_roll_deg=local_roll_deg)
    tagged_pdb = build_out_filename(out_pdb, params)
    write_pdb_ca_multichain(chains, tagged_pdb, resname_disp=resname)

    print("\nPyMOL quick tips:")
    print("  load", tagged_pdb)
    print("  util.cbc; show sticks, all; orient; set stick_radius, 0.2")

if __name__=="__main__":
    import sys
    config = sys.argv[1] if len(sys.argv)>1 else "config_helix.json"
    main(config)
