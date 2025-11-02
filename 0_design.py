"""
This script performs protein structure manipulation using PyRosetta:

1. Input Structure Generation:
    - Can either build an ideal alpha-helix from scratch
    - Or load an existing protein structure from a PDB file

2. Initial Structure Preparation:
    - Centers structure at origin 
    - Aligns helix/structure axis to target direction

3. Symmetric Assembly Generation:
    - Creates cyclic (Cn) symmetry by placing copies in a ring
    - Optional global tilt of each unit away from symmetry axis
    - Optional dihedral (Dn) symmetry by reflecting the ring
    - Optional stacking of multiple rings with translations/rotations

4. Output:
    - Saves the final assembly as a PDB file (CA atoms only)

Usage:
  conda activate pyrosetta
  python sym_helix_extended_pyrosetta.py config_template.json
  
  OR
  python sym_helix_extended_pyrosetta.py
"""

import os, math, json, string
from typing import List, Tuple
import numpy as np
import pyrosetta
from pyrosetta import rosetta
import hashlib
import datetime

# =========================================================
# Default parameters
# =========================================================
# For demo: single helix
helix_len = 28 # number of residues in the helix
phi_deg = -57.8 # typical alpha-helix phi angle
psi_deg = -47.0 # typical alpha-helix psi angle
omega_deg = 180.0 # typical alpha-helix omega angle

# --- Otherwise, use your own pdb ---
input_pdb_path = "/home/eva/20251031_parametric_design_playground/8flx.pdb"      # empty string means "disabled"
input_chain = ""         # "A"; empty means "use whole pose or infer"
input_pdb_range = ""     # "5-42" in PDB numbering within the chosen chain; empty means "full chain"

# --- Input parameters compatible for both de novo build or input pdb ---
input_axis_mode = "auto_pca"      # "auto_pca" | "auto_ends" - use N-C end as axis which is much faster than PCA, could do poorly for short/curved segments
input_target_axis = "0,0,1"        # DEFAULT: if None/""/"none", aligns to +Z (0,0,1); if a vector "x,y,z", aligns to that vector.
roll_deg = 0           # roll around the (aligned) target axis

# --- Cn symmetry parameters ---
sym_n = 3 # number of symmetry units; 0 means no symmetry
sym_axis = "0,0,1" # symmetry axis direction, default Z
sym_center = "0,0,0" # symmetry center point
sym_radius = 30.0 # radius from symmetry axis to helix center
sym_start_angle = 0.0 # starting angle for symmetry placement (azimuth of the first helix/pdb), like rotate the whole ring
global_tilt_deg = 0.0 # tilt each helix away from sym_axis by this many degrees (lean in/out)

# --- Dihedral extension (C_n -> D_n) ---
dihedral_enable = False # create partner ring by a 180° rotation about dihedral_axis through sym_center
dihedral_axis = "1,0,0"         # a C2 axis perpendicular to sym_axis
dihedral_post_shift_z   = 0.0      # translate/shift along sym_axis (Å)
dihedral_post_delta_r   = 0.0      # push outward radially from sym_center (Å)
dihedral_twist_deg = 0.0        # extra twist about sym_axis immediately after reflection

# --- Ring stacking / expansion ---
ring_stack_copies = 0           # number of extra rings to add
ring_delta_radius = 0.0         # outward shift per ring (Å)
ring_delta_z = 0.0              # axial shift per ring (Å)
ring_delta_twist_deg = 0.0      # extra azimuth per ring (deg)

resname = "ALA"     # residue name to display in PDB
out_pdb = "outs/8flx.pdb" # the bash path will be used as dictionary, filename will joined with hash key to save

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

def v3(s):
    # Convert comma-separated string (from json or global) to 3D vector
    return np.array([float(x) for x in s.split(",")],float)

def parse_axis(val):
    # Accept None, "", "none", "None" as "no target"
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in {"", "none"}:
        return None
    return unit(v3(val))  # otherwise, normalize the "x,y,z" string

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

def load_config_json(path:str):
    if not os.path.exists(path): 
        print(f"[WARN] JSON config {path} not exist, use default."); return {}
    with open(path,"r") as f: 
        data=json.load(f)
    print(f"[OK] Read JSON config: {path}")
    return data

# =========================================================
# PyRosetta operations
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

def assemble_as_multichain_pose(poses):
    """
    Combine a list of Poses (full-atom) into a single Pose with one chain per input Pose.
    Uses append_pose_to_pose with new_chain=True so each Pose becomes its own chain.
    """
    if not poses:
        raise ValueError("No poses to assemble.")
    out = poses[0].clone()
    for p in poses[1:]:
        rosetta.core.pose.append_pose_to_pose(out, p, new_chain=True)
    return out

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

def extract_CA_coords(pose):
    # Extract CA atom coordinates as Nx3 numpy array
    # It is simple but very useful when we need the geomtry of the pose
    # e.g., compute helix axis, center, aligning, etc.
    arr=[]
    for i in range(1,pose.size()+1):
        a=pose.xyz(rosetta.core.id.AtomID(pose.residue(i).atom_index("CA"),i))
        arr.append([a.x,a.y,a.z])
    return np.array(arr)

def pose_center_CA(pose):
    # use extract_CA_coords to compute the center of mass of CA atoms
    ca_coords = extract_CA_coords(pose)
    return np.mean(ca_coords, axis=0)

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
            # the formula is: p2 = R*(p - about) + about + t
            p2 = (R @ (p - about)) + about + t
            # set new xyz to pose
            pose.set_xyz(aid,rosetta.numeric.xyzVector_double_t(*p2))

# =========================================================
# Pose axis computations
# =========================================================
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
    """Rotate pose to align source_axis to target_axis about about_point."""
    if target_axis is None:
        return  # explicitly skip alignment, the input object will be defaulted to align to z-axis

    a = unit(np.asarray(source_axis, float))
    b = unit(np.asarray(target_axis, float))
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    # Handle special cases
    # 1) a and b are parallel (c=1)
    # 2) a and b are anti-parallel (c=-1)
    # In case (1), no rotation needed
    # In case (2), pick any perpendicular axis to rotate 180°
    # Otherwise, use axis-angle rotation
    if s < 1e-8:
        if c > 0:
            return
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

# =========================================================
# Cn
# =========================================================
def c_n_symmetry_place_pose(seed, n, sym_axis_vec, center_vec, radius_xy,
                            start_angle_deg, global_tilt_deg):
    """
    Generate Cn symmetry copies of a seed Pose, arranged evenly around a defined symmetry axis.
    Each copy is rotated about the symmetry axis (true rotational symmetry) *and* placed around a ring.
    """
    sym_axis = unit(np.array(sym_axis_vec, float))
    center = np.array(center_vec, float)
    # Build stable perpendicular axes for placement
    tmp = np.array([1,0,0]) if abs(np.dot([1,0,0], sym_axis)) < 0.9 else np.array([0,1,0])
    x_dir = unit(np.cross(sym_axis, tmp))
    y_dir = unit(np.cross(sym_axis, x_dir))

    chains = []
    base = seed.clone()

    for k in range(n):
        ang = start_angle_deg + 360.0 * k / n
        # compute radial direction in XY plane
        radial = math.cos(math.radians(ang)) * x_dir + math.sin(math.radians(ang)) * y_dir
        # compute placement point
        place = center + radius_xy * radial

        P = base.clone()

        # (1) Rotate subunit by the same azimuthal angle about the symmetry axis
        R_rot = rotmat_axis_angle(sym_axis, ang)
        transform_pose(P, R=R_rot, about=center)

        # (2) Translate to the ring position
        c0 = pose_center_CA(P)
        transform_pose(P, t=place - c0)

        # (3) Apply global tilt if needed (lean away from axis)
        if abs(global_tilt_deg) > 1e-6:
            tangent = unit(np.cross(sym_axis, radial))
            R_tilt = rotmat_axis_angle(tangent, global_tilt_deg)
            transform_pose(P, R=R_tilt, about=place)

        chains.append(P)

    return chains

# =========================================================
# Dihedral extensions
# =========================================================
def decompose_along_axis(vec, axis_dir, axis_point):
    """
    Return (axis_component, radial_component) of vec relative to a line with direction axis_dir through axis_point.
    If v = [3, 4, 5], u = [0, 0, 1],
    then np.dot(v, u) = 5
    and axis_comp = 5 * [0, 0, 1] = [0, 0, 5].
    radial_comp = [3, 4, 5] - [0, 0, 5] = [3, 4, 0]
    """
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
    dihedral_axis_vec, # axis for 180° reflection
    center_point,
    dihedral_twist_deg=0.0, # twist about sym_axis after reflection
    sym_axis_vec=None,     # symmetry axis for twist
    post_shift_z=0.0,      # extra translate along sym_axis
    post_delta_r=0.0,      # extra radial expansion from center
    ) -> List[rosetta.core.pose.Pose]:
    """
    Build Dn partner set from a Cn ring:
      1) 180° rotation about dihedral_axis through center_point (C2)
      2) optional interdigitation twist about sym_axis by dihedral_twist_deg
      3) translate along sym_axis by post_shift_z
      4) translate radially outward by post_delta_r
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

        # (3a) rotate around sym_axis (post_twist)
        #if U is not None and abs(post_twist_deg) > 1e-9:
        #    rotate_about_axis(q, U, post_twist_deg, C)

        # (3c) translate along sym_axis (post_shift_z)
        if U is not None and abs(post_shift_z) > 1e-12:
            transform_pose(q, t=U * post_shift_z)

        # (3d) push radially outward from center by post_delta_r
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

# =========================================================
# Stack rings
# =========================================================

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
            # current center of mass
            c0 = pose_center_CA(q)
            axis_comp, radial_comp = decompose_along_axis(c0, u, C)
            radial_dir = radial_comp
            # ensure radial_dir is not zero
            if np.linalg.norm(radial_dir) < 1e-12:
                radial_dir = np.cross(u, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(radial_dir) < 1e-12:
                    radial_dir = np.cross(u, np.array([0.0, 1.0, 0.0]))
            radial_dir = unit(radial_dir)
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
def build_out_filename(base_name: str, params: dict, max_basename_len: int = 10) -> str:
    """
    Build an output filename that encodes *all* parameters.
    - Compact numeric formatting
    - Paths reduced to basenames
    - Vectors like "x,y,z" converted to "x_y_z"
    - Booleans to 1/0
    - Falls back to adding an md5 short hash if too long for common filesystems
    """
    base, ext = os.path.splitext(base_name)

    def fmt_val(v):
        # Normalize and compact values
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, float):
            # 3 significant digits usually enough; keep minus sign and decimal
            s = f"{v:.3g}"
            # avoid filesystem/CLI quirks from plus signs
            return s.replace("+", "")
        if isinstance(v, str):
            if v == "":
                return "none"
            # compress file paths to basenames
            v2 = os.path.basename(v)
            # commas to underscores for vectors; spaces to nothing
            v2 = v2.replace(",", "_").replace(" ", "")
            # avoid accidental path separators / backslashes / colons
            for ch in ["/", "\\", ":", ";"]:
                v2 = v2.replace(ch, "-")
            return v2
        # fallback
        return str(v)

    # Sort keys for stable names
    tokens = [f"{k}-{fmt_val(params[k])}" for k in sorted(params.keys())]
    tag_str = "__" + "__".join(tokens) if tokens else ""
    candidate = f"{base}{tag_str}{ext}"

    # create an empty dictionary to save candidate filenames and hashkey
    candidate_dict = {}
    # Guard against very long basenames (common FS limit ~255)
    bn = os.path.basename(candidate)
    if len(bn) > max_basename_len:
        short_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        # save the candidate filename and hashkey to dictionary
        candidate_dict[candidate] = short_hash
        # save the dictionary to a json file
        with open(f"filename_hash.json", "a") as f:
            # dump the date and time into the json file
            json.dump({"date": str(datetime.datetime.now()), "files": candidate_dict}, f, indent=2)
        candidate = f"{base}_hash-{short_hash}{ext}"
    return candidate

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
    # Build seed: either import PDB or build an ideal helix
    # Then: ALWAYS center COM at origin AND align to desired axis
    # ============================================================
    if input_pdb_path:
        print(f"[INFO] Using user PDB as seed: {input_pdb_path} (chain='{input_chain}', range='{input_pdb_range}')")
        seed = load_seed_pose_from_pdb(input_pdb_path, chain=input_chain, pdb_range=input_pdb_range)
    else:
        seed = build_ideal_helix_pose(helix_len, "A")

    #--------(A) ALWAYS recenter the seed pose to origin ---------
    c_seed = pose_center_CA(seed)
    transform_pose(seed, t=-np.asarray(c_seed, float))   # COM → (0,0,0)
    
    # --- (B) Pick the source axis (from current pose coords) ---
    if input_axis_mode == "auto_pca":
        src_axis = principal_axis_of_pose(seed)
    elif input_axis_mode == "auto_ends":
        src_axis = ends_axis_of_pose(seed)
    else:
        print(f"[WARN] Unknown input_axis_mode='{input_axis_mode}', fallback to auto_ends.")
        src_axis = ends_axis_of_pose(seed)
        
    # --- (C) Decide target axis (can be None = skip) ---
    _tgt = parse_axis(globals().get("input_target_axis", None))
    # So if _tgt is None → desired axis is +Z ; if not None → use _tgt.
    # If you prefer "no alignment at all" when None, then set desired_axis = None here.
    desired_axis = _tgt if _tgt is not None else np.array([0.0, 0.0, 1.0], float)
    
    #--- (D) Align source axis to target axis ---
    # ensure src_axis points roughly the same hemisphere as desired_axis
    if np.dot(src_axis, desired_axis) < 0:
        src_axis = -src_axis
    align_pose_axis_to(seed, src_axis, desired_axis, about_point=np.zeros(3))
    
    # --- (E) optional roll around the new axis ---
    if abs(roll_deg) > 1e-6:
        R_roll = rotmat_axis_angle(desired_axis, roll_deg)
        transform_pose(seed, R=R_roll, about=np.zeros(3))

    # ============================================================
    # Symmetry, Dihedral, Stacking
    # ============================================================
    if sym_n > 0:
        chains = c_n_symmetry_place_pose(seed, sym_n, v3(sym_axis), v3(sym_center),
                                         sym_radius, sym_start_angle, global_tilt_deg)
    else:
        chains = [seed.clone()]

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
    params = dict(
        # build/seed
        helix_len=helix_len, phi_deg=phi_deg, psi_deg=psi_deg, omega_deg=omega_deg,
        input_pdb_path=input_pdb_path, input_chain=input_chain, input_pdb_range=input_pdb_range,
        # alignment
        input_axis_mode=input_axis_mode, input_target_axis=input_target_axis, roll_deg=roll_deg,
        # Cn
        sym_n=sym_n, sym_axis=sym_axis, sym_center=sym_center, sym_radius=sym_radius,
        sym_start_angle=sym_start_angle, global_tilt_deg=global_tilt_deg,
        # Dn extension
        dihedral_enable=dihedral_enable, dihedral_axis=dihedral_axis,
        dihedral_post_shift_z=dihedral_post_shift_z, dihedral_post_delta_r=dihedral_post_delta_r,
        dihedral_twist_deg=dihedral_twist_deg,
        # ring stacking
        ring_stack_copies=ring_stack_copies, ring_delta_radius=ring_delta_radius,
        ring_delta_z=ring_delta_z, ring_delta_twist_deg=ring_delta_twist_deg,
        # formatting/display
        resname=resname
    )

    tagged_pdb = build_out_filename(out_pdb, params)

    # Decide how to write:
    # - If we imported a PDB, preserve native residue types & side chains (full-atom PDB).
    # - If we built de novo, keep Ala-only (CA-only file as before).
    imported_seed = bool(input_pdb_path)  # non-empty path means we loaded from PDB

    if imported_seed:
        # Full-atom, preserve residue names & side chains
        assembled = assemble_as_multichain_pose(chains)
        os.makedirs(os.path.dirname(tagged_pdb) or ".", exist_ok=True)
        rosetta.core.io.pdb.dump_pdb(assembled, tagged_pdb)
        print(f"[OK] Wrote full-atom multichain PDB preserving side chains to {tagged_pdb}")
    else:
        # De novo Ala helix: CA-only with ALA residue label (your original behavior)
        write_pdb_ca_multichain(chains, tagged_pdb, resname_disp="ALA")
        print(f"[OK] Wrote CA-only ALA multichain PDB to {tagged_pdb}")

    print(f"[DONE] Total chains: {len(chains)}")

if __name__=="__main__":
    import sys
    config = sys.argv[1] if len(sys.argv)>1 else "config_helix.json"
    main(config)
