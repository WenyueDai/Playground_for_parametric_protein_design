import os, math
import numpy as np
import pyrosetta
from pyrosetta import rosetta


import math
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc, HarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint, AtomPairConstraint
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
from pyrosetta.rosetta.numeric import xyzVector_double_t

def _ensure_init():
    try:
        pyrosetta.get_fa_scorefxn()
    except Exception:
        pyrosetta.init("-mute all")
        
def _pose_from_seq(seq: str, rts_name="fa_standard") -> rosetta.core.pose.Pose:
    """Create a Pose object from a given amino acid sequence.
    """
    cm = rosetta.core.chemical.ChemicalManager.get_instance()
    rts = cm.residue_type_set(rts_name)
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, rts)
    return pose

def _set_phi_psi_omega(pose, start, end, phi, psi, omg=180.0):
    """Set the backbone dihedral angles (phi, psi, omega) for residues in a Pose.
    """
    for i in range(start, end + 1):
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, omg)
        
def _get_sf_with_csts(DO_CONSTRAINTS=True, CST_WEIGHT=1.0):
    """Get a score function with optional constraints.
    """
    sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015")
    if DO_CONSTRAINTS:
        sf.set_weight(rosetta.core.scoring.atom_pair_constraint, CST_WEIGHT)
    return sf
   
def _relax(pose, RELAX_ROUNDS=5):
    """Relax the Pose.
    """
    sf = _get_sf_with_csts()
    fr = rosetta.protocols.relax.FastRelax(RELAX_ROUNDS)
    fr.set_scorefxn(sf)
    fr.apply(pose)
    
def _add_sheet_ca_constraints(pose, pairs, DO_CONSTRAINTS=True, CST_CA_DIST=10.0, CST_CA_SD=1.0):
    """
    Keep C-alpha atoms of specified residue pairs at a target distance using constraints. 
    For example, if pairs = [(1,10), (5,15)], it will add constraints between CA of residue 1 and 10,
    and between CA of residue 5 and 15. Tells Rosetta to keep these 1 - 10 CA atoms approximately CST_CA_DIST apart.
    CST_CA_SD controls how tightly the distance is enforced (smaller = tighter).
    """
    if not DO_CONSTRAINTS:
        return 
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
    for i, j in pairs:
        ai = AtomID(pose.residue(i).atom_index("CA"), i)
        aj = AtomID(pose.residue(j).atom_index("CA"), j)
        func = HarmonicFunc(CST_CA_DIST, CST_CA_SD)
        pose.add_constraint(AtomPairConstraint(ai, aj, func))
        
def _rotmat(axis, angle_deg):
    """Generate a rotation matrix for rotating around a given axis by a specified angle in degrees.
    axis = [0,1,0] for y-axis, angle_deg = 45 for 45 degrees - that means a 45 degree rotation around y-axis.
    """
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    x, y, z = u
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    
def _transform_pose_inplace(pose, R=None, t=None):
    """Apply a rigid body transformation to all atoms in the Pose.
    """
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    for i in range(1, pose.size() + 1):
        rsd = pose.residue(i)
        for j in range(1, rsd.natoms() + 1):
            aid = rosetta.core.id.AtomID(j, i)
            xyz = pose.xyz(aid)
            p = np.array([xyz.x, xyz.y, xyz.z])
            p2 = R @ p + t
            pose.set_xyz(aid, rosetta.numeric.xyzVector_double_t(*p2))
            
def _transform_pose_segment(pose, start, end, R=None, t=None):
    """Apply a rigid body transformation to a segment of residues in the Pose.
    """
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    for i in range(start, end+1):
        rsd = pose.residue(i)
        for j in range(1, rsd.natoms()+1):
            aid = rosetta.core.id.AtomID(j, i)
            xyz = pose.xyz(aid)
            p = np.array([xyz.x, xyz.y, xyz.z])
            p2 = R @ p + t
            pose.set_xyz(aid, rosetta.numeric.xyzVector_double_t(*p2))
            
def _segment_centroid(pose, start, end):
    """Calculate the centroid of C-alpha atoms for a segment of residues in the Pose.
    """
    import numpy as np
    from pyrosetta import rosetta
    c = np.zeros(3)
    n = 0
    for i in range(start, end+1):
        ai = rosetta.core.id.AtomID(pose.residue(i).atom_index("CA"), i)
        xyz = pose.xyz(ai)
        c += np.array([xyz.x, xyz.y, xyz.z])
        n += 1
    return c / max(n,1)

def _align_segment_axis_to_z(pose, start, end):
    """
    Align the axis defined by the C-alpha atoms of the start and end residues of a segment to the z-axis.
    """
    import numpy as np, math
    from pyrosetta import rosetta

    # Centroid
    cen = _segment_centroid(pose, start, end)

    # Get start and end CA atoms to estimate axis
    a1 = rosetta.core.id.AtomID(pose.residue(start).atom_index("CA"), start)
    a2 = rosetta.core.id.AtomID(pose.residue(end).atom_index("CA"),   end)
    p1 = pose.xyz(a1); p2 = pose.xyz(a2)
    v  = np.array([p2.x-p1.x, p2.y-p1.y, p2.z-p1.z])
    if np.linalg.norm(v) < 1e-6: 
        return
    v = v / np.linalg.norm(v)
    z = np.array([0.,0.,1.])

    # Rotate to z-axis: axis = v×z, angle = arccos(v·z)
    axis = np.cross(v, z); axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:   # Already aligned or opposite
        if np.dot(v, z) > 0: 
            return
        # Opposite: rotate 180° around any axis perpendicular to z
        axis = np.array([1.,0.,0.]); axis_norm = 1.0
    axis = axis / axis_norm
    ang  = math.degrees(math.acos(np.clip(np.dot(v, z), -1.0, 1.0)))

    # Rotate around centroid: translate to origin -> rotate -> translate back
    R = _rotmat(axis, ang)
    t1 = -cen
    t2 = cen
    _transform_pose_segment(pose, start, end, R=np.eye(3), t=t1)
    _transform_pose_segment(pose, start, end, R=R,        t=np.zeros(3))
    _transform_pose_segment(pose, start, end, R=np.eye(3), t=t2)
    
def _add_coordinate_anchors_for_segment(pose, start, end, stride=3, sd=0.5):
    """
    Add CoordinateConstraint to CA atoms every `stride` residues in [start, end],
    anchoring them to their current coordinates (keeps helices rigid).
    """

    # use the first residue N atom as the reference frame origin
    ref = AtomID(pose.residue(1).atom_index("N"), 1)
    # add constraints to CA atoms in the segment
    for i in range(start, end + 1, stride):
        ai = AtomID(pose.residue(i).atom_index("CA"), i)
        xyz = pose.xyz(ai)
        target = xyzVector_double_t(xyz.x, xyz.y, xyz.z)  # anchor to current spot
        func = HarmonicFunc(0.0, sd)
        pose.add_constraint(CoordinateConstraint(ai, ref, target, func))
        
def _add_helix_geometry_constraints(pose, start, end,
                                    phi, psi,
                                    sd_tors_deg=8.0,   # tighter torsion SD
                                    ca_i3=5.2, ca_i4=6.4, sd_ca=0.25,
                                    add_hbond=False, d_NO=2.9, sd_NO=0.25):
    """
    Adds: φ/ψ/ω dihedral constraints, CA(i)-CA(i+3/4) distances,
    and (optional) N(i+4)-O(i) pseudo-Hbond distances.
    """
    rad = math.radians
    for i in range(start+1, end-1):
        # ids
        C_im1 = AtomID(pose.residue(i-1).atom_index("C"), i-1)
        N_i   = AtomID(pose.residue(i).atom_index("N"), i)
        CA_i  = AtomID(pose.residue(i).atom_index("CA"), i)
        C_i   = AtomID(pose.residue(i).atom_index("C"), i)
        N_ip1 = AtomID(pose.residue(i+1).atom_index("N"), i+1)

        # φ(i) and ψ(i)
        pose.add_constraint(DihedralConstraint(C_im1, N_i, CA_i, C_i,
                         CircularHarmonicFunc(rad(phi), rad(sd_tors_deg))))
        pose.add_constraint(DihedralConstraint(N_i, CA_i, C_i, N_ip1,
                         CircularHarmonicFunc(rad(psi), rad(sd_tors_deg))))
        # ω(i) ~ 180°
        OMEGA = 180.0
        CA_im1 = AtomID(pose.residue(i-1).atom_index("CA"), i-1)
        pose.add_constraint(DihedralConstraint(CA_im1, C_im1, N_i, CA_i,
                         CircularHarmonicFunc(rad(OMEGA), rad(6.0))))

    # Add Cα(i)–Cα(i+3/4) distance signatures
    for i in range(start, end-4):
        CA_i = AtomID(pose.residue(i).atom_index("CA"), i)
        CA_i3 = AtomID(pose.residue(i+3).atom_index("CA"), i+3)
        CA_i4 = AtomID(pose.residue(i+4).atom_index("CA"), i+4)
        pose.add_constraint(AtomPairConstraint(CA_i, CA_i3, HarmonicFunc(ca_i3, sd_ca)))
        pose.add_constraint(AtomPairConstraint(CA_i, CA_i4, HarmonicFunc(ca_i4, sd_ca)))

    # Optional pseudo-Hbond N(i+4)–O(i)
    if add_hbond:
        for i in range(start, end-4):
            N_ip4 = AtomID(pose.residue(i+4).atom_index("N"), i+4)
            O_i   = AtomID(pose.residue(i).atom_index("O"), i)
            pose.add_constraint(AtomPairConstraint(N_ip4, O_i, HarmonicFunc(d_NO, sd_NO)))

def _append_residue(pose, res):
    if pose.size() == 0:
        pose.append_residue_by_bond(res, True)
    else:
        pose.append_residue_by_bond(res, False)

if __name__ == "__main__":
    _ensure_init()
    # test different functions
    seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids
    pose = _pose_from_seq(seq)

    _set_phi_psi_omega(pose, 1, pose.size(), -60.0, -45.0)

    _relax(pose, RELAX_ROUNDS=3)
    
    # test adding constraints
    pairs = [(1,10), (5,15)]
    _add_sheet_ca_constraints(pose, pairs, DO_CONSTRAINTS=True)
    
    # test transformations
    R = _rotmat([0,1,0], 45.0) # rotate 45° around y-axis
    t = np.array([5.0, 0.0, 0.0]) # translate 5 Å along x-axis
    _transform_pose_inplace(pose, R, t)
    
    # test segment alignment
    _align_segment_axis_to_z(pose, 1, pose.size())
    
    # test adding helix geometry constraints
    PHI_ALPHA = -57.8
    PSI_ALPHA = -47.0
    _add_helix_geometry_constraints(pose, 1, pose.size(),
                                    phi=PHI_ALPHA, psi=PSI_ALPHA,
                                    add_hbond=True)
    
