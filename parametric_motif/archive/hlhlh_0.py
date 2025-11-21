import math, numpy as np
from pyrosetta import rosetta
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc, CircularHarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint, DihedralConstraint, CoordinateConstraint
from pyrosetta.rosetta.core.kinematics import FoldTree, MoveMap
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.protocols.loops import Loop
from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import CCDLoopClosureMover
from pyrosetta.rosetta.protocols.idealize import IdealizeMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from utils import _rotmat, _transform_pose_segment, _segment_centroid, _align_segment_axis_to_z
from utils import _add_coordinate_anchors_for_segment, _add_helix_geometry_constraints
from utils import _relax, _ensure_init, _pose_from_seq, _set_phi_psi_omega
from helix import PHI_ALPHA, PSI_ALPHA
from pyrosetta import init

# α-torsion constraints (keeps helices helical during minimization)
def add_alpha_dihedral_csts(pose, start, end, sd=15.0):
    # The periodic harmonic penalty was applied for residue start+1 .. end-1 (skip the end)
    for i in range(start+1, end-1):
        # constrain ϕ(i) around PHI_ALPHA and ψ(i) around PSI_ALPHA
        n  = AtomID(pose.residue(i).atom_index("N"), i)
        ca = AtomID(pose.residue(i).atom_index("CA"), i)
        c  = AtomID(pose.residue(i).atom_index("C"), i)
        p  = AtomID(pose.residue(i-1).atom_index("C"), i-1) 
        n2 = AtomID(pose.residue(i+1).atom_index("N"), i+1)
        # ϕ = C(i-1)-N(i)-CA(i)-C(i)
        # CircularHarmonicFunc applies a periodic harmonic penalty to angular variables (such as torsion angles).
        pose.add_constraint(DihedralConstraint(p, n, ca, c, CircularHarmonicFunc(math.radians(PHI_ALPHA), math.radians(sd))))
        # ψ = N(i)-CA(i)-C(i)-N(i+1)
        pose.add_constraint(DihedralConstraint(n, ca, c, n2, CircularHarmonicFunc(math.radians(PSI_ALPHA), math.radians(sd))))

def flip_seg(pose, a,b,flag):
    # flip_seg rotates the selected helix 180° about its own axis (via centroid–rotate–restore pattern), 
    # making it anti-parallel without translating it.
    if not flag: 
        return
    R = _rotmat([1,0,0], 180.0)
    # mass center
    cen = _segment_centroid(pose,a,b)
    _transform_pose_segment(pose,a,b,t=-cen)
    _transform_pose_segment(pose,a,b,R=R)
    _transform_pose_segment(pose,a,b,t=cen)
    
def axis_sign(pose, a,b):
    # return +1 if helix from a to b goes up (z+), else -1
    ca1 = pose.xyz(AtomID(pose.residue(a).atom_index("CA"), a))
    ca2 = pose.xyz(AtomID(pose.residue(b).atom_index("CA"), b))
    return 1.0 if (ca2.z - ca1.z) >= 0 else -1.0

def C_ofs(s,half): return (+half if s>0 else -half) # helix C-end offset
def N_ofs(s,half): return (-half if s>0 else +half) # helix N-end offset

# inward micro-tilt for packing, the tilt is happen perpendicular to its phase direction
def microtilt(pose, a,b,phase,deg=6.0):
    th = math.radians(phase)
    ax = np.array([-math.sin(th), math.cos(th), 0.0])
    R  = _rotmat(ax, deg)
    cen = _segment_centroid(pose,a,b)
    _transform_pose_segment(pose,a,b,t=-cen)
    _transform_pose_segment(pose,a,b,R=R)
    _transform_pose_segment(pose,a,b,t=cen)
    
# ca(i) helper fetches an AtomID for Cα of residue i.
def ca(pose, i): 
    return AtomID(pose.residue(i).atom_index("CA"), i)
    
def add_pair_grid(pose, hA_a,hA_b,hB_a,hB_b, target=8.4, sd=0.4):
    # Lays a grid of Cα–Cα distance constraints every other residue between each helix pair.
    idxA = list(range(hA_a+2, hA_b-1, 2))
    idxB = list(range(hB_a+2, hB_b-1, 2))
    for i,j in zip(idxA, idxB):
        pose.add_constraint(AtomPairConstraint(ca(pose, i), ca(pose, j), HarmonicFunc(target, sd)))
        
# ---------- helper: staged cartesian minimization with coord/pair weights ----------
def cart_min(pose, N, start, end, coord_w, pair_w, linear_w=0.0, dih_w=1.0, allow_helix_bb=False):
    # cartesian minimization from start to end with given constraint weights
    sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015_cart")
    sfxn.set_weight(rosetta.core.scoring.atom_pair_constraint, pair_w) # (packing/loop-end targets)
    sfxn.set_weight(rosetta.core.scoring.coordinate_constraint, coord_w) # (anchors)
    sfxn.set_weight(rosetta.core.scoring.dihedral_constraint, dih_w) # (helical torsions)
    if linear_w>0: sfxn.set_weight(rosetta.core.scoring.linear_chainbreak, linear_w) # (to seal cutpoints)
    mm = MoveMap()
    # move loops only; keep helices fixed except optional tiny tweaks
    # Creates a MoveMap that mobilizes only the target range (start..end, typically a loop) — 
    # helices are fixed except optional tiny “breathing” on every 6th residue (very limited bb motion).
    for r in range(1, N+1):
        in_loop = (start<=r<=end)
        if in_loop:
            mm.set_bb(r, True); mm.set_chi(r, True)
        else:
            mm.set_bb(r, allow_helix_bb and (r%6==0))  # tiny breathing if enabled
            mm.set_chi(r, False)
    m = MinMover(); m.score_function(sfxn); m.movemap(mm); m.cartesian(True)
    m.min_type("lbfgs_armijo_nonmonotone")
    m.apply(pose)

# ---------- robust loop closure (sealed bonds) ----------
def close_loop(pose, N, start, end):
    # close loop in [start,end] using CCD with cutpoint sealing
    if end < start: return
    # Uses a cutpoint halfway through the loop (cut), marking residues cut and cut+1 as 
    # special “CUTPOINT” variants so Rosetta can treat the break explicitly.
    cut = start + (end-start)//2
    # Resets the fold tree to a simple linear one (important for CCD).
    ft = FoldTree(); ft.simple_tree(N); pose.fold_tree(ft)
    rosetta.core.pose.add_variant_type_to_pose_residue(pose, VariantType.CUTPOINT_LOWER, cut)
    rosetta.core.pose.add_variant_type_to_pose_residue(pose, VariantType.CUTPOINT_UPPER, cut+1)
    # Defines the Loop object for the closure region.
    loop = Loop(start, end, cut)
    mm = MoveMap()
    # Enables backbone/sidechain motion for that region.
    for r in range(start, end+1): mm.set_bb(r, True); mm.set_chi(r, True)
    # Applies the CCDLoopClosureMover twice to improve closure convergence — moves torsions to seal the geometric gap between loop ends.
    CCDLoopClosureMover(loop, mm).apply(pose)
    CCDLoopClosureMover(loop, mm).apply(pose)
    # Removes the cutpoint variants once closure is done.
    rosetta.core.pose.remove_variant_type_from_pose_residue(pose, VariantType.CUTPOINT_LOWER, cut)
    rosetta.core.pose.remove_variant_type_from_pose_residue(pose, VariantType.CUTPOINT_UPPER, cut+1)
    # Runs IdealizeMover to reset bond lengths/angles to ideal geometry (removes strain).
    IdealizeMover().apply(pose)

    # Three cartesian minimization stages:
    # Focused on the loop with strong coord + pair + linear_chainbreak=6.0 to crush any gap.
    # Still loop-focused, but weights relaxed (chainbreak=2.0).
    # Whole-pose light pass to settle the bundle (no chainbreak term).
    cart_min(pose, N=N, start, end, coord_w=HLH_COORD_CST_WEIGHT_STAGE1, pair_w=max(2.0, CST_WEIGHT),
        linear_w=6.0, dih_w=2.0, allow_helix_bb=False)
    cart_min(start, end, coord_w=HLH_COORD_CST_WEIGHT_STAGE2, pair_w=max(1.5, 0.75*CST_WEIGHT),
            linear_w=2.0, dih_w=1.2, allow_helix_bb=False)
    cart_min(pose, N=N, start=1, end=N, coord_w=HLH_COORD_CST_WEIGHT_STAGE3, pair_w=1.0,
            linear_w=0.0, dih_w=1.0, allow_helix_bb=False)
    # Restores a simple fold tree afterwards.
    ft2 = FoldTree(); ft2.simple_tree(N); pose.fold_tree(ft2)
    
def place(pose, a,b,phase_deg,r,z):
    # place helix segment a-b at (r,phase,z) in cylindrical coords
    th = math.radians(phase_deg)
    x,y = r*math.cos(th), r*math.sin(th)
    Rz  = _rotmat([0,0,1], phase_deg)
    cen = _segment_centroid(pose,a,b)
    _transform_pose_segment(pose,a,b,t=-cen); _transform_pose_segment(pose,a,b,R=Rz)
    _transform_pose_segment(pose,a,b,t=np.array([x,y,z]))

def build_hlhlh_single(HLH1_LEN=10,
                       HLH_LOOP1_LEN=5,
                       HLH2_LEN=10,
                       HLH_LOOP2_LEN=5,
                       HLH3_LEN=10,
                       HLH_HELIX_AA="A",
                       HLH_LOOP_AA="G",
                       HLH_PHASES_DEG=None,
                       HLH_INVERT=[False, True, False],
                       HLH_R0=4.5,
                       HLH_Z_SPACING_SCALE=0.6,
                       OMG=-180.0,
                       DO_RELAX=True,
                       RELAX_ROUNDS=5,
                       CST_WEIGHT=2.0,
                       HLH_LOOP_PER_RES_DISTANCE = 3.8,
                       HLH_END_CST_SD = 0.5,
                       HLH_COORD_CST_WEIGHT_STAGE1 = 5.0,
                       HLH_COORD_CST_WEIGHT_STAGE2 = 2.5,
                       HLH_COORD_CST_WEIGHT_STAGE3 = 1.0,
                       DO_RELAX=True):
    """
    HLHLH single chain with 3 helices packed as a tight coiled-coil bundle.
    Key fixes: turn on coordinate_constraint weight; add dense packing & torsion csts;
    staged minimization so helices don't wander while loops close.
    """

    # ---------- sequence ----------
    L1, L2, L3 = HLH1_LEN, HLH2_LEN, HLH3_LEN
    T1, T2 = HLH_LOOP1_LEN, HLH_LOOP2_LEN
    seq = (HLH_HELIX_AA * L1) + (HLH_LOOP_AA * T1) + \
          (HLH_HELIX_AA * L2) + (HLH_LOOP_AA * T2) + \
          (HLH_HELIX_AA * L3)
    pose = _pose_from_seq(seq)
    N = pose.size()

    # indices
    H1_a, H1_b = 1, L1
    L1_a, L1_b = H1_b + 1, H1_b + T1
    H2_a, H2_b = L1_b + 1, L1_b + L2
    L2_a, L2_b = H2_b + 1, H2_b + T2
    H3_a, H3_b = L2_b + 1, L2_b + L3

    # ---------- torsions ----------
    # helices
    _set_phi_psi_omega(pose, H1_a, H1_b, PHI_ALPHA, PSI_ALPHA, OMG)
    _set_phi_psi_omega(pose, H2_a, H2_b, PHI_ALPHA, PSI_ALPHA, OMG)
    _set_phi_psi_omega(pose, H3_a, H3_b, PHI_ALPHA, PSI_ALPHA, OMG)
    
    # loops to extended
    for i in range(L1_a, L1_b + 1): 
        pose.set_phi(i, -60.0)
        pose.set_psi(i, 140.0)
        
    for i in range(L2_a, L2_b + 1): 
        pose.set_phi(i, -60.0)
        pose.set_psi(i, 140.0)
  
    # Add constraints for each helix
    add_alpha_dihedral_csts(pose, H1_a, H1_b, sd=12.0)
    add_alpha_dihedral_csts(pose, H2_a, H2_b, sd=12.0)
    add_alpha_dihedral_csts(pose, H3_a, H3_b, sd=12.0)

    # ---------- align + invert (H2 antiparallel) ----------
    # align each helix axis to global Z
    _align_segment_axis_to_z(pose, H1_a, H1_b)
    _align_segment_axis_to_z(pose, H2_a, H2_b)
    _align_segment_axis_to_z(pose, H3_a, H3_b)
    
    # add detailed helix geometry constraints
    # Adds: φ/ψ/ω dihedral constraints, CA(i)-CA(i+3/4) distances,
    # and (optional) N(i+4)-O(i) pseudo-Hbond distances.
    _add_helix_geometry_constraints(pose, H1_a, H1_b, phi = PHI_ALPHA, psi = PSI_ALPHA, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)
    _add_helix_geometry_constraints(pose, H2_a, H2_b, phi = PHI_ALPHA, psi = PSI_ALPHA, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)
    _add_helix_geometry_constraints(pose, H3_a, H3_b, phi = PHI_ALPHA, psi = PSI_ALPHA, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)

    
    # invert chooses which helices to flip; default is only H2 flipped -> antiparallel relative to H1/H3.
    invert = HLH_INVERT if HLH_INVERT else [False, True, False]
    
    flip_seg(pose, H1_a,H1_b,invert[0])
    flip_seg(pose, H2_a,H2_b,invert[1])
    flip_seg(pose, H3_a,H3_b,invert[2])

    # ---------- tight tri-bundle placement ----------
    R0 = max(3.8, min(4.5, HLH_R0))  # Sets a tight bundle radius R0 clamped to [3.8, 4.5] Å.
    phases = HLH_PHASES_DEG if HLH_PHASES_DEG else [0.0, 120.0, 240.0] # Azimuthal phases default to 0°, 120°, 240° (triangular symmetry).
    rise = 1.5 * (HLH_Z_SPACING_SCALE if 'HLH_Z_SPACING_SCALE' in globals() else 0.6) # per-residue axial spacing
    h1h, h2h, h3h = 0.5*L1*rise, 0.5*L2*rise, 0.5*L3*rise 

    # Determines each helix’s direction along Z
    s1, s2, s3 = axis_sign(pose, H1_a,H1_b), axis_sign(pose, H2_a,H2_b), axis_sign(pose, H3_a,H3_b)


    #Computes axial positions (Z) so helix ends meet loop starts; small ±0.6 Å nudges to avoid clashes/overlap.
    z1 = 0.0
    z2 = (z1 + C_ofs(s1,h1h)) - N_ofs(s2,h2h)
    z3 = (z2 + C_ofs(s2,h2h)) - N_ofs(s3,h3h)
    z2 += 0.6 * (1 if z2>=z1 else -1)
    z3 += 0.6 * (1 if z3>=z2 else -1)
    
    # place helices
    place(pose, H1_a,H1_b,phases[0],R0,z1)
    place(pose, H2_a,H2_b,phases[1],R0,z2)
    place(pose, H3_a,H3_b,phases[2],R0,z3)

    for (a,b,p) in [(H1_a,H1_b,phases[0]), (H2_a,H2_b,phases[1]), (H3_a,H3_b,phases[2])]:
        microtilt(pose,a,b,p,deg=5.0)

    # ---------- constraints ----------
    # loop-end targets
    # Adds distance constraints between helix ends and the next helix starts:
    # Target distance ≈ loop_length × per_residue_loop_distance → encourages loops to bridge without overstretch.
    # HarmonicFunc gives a quadratic penalty around the target, width set by HLH_END_CST_SD.
    # Define defaults for per-residue loop distance and constraint width if not provided.

    if T1>0: 
        pose.add_constraint(AtomPairConstraint(ca(pose, H1_b), ca(pose, H2_a),
                        HarmonicFunc(T1*HLH_LOOP_PER_RES_DISTANCE, HLH_END_CST_SD)))
    if T2>0: 
        pose.add_constraint(AtomPairConstraint(ca(pose, H2_b), ca(pose, H3_a),
                        HarmonicFunc(T2*HLH_LOOP_PER_RES_DISTANCE, HLH_END_CST_SD)))


    add_pair_grid(pose, H1_a,H1_b,H2_a,H2_b, target=8.4, sd=0.4)
    add_pair_grid(pose, H2_a,H2_b,H3_a,H3_b, target=8.4, sd=0.4)
    add_pair_grid(pose, H1_a,H1_b,H3_a,H3_b, target=8.4, sd=0.4)

    # strong coordinate anchors on helices (this needed a score weight!)
    # Adds coordinate constraints (to reference positions) along every residue of each helix (stride 1), tight sd=0.25 Å.
    # Keeps helices from wandering while loops move/close and during minimization.
    _add_coordinate_anchors_for_segment(pose, H1_a, H1_b, stride=1, sd=0.25)
    _add_coordinate_anchors_for_segment(pose, H2_a, H2_b, stride=1, sd=0.25)
    _add_coordinate_anchors_for_segment(pose, H3_a, H3_b, stride=1, sd=0.25)


    # Closes loop 1 and loop 2 with the robust procedure above.
    close_loop(L1_a, L1_b)
    close_loop(L2_a, L2_b)

    # One more whole-pose polish while coordinate anchors remain strong → maintains tight packing.
    cart_min(1, N, coord_w=HLH_COORD_CST_WEIGHT_STAGE3, pair_w=1.0, linear_w=0.0, allow_helix_bb=False)

    # Optional FastRelax in cartesian mode, with moderate constraint weights, 
    # for gentle all-atom refinement without breaking the bundle geometry.
    if DO_RELAX:
        sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015_cart")
        sfxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 1.0)
        sfxn.set_weight(rosetta.core.scoring.coordinate_constraint, 0.5)
        fr = rosetta.protocols.relax.FastRelax(sfxn, max(1, RELAX_ROUNDS//2))
        fr.cartesian(True)
        fr.apply(pose)

    print("[OK] HLHLH: tight tri-bundle; loops closed (no chainbreak).")
    return pose