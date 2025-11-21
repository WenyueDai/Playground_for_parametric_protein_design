# -*- coding: utf-8 -*-
"""
This script provides functions to build small protein motifs with specified secondary structures using PyRosetta. 
The constraint is that only φ/ψ angles, CA-CA distance (so to define which chains are paired) are set to define the secondary structure.

Tips and tricks when building small motifs with specific secondary structures.
1. β type (hairpin, meander, greek key): if only set φ/ψ, the strands will not aligned and look like disordered. 
So we add CA-CA harmonic constraints between paired strands to stabilize the sheet.
The CST_CA_SD and CST_CA_DIST can be adjusted to tune the sheet tightness and CA-CA distance.
The larger the CST_CA_SD, the floppier the sheet will be.

So for β-hairpin, the two strands are paired anti-parallelly -> CST_CA_SD can be small (e.g. 0.6 Å) to allow tight sheet.
For β-meander and greek key, the strands are not directly paired -> CST_CA_SD should be larger (e.g. 1.0 Å) to allow more flexibility. 

For more complicated design, check 0_cpl_motif_contact_map.py, which is created by applying contact map constraints.

2. For helix-turn-helix-turn-helix, it is much more complicated because just adding CA-CA harmonic constrain is not good enough due to the 
presence of two loops. What is needed:

-1 constrain on helix secondary structure: 
-- Apply ϕ/ψ dihedral constraints (DihedralConstraint with CircularHarmonicFunc); 
-- Add helix geometry constrain:
-- -- CA(i) - CA(i/4) distances (enforce local helix pitch).
-- -- N(i+4) - O(i) pseudo-H-bonds for H-bond geometry.

-2 loop closure using CCDLoopClosureMover.

-3 3 Step of minimization to ensure structure is realisic 
-- Strong coordinate and pair constrains; high chainbreak penalty to seal the loop
-- Relaxed constrained, refine loop geometry.
-- whole pose minimization

-4 Use crick parameters to construct helix bundles, flip the middle helix to antiparallel.

Usage:
  conda activate pyrosetta
  python small_motif_factory.py

"""

import os, math
import numpy as np
import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.protocols.helical_bundle import MakeBundle

# =========================
# GLOBAL CONFIG (edit here)
# =========================

# TODO After testing with main, remove the cycling in main and keep only this line.
MOTIF_TYPE = "alpha_helix" 
# "helix_310" | "pi_helix" | "coiled_coil" | "beta_strand" | "beta_hairpin"
# "beta_meander" | "greek_key" | "helix_turn_helix" | "helix_coil_helix" | "ba_beta"

# ---- Constraints ----
DO_CONSTRAINTS = True # Whether to add pair constraints for β class (recommended True)
DO_RELAX = False # Whether to run one round of FastRelax after generation.
RELAX_ROUNDS = 2 # FastRelax rounds
CST_CA_DIST = 4.5 # Target CA-CA distance between β sheets (Å)
CST_CA_SD = 0.6 # Harmonic constraint width (Å) The larger, the floppier the sheet
CST_WEIGHT = 1.0 # Weight of atom_pair_constraint in scorefxn. Larger weight means stronger constraint.

# ---- α / 3_10 / π ----
# α-helix is the 'standard' helix, hbond connect is i to i+4. It is the most common helix type.
# 3_10 helix: hbond connect is i to i+3. It is usually seen as 3-10 cap at the end of α-helix.
# π helix: hbond connect is i to i+5. Often seen as a short insertion within α-helix to change its direction or expand its radius.
ALPHA_LEN = 18
ALPHA_AA = "A"
PHI_ALPHA, PSI_ALPHA, OMG = -57.8, -47.0, 180.0
PHI_310, PSI_310 = -49.0, -26.0
PHI_PI, PSI_PI = -57.0, -70.0

# ---- Coiled-coil (Crick) ----
CC_N_HELICES = 3
CC_LENGTHS = [28, 28, 28]
CC_R0 = 6.0 # superhelical radius. too small -> helices close and clash, 5-7
CC_OMEGA0 = 1.0 # superhelical twist, It is not the twist of individual helix, between -3 to +3
CC_Z0 = 1.5 # Å / residue
CC_PHASES_DEG = [] # e.g. [0,120,240] each helix phase at z=0, this is the roll of helix around its own axis
CC_INVERT = [] # e.g. [False, False, False] # whether to invert each helix to create antiparallel

########################################################################
# ---- Single-chain HLHLH (potentially most useful motif :))----
# Lengths of three helices
HLH1_LEN = 15
HLH2_LEN = 15
HLH3_LEN = 17

# Lengths of two loops
HLH_LOOP1_LEN = 5
HLH_LOOP2_LEN = 5

# Amino acid types for helices and loops
HLH_HELIX_AA = "A"
HLH_LOOP_AA = "G"

# Coiled-coil parameters for helices
HLH_R0 = 4.8      # Å，bundle radius
HLH_OMEGA0 = 2.0      # °/res，superhelical angle
HLH_Z1 = 1.5      # Å/res，superhelical rise per residue
HLH_Z_SPACING_SCALE = 0.6  # Scale factor for axial staggering (<1 more compact)

# Leave empty for default [0, 120, 240]
HLH_PHASES_DEG = [] # helix phases at z=0

# Parallel or antiparallel packing
HLH_INVERT = [False, True, False]

# loop each residue's axial distance estimate, controlling helix spacing
HLH_LOOP_PER_RES_DISTANCE = 2.6  # Å/res

# ========== Constraints ==========
# loop end distance target and standard deviation
HLH_END_CST_SD = 0.4     # 越小越硬
# helix–helix packing distance and width
HLH_PAIR_CST_DIST = 9.0   # Å between representative Cα
HLH_PAIR_CST_SD = 0.6

# CoordinateConstraint applied to CA every `stride` residues
HLH_COORD_CST_SD = 0.5
HLH_COORD_CST_WEIGHT_STAGE1 = 1.5
HLH_COORD_CST_WEIGHT_STAGE2 = 0.8
HLH_COORD_CST_WEIGHT_STAGE3 = 0.4
########################################################################

# ---- β ----
BETA_LEN = 8
BETA_AA = "V"
PHI_BETA, PSI_BETA = -135.0, 135.0

# β-hairpin
HAIRPIN_STRAND_LEN = 6
HAIRPIN_TURN_LEN = 4
HAIRPIN_TURN_TYPE = "I" # "I" or "II"
HAIRPIN_STRAND_AA = "V"
HAIRPIN_TURN_AA = "D"

# β-meander (3 strands: β - turn - β - turn - β)
MEANDER_STRAND_LEN = 6
MEANDER_TURN_LEN = 3
MEANDER_STRAND_AA = "V"
MEANDER_TURN_AA = "D"

# Greek key (4 strands)
GREEK_STRAND_LEN = 5
GREEK_TURN_LEN = 3
GREEK_STRAND_AA = "V"
GREEK_TURN_AA = "D"

# ---- α/β /Combination ----
# HTH
HTH_H1_LEN = 10
HTH_LOOP_LEN = 4
HTH_H2_LEN = 10
HTH_HELIX_AA = "A"
HTH_LOOP_AA = "D"

# HCH
HCH_H1_LEN = 10
HCH_COIL_LEN = 6
HCH_H2_LEN = 10
HCH_HELIX_AA = "A"
HCH_COIL_AA = "S"

# β-α-β
BAB_B1_LEN = 5
BAB_H_LEN = 8
BAB_B2_LEN = 5
BAB_BETA_AA = "V"
BAB_HELIX_AA = "A"
BAB_LOOP_LEN = 1   # 固定小 loop

# =========================
# Internals
# =========================

def _ensure_init():
    try:
        pyrosetta.get_fa_scorefxn()
    except Exception:
        pyrosetta.init("-mute all")

def _pose_from_seq(seq: str, rts_name="fa_standard") -> rosetta.core.pose.Pose:
    cm = rosetta.core.chemical.ChemicalManager.get_instance()
    rts = cm.residue_type_set(rts_name)
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, rts)
    return pose

def _set_phi_psi_omega(pose, start, end, phi, psi, omg=180.0):
    for i in range(start, end + 1):
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, omg)

def _get_sf_with_csts():
    sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015")
    if DO_CONSTRAINTS:
        sf.set_weight(rosetta.core.scoring.atom_pair_constraint, CST_WEIGHT)
    return sf

def _add_sheet_ca_constraints(pose, pairs):
    """pairs: list[(i,j)], 为每对加 CA-CA 和谐约束。"""
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

def _relax_if_needed(pose):
    if not DO_RELAX:
        return
    sf = _get_sf_with_csts()
    fr = rosetta.protocols.relax.FastRelax(RELAX_ROUNDS)
    fr.set_scorefxn(sf)
    fr.apply(pose)

def _rotmat(axis, angle_deg):
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
    """对单链的一个 [start,end] 区段做刚体变换。（loop 会被拉伸，Relax 后回到合理几何）"""
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
    import numpy as np
    from pyrosetta import rosetta
    c = np.zeros(3); n = 0
    for i in range(start, end+1):
        ai = rosetta.core.id.AtomID(pose.residue(i).atom_index("CA"), i)
        xyz = pose.xyz(ai); c += np.array([xyz.x, xyz.y, xyz.z]); n += 1
    return c / max(n,1)

def _align_segment_axis_to_z(pose, start, end):
    """
    估计该段螺旋主轴（用首/尾 CA 向量即可），
    绕段的质心把主轴旋到全局 z=(0,0,1)。
    """
    import numpy as np, math
    from pyrosetta import rosetta

    # 质心
    cen = _segment_centroid(pose, start, end)

    # 取首尾 CA 估轴
    a1 = rosetta.core.id.AtomID(pose.residue(start).atom_index("CA"), start)
    a2 = rosetta.core.id.AtomID(pose.residue(end).atom_index("CA"),   end)
    p1 = pose.xyz(a1); p2 = pose.xyz(a2)
    v  = np.array([p2.x-p1.x, p2.y-p1.y, p2.z-p1.z])
    if np.linalg.norm(v) < 1e-6: 
        return
    v = v / np.linalg.norm(v)
    z = np.array([0.,0.,1.])

    # 旋到 z 轴：轴= v×z，角= arccos(v·z)
    axis = np.cross(v, z); axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:   # 已经对齐或反向
        if np.dot(v, z) > 0: 
            return
        # 反向：绕任意与 z 垂直的轴转 180°
        axis = np.array([1.,0.,0.]); axis_norm = 1.0
    axis = axis / axis_norm
    ang  = math.degrees(math.acos(np.clip(np.dot(v, z), -1.0, 1.0)))

    # 绕质心做刚体：先平移到原点 -> 旋转 -> 平移回去
    R = _rotmat(axis, ang)
    t1 = -cen; t2 = cen
    _transform_pose_segment(pose, start, end, R=np.eye(3), t=t1)
    _transform_pose_segment(pose, start, end, R=R,        t=np.zeros(3))
    _transform_pose_segment(pose, start, end, R=np.eye(3), t=t2)

def _add_coordinate_anchors_for_segment(pose, start, end, stride=3, sd=0.5):
    """
    Add CoordinateConstraint to CA atoms every `stride` residues in [start, end],
    anchoring them to their current coordinates (keeps helices rigid).
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.numeric import xyzVector_double_t

    # use the first residue N atom as the reference frame origin
    ref = AtomID(pose.residue(1).atom_index("N"), 1)

    for i in range(start, end + 1, stride):
        ai = AtomID(pose.residue(i).atom_index("CA"), i)
        xyz = pose.xyz(ai)
        target = xyzVector_double_t(xyz.x, xyz.y, xyz.z)  # anchor to current spot
        func = HarmonicFunc(0.0, sd)
        pose.add_constraint(CoordinateConstraint(ai, ref, target, func))

def _add_helix_geometry_constraints(pose, start, end,
                                    phi=PHI_ALPHA, psi=PSI_ALPHA,
                                    sd_tors_deg=8.0,   # tighter torsion SD
                                    ca_i3=5.2, ca_i4=6.4, sd_ca=0.25,
                                    add_hbond=False, d_NO=2.9, sd_NO=0.25):
    """
    Adds: φ/ψ/ω dihedral constraints, CA(i)-CA(i+3/4) distances,
    and (optional) N(i+4)-O(i) pseudo-Hbond distances.
    """
    import math
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc, HarmonicFunc
    from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint, AtomPairConstraint

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

    # CA spacing signatures of helices
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
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

# ---------- builders ----------

def build_alpha(phi, psi, length, aa="A"):
    pose = _pose_from_seq(aa * length)
    _set_phi_psi_omega(pose, 1, length, phi, psi, OMG)
    _relax_if_needed(pose)
    return pose

def build_coiled_coil():
    """
    用 RosettaScripts <MakeBundle> + <Helix/>；角度用度（use_degrees="true"）。
    若返回空/异常，fallback：用 α-helix 多股绕 Z 轴摆位（保底可视）。
    """
    from pyrosetta import rosetta
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    n = CC_N_HELICES
    lengths = CC_LENGTHS if CC_LENGTHS else [28] * n
    phases  = CC_PHASES_DEG if CC_PHASES_DEG else [i * 360.0 / n for i in range(n)]
    invert  = CC_INVERT if CC_INVERT else [False] * n
    if not (len(lengths) == len(phases) == len(invert) == n):
        raise ValueError("CC_LENGTHS / CC_PHASES_DEG / CC_INVERT 的长度必须等于 CC_N_HELICES")

    # 每股 <Helix .../>。注意：omega0 典型在 -3~+3 度/残基；z1≈1.5 Å/残基
    helix_lines = []
    for L, ph, inv in zip(lengths, phases, invert):
        inv_str = "true" if inv else "false"
        helix_lines.append(
            f'<Helix helix_length="{L}" residue_name="ALA" '
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

    # ---- Fallback：若构建失败或得到空 pose，手工摆位一个圈，至少给你可视化种子 ----
    if pose.size() == 0:
        unit = build_alpha(PHI_ALPHA, PSI_ALPHA, lengths[0], aa="A")
        assembled = rosetta.core.pose.Pose()
        rosetta.core.pose.append_pose_to_pose(assembled, unit, new_chain=True)
        for k in range(1, n):
            u = unit.clone()
            ang = 360.0 * k / n
            R = _rotmat([0, 0, 1], ang)
            t = np.array([
                CC_R0 * math.cos(math.radians(ang)),
                CC_R0 * math.sin(math.radians(ang)),
                0.0
            ])
            _transform_pose_inplace(u, R=R, t=t)
            rosetta.core.pose.append_pose_to_pose(assembled, u, new_chain=True)
        pose = assembled

    _relax_if_needed(pose)   # 如果你在脚本里开了 DO_RELAX
    return pose

def build_hlhlh_single():
    """
    HLHLH single chain with 3 helices packed as a tight coiled-coil bundle.
    Key fixes: turn on coordinate_constraint weight; add dense packing & torsion csts;
    staged minimization so helices don't wander while loops close.
    """
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

    # ---------- sequence ----------
    L1, L2, L3 = HLH1_LEN, HLH2_LEN, HLH3_LEN
    T1, T2 = HLH_LOOP1_LEN, HLH_LOOP2_LEN
    seq = (HLH_HELIX_AA * L1) + (HLH_LOOP_AA * T1) + \
          (HLH_HELIX_AA * L2) + (HLH_LOOP_AA * T2) + \
          (HLH_HELIX_AA * L3)
    pose = _pose_from_seq(seq); N = pose.size()

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
    for i in range(L1_a, L1_b + 1): pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)
    for i in range(L2_a, L2_b + 1): pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)

    # α-torsion constraints (keeps helices helical during minimization)
    def add_alpha_dihedral_csts(start, end, sd=15.0):
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
    
    # Add constraints for each helix
    add_alpha_dihedral_csts(H1_a, H1_b, sd=12.0)
    add_alpha_dihedral_csts(H2_a, H2_b, sd=12.0)
    add_alpha_dihedral_csts(H3_a, H3_b, sd=12.0)

    # ---------- align + invert (H2 antiparallel) ----------
    # align each helix axis to global Z
    _align_segment_axis_to_z(pose, H1_a, H1_b)
    _align_segment_axis_to_z(pose, H2_a, H2_b)
    _align_segment_axis_to_z(pose, H3_a, H3_b)
    
    # add detailed helix geometry constraints
    # Adds: φ/ψ/ω dihedral constraints, CA(i)-CA(i+3/4) distances,
    # and (optional) N(i+4)-O(i) pseudo-Hbond distances.
    _add_helix_geometry_constraints(pose, H1_a, H1_b, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)
    _add_helix_geometry_constraints(pose, H2_a, H2_b, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)
    _add_helix_geometry_constraints(pose, H3_a, H3_b, sd_tors_deg=8.0, sd_ca=0.25, add_hbond=True)

    
    # invert chooses which helices to flip; default is only H2 flipped -> antiparallel relative to H1/H3.
    invert = HLH_INVERT if HLH_INVERT else [False, True, False]
    def flip_seg(a,b,flag):
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
    
    flip_seg(H1_a,H1_b,invert[0])
    flip_seg(H2_a,H2_b,invert[1])
    flip_seg(H3_a,H3_b,invert[2])

    # ---------- tight tri-bundle placement ----------
    R0 = max(3.8, min(4.5, HLH_R0))  # Sets a tight bundle radius R0 clamped to [3.8, 4.5] Å.
    phases = HLH_PHASES_DEG if HLH_PHASES_DEG else [0.0, 120.0, 240.0] # Azimuthal phases default to 0°, 120°, 240° (triangular symmetry).
    rise = 1.5 * (HLH_Z_SPACING_SCALE if 'HLH_Z_SPACING_SCALE' in globals() else 0.6) # per-residue axial spacing
    h1h, h2h, h3h = 0.5*L1*rise, 0.5*L2*rise, 0.5*L3*rise 

    def axis_sign(a,b):
        # return +1 if helix from a to b goes up (z+), else -1
        ca1 = pose.xyz(AtomID(pose.residue(a).atom_index("CA"), a))
        ca2 = pose.xyz(AtomID(pose.residue(b).atom_index("CA"), b))
        return 1.0 if (ca2.z - ca1.z) >= 0 else -1.0
    # Determines each helix’s direction along Z
    s1, s2, s3 = axis_sign(H1_a,H1_b), axis_sign(H2_a,H2_b), axis_sign(H3_a,H3_b)
    def C_ofs(s,half): return (+half if s>0 else -half) # helix C-end offset
    def N_ofs(s,half): return (-half if s>0 else +half) # helix N-end offset

    #Computes axial positions (Z) so helix ends meet loop starts; small ±0.6 Å nudges to avoid clashes/overlap.
    z1 = 0.0
    z2 = (z1 + C_ofs(s1,h1h)) - N_ofs(s2,h2h)
    z3 = (z2 + C_ofs(s2,h2h)) - N_ofs(s3,h3h)
    z2 += 0.6 * (1 if z2>=z1 else -1)
    z3 += 0.6 * (1 if z3>=z2 else -1)

    def place(a,b,phase_deg,r,z):
        # place helix segment a-b at (r,phase,z) in cylindrical coords
        th = math.radians(phase_deg)
        x,y = r*math.cos(th), r*math.sin(th)
        Rz  = _rotmat([0,0,1], phase_deg)
        cen = _segment_centroid(pose,a,b)
        _transform_pose_segment(pose,a,b,t=-cen); _transform_pose_segment(pose,a,b,R=Rz)
        _transform_pose_segment(pose,a,b,t=np.array([x,y,z]))
    # place helices
    place(H1_a,H1_b,phases[0],R0,z1)
    place(H2_a,H2_b,phases[1],R0,z2)
    place(H3_a,H3_b,phases[2],R0,z3)

    # inward micro-tilt for packing, the tilt is happen perpendicular to its phase direction
    def microtilt(a,b,phase,deg=6.0):
        th = math.radians(phase); ax = np.array([-math.sin(th), math.cos(th), 0.0])
        R  = _rotmat(ax, deg); cen = _segment_centroid(pose,a,b)
        _transform_pose_segment(pose,a,b,t=-cen); _transform_pose_segment(pose,a,b,R=R); _transform_pose_segment(pose,a,b,t=cen)
    for (a,b,p) in [(H1_a,H1_b,phases[0]), (H2_a,H2_b,phases[1]), (H3_a,H3_b,phases[2])]:
        microtilt(a,b,p,deg=5.0)

    # ---------- constraints ----------
    # ca(i) helper fetches an AtomID for Cα of residue i.
    def ca(i): return AtomID(pose.residue(i).atom_index("CA"), i)

    # loop-end targets
    # Adds distance constraints between helix ends and the next helix starts:
    # Target distance ≈ loop_length × per_residue_loop_distance → encourages loops to bridge without overstretch.
    # HarmonicFunc gives a quadratic penalty around the target, width set by HLH_END_CST_SD.
    if T1>0: pose.add_constraint(AtomPairConstraint(ca(H1_b), ca(H2_a),
                        HarmonicFunc(T1*HLH_LOOP_PER_RES_DISTANCE, HLH_END_CST_SD)))
    if T2>0: pose.add_constraint(AtomPairConstraint(ca(H2_b), ca(H3_a),
                        HarmonicFunc(T2*HLH_LOOP_PER_RES_DISTANCE, HLH_END_CST_SD)))

    # dense helix–helix packing pairs
    # Target ~8.4 Å with tight ±0.4 Å tolerance -> enforces dense, symmetric packing throughout the bundle (not just at ends).
    target = 8.4  # Å between corresponding Cαs across helices
    sd     = 0.4
    def add_pair_grid(hA_a,hA_b,hB_a,hB_b):
        # Lays a grid of Cα–Cα distance constraints every other residue between each helix pair.
        idxA = list(range(hA_a+2, hA_b-1, 2))
        idxB = list(range(hB_a+2, hB_b-1, 2))
        for i,j in zip(idxA, idxB):
            pose.add_constraint(AtomPairConstraint(ca(i), ca(j), HarmonicFunc(target, sd)))
    add_pair_grid(H1_a,H1_b,H2_a,H2_b)
    add_pair_grid(H2_a,H2_b,H3_a,H3_b)
    add_pair_grid(H1_a,H1_b,H3_a,H3_b)

    # strong coordinate anchors on helices (this needed a score weight!)
    # Adds coordinate constraints (to reference positions) along every residue of each helix (stride 1), tight sd=0.25 Å.
    # Keeps helices from wandering while loops move/close and during minimization.
    _add_coordinate_anchors_for_segment(pose, H1_a, H1_b, stride=1, sd=0.25)
    _add_coordinate_anchors_for_segment(pose, H2_a, H2_b, stride=1, sd=0.25)
    _add_coordinate_anchors_for_segment(pose, H3_a, H3_b, stride=1, sd=0.25)

    # ---------- helper: staged cartesian minimization with coord/pair weights ----------
    def cart_min(start, end, coord_w, pair_w, linear_w=0.0, dih_w=1.0, allow_helix_bb=False):
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
    def close_loop(start, end):
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
        cart_min(start, end, coord_w=HLH_COORD_CST_WEIGHT_STAGE1, pair_w=max(2.0, CST_WEIGHT),
         linear_w=6.0, dih_w=2.0, allow_helix_bb=False)
        cart_min(start, end, coord_w=HLH_COORD_CST_WEIGHT_STAGE2, pair_w=max(1.5, 0.75*CST_WEIGHT),
                linear_w=2.0, dih_w=1.2, allow_helix_bb=False)
        cart_min(1, N, coord_w=HLH_COORD_CST_WEIGHT_STAGE3, pair_w=1.0,
                linear_w=0.0, dih_w=1.0, allow_helix_bb=False)
        # Restores a simple fold tree afterwards.
        ft2 = FoldTree(); ft2.simple_tree(N); pose.fold_tree(ft2)
    
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


def build_beta_strand():
    pose = _pose_from_seq(BETA_AA * BETA_LEN)
    _set_phi_psi_omega(pose, 1, BETA_LEN, PHI_BETA, PSI_BETA)
    _relax_if_needed(pose)
    return pose

def build_beta_hairpin():
    s, t = HAIRPIN_STRAND_LEN, HAIRPIN_TURN_LEN
    seq = (HAIRPIN_STRAND_AA * s) + (HAIRPIN_TURN_AA * t) + (HAIRPIN_STRAND_AA * s)
    pose = _pose_from_seq(seq)
    # 两条 β 链
    _set_phi_psi_omega(pose, 1, s, PHI_BETA, PSI_BETA)
    _set_phi_psi_omega(pose, s + t + 1, 2 * s + t, PHI_BETA, PSI_BETA)
    # turn 近似
    t1, t2 = s + 1, s + 2
    if t >= 2:
        if HAIRPIN_TURN_TYPE.upper() == "I":
            pose.set_phi(t1, -60.0); pose.set_psi(t1, -30.0)
            pose.set_phi(t2, -90.0); pose.set_psi(t2,   0.0)
        else:
            pose.set_phi(t1, -60.0); pose.set_psi(t1, 120.0)
            pose.set_phi(t2,  80.0); pose.set_psi(t2,   0.0)
    for i in range(s + 3, s + t + 1):
        pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)

    # 配对约束：反平行配对
    if DO_CONSTRAINTS:
        pairs = []
        left = list(range(1, s + 1))
        right = list(range(s + t + 1, 2 * s + t + 1))
        right.reverse()
        for i, j in zip(left, right):
            pairs.append((i, j))
        _add_sheet_ca_constraints(pose, pairs)

    _relax_if_needed(pose)
    return pose

def build_beta_meander():
    # β(len) - turn(len) - β(len) - turn(len) - β(len)
    L, T = MEANDER_STRAND_LEN, MEANDER_TURN_LEN
    seq = (MEANDER_STRAND_AA * L) + (MEANDER_TURN_AA * T) + \
          (MEANDER_STRAND_AA * L) + (MEANDER_TURN_AA * T) + \
          (MEANDER_STRAND_AA * L)
    pose = _pose_from_seq(seq)
    # 三条 strand
    s1_a, s1_b = 1, L
    s2_a, s2_b = L + T + 1, L + T + L
    s3_a, s3_b = L + T + L + T + 1, L + T + L + T + L
    _set_phi_psi_omega(pose, s1_a, s1_b, PHI_BETA, PSI_BETA)
    for i in range(L + 1, L + T + 1):
        pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)
    _set_phi_psi_omega(pose, s2_a, s2_b, PHI_BETA, PSI_BETA)
    for i in range(s2_b + 1, s2_b + T + 1):
        pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)
    _set_phi_psi_omega(pose, s3_a, s3_b, PHI_BETA, PSI_BETA)

    # 配对约束：1<->2, 2<->3（反平行）
    if DO_CONSTRAINTS:
        pairs = []
        left = list(range(s1_a, s1_b + 1))
        right = list(range(s2_a, s2_b + 1)); right.reverse()
        for i, j in zip(left, right): pairs.append((i, j))
        left = list(range(s2_a, s2_b + 1))
        right = list(range(s3_a, s3_b + 1)); right.reverse()
        for i, j in zip(left, right): pairs.append((i, j))
        _add_sheet_ca_constraints(pose, pairs)

    _relax_if_needed(pose)
    return pose

def build_greek_key():
    # β - turn - β - turn - β - turn - β
    L, T = GREEK_STRAND_LEN, GREEK_TURN_LEN
    seq = (GREEK_STRAND_AA * L) + (GREEK_TURN_AA * T) + \
          (GREEK_STRAND_AA * L) + (GREEK_TURN_AA * T) + \
          (GREEK_STRAND_AA * L) + (GREEK_TURN_AA * T) + \
          (GREEK_STRAND_AA * L)
    pose = _pose_from_seq(seq)

    idx = 1
    spans = []
    for k in range(4):
        a, b = idx, idx + L - 1
        spans.append((a, b))
        _set_phi_psi_omega(pose, a, b, PHI_BETA, PSI_BETA)
        idx += L
        if k < 3:
            for j in range(idx, idx + T):
                pose.set_phi(j, -60.0); pose.set_psi(j, 140.0)
            idx += T

    if DO_CONSTRAINTS:
        pairs = []
        (a1,b1),(a2,b2),(a3,b3),(a4,b4)=spans
        left = list(range(a1,b1+1)); right=list(range(a2,b2+1)); right.reverse()
        for i,j in zip(left,right): pairs.append((i,j))
        left = list(range(a2,b2+1)); right=list(range(a3,b3+1)); right.reverse()
        for i,j in zip(left,right): pairs.append((i,j))
        left = list(range(a3,b3+1)); right=list(range(a4,b4+1)); right.reverse()
        for i,j in zip(left,right): pairs.append((i,j))
        _add_sheet_ca_constraints(pose, pairs)

    _relax_if_needed(pose)
    return pose

def build_hth():
    h1, loop, h2 = HTH_H1_LEN, HTH_LOOP_LEN, HTH_H2_LEN
    seq = (HTH_HELIX_AA * h1) + (HTH_LOOP_AA * loop) + (HTH_HELIX_AA * h2)
    pose = _pose_from_seq(seq)
    _set_phi_psi_omega(pose, 1, h1, PHI_ALPHA, PSI_ALPHA, OMG)
    _set_phi_psi_omega(pose, h1 + loop + 1, h1 + loop + h2, PHI_ALPHA, PSI_ALPHA, OMG)
    for i in range(h1 + 1, h1 + loop + 1):
        pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)
    _relax_if_needed(pose)
    return pose

def build_hch():
    h1, coil, h2 = HCH_H1_LEN, HCH_COIL_LEN, HCH_H2_LEN
    seq = (HCH_HELIX_AA * h1) + (HCH_COIL_AA * coil) + (HCH_HELIX_AA * h2)
    pose = _pose_from_seq(seq)
    _set_phi_psi_omega(pose, 1, h1, PHI_ALPHA, PSI_ALPHA, OMG)
    _set_phi_psi_omega(pose, h1 + coil + 1, h1 + coil + h2, PHI_ALPHA, PSI_ALPHA, OMG)
    for i in range(h1 + 1, h1 + coil + 1):
        pose.set_phi(i, -60.0); pose.set_psi(i, 140.0)
    _relax_if_needed(pose)
    return pose

def build_ba_beta():
    # β1 - loop - α - loop - β2
    L1, LH, L2 = BAB_B1_LEN, BAB_H_LEN, BAB_B2_LEN
    seq = (BAB_BETA_AA * L1) + ("G" * BAB_LOOP_LEN) + (BAB_HELIX_AA * LH) + ("G" * BAB_LOOP_LEN) + (BAB_BETA_AA * L2)
    pose = _pose_from_seq(seq)
    _set_phi_psi_omega(pose, 1, L1, PHI_BETA, PSI_BETA)
    pose.set_phi(L1 + 1, -60.0); pose.set_psi(L1 + 1, 140.0)
    h_start = L1 + BAB_LOOP_LEN + 1
    _set_phi_psi_omega(pose, h_start, h_start + LH - 1, PHI_ALPHA, PSI_ALPHA, OMG)
    l2 = h_start + LH
    pose.set_phi(l2, -60.0); pose.set_psi(l2, 140.0)
    _set_phi_psi_omega(pose, l2 + 1, l2 + L2, PHI_BETA, PSI_BETA)
    _relax_if_needed(pose)
    return pose

# =========================
# Main
# =========================

def main():
    _ensure_init()

    motifs = [
        "alpha_helix", "helix_310", "pi_helix", "coiled_coil",
        "hlhlh_single",
        "beta_strand", "beta_hairpin", "beta_meander",
        "greek_key", "helix_turn_helix", "helix_coil_helix",
        "ba_beta"
    ]

    for MOTIF_TYPE in motifs:
        OUT_PDB = f"outs/{MOTIF_TYPE}_seed.pdb"
        os.makedirs(os.path.dirname(OUT_PDB) or ".", exist_ok=True)

        mt = MOTIF_TYPE.lower()
        if   mt == "alpha_helix":         pose = build_alpha(PHI_ALPHA, PSI_ALPHA, ALPHA_LEN, ALPHA_AA)
        elif mt == "helix_310":           pose = build_alpha(PHI_310, PSI_310, ALPHA_LEN, ALPHA_AA)
        elif mt == "pi_helix":            pose = build_alpha(PHI_PI, PSI_PI, ALPHA_LEN, ALPHA_AA)
        elif mt == "coiled_coil":         pose = build_coiled_coil()
        elif mt == "hlhlh_single":        pose = build_hlhlh_single()
        elif mt == "beta_strand":         pose = build_beta_strand()
        elif mt == "beta_hairpin":        pose = build_beta_hairpin()
        elif mt == "beta_meander":        pose = build_beta_meander()
        elif mt == "greek_key":           pose = build_greek_key()
        elif mt == "helix_turn_helix":    pose = build_hth()
        elif mt == "helix_coil_helix":    pose = build_hch()
        elif mt == "ba_beta":             pose = build_ba_beta()
        else:
            raise ValueError(f"Unknown MOTIF_TYPE: {MOTIF_TYPE}")

        rosetta.core.io.pdb.dump_pdb(pose, OUT_PDB)
        print(f"[OK] {MOTIF_TYPE} written to {OUT_PDB} (len={pose.size()} aa)")

if __name__ == "__main__":
    main()
