# -*- coding: utf-8 -*-
"""
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

# Choose one (main() 会遍历全部，这里仅供单跑时参考)
MOTIF_TYPE = "alpha_helix"
# "helix_310" | "pi_helix" | "coiled_coil" | "beta_strand" | "beta_hairpin"
# "beta_meander" | "greek_key" | "helix_turn_helix" | "helix_coil_helix" | "ba_beta"

# ---- 稳定化与输出控制 ----
DO_CONSTRAINTS = True     # β 类是否加配对约束（建议 True）
DO_RELAX       = False     # 生成后是否跑一轮 FastRelax（建议 True）
RELAX_ROUNDS   = 2        # FastRelax 轮数
CST_CA_DIST    = 4.8      # β 片层间目标 CA-CA 距离(Å)
CST_CA_SD      = 0.6      # 和谐约束宽度(Å)
CST_WEIGHT     = 1.0      # scorefxn 中 atom_pair_constraint 权重

# ---- α / 3_10 / π ----
ALPHA_LEN      = 18
ALPHA_AA       = "A"
PHI_ALPHA, PSI_ALPHA, OMG = -57.8, -47.0, 180.0
PHI_310,   PSI_310        = -49.0, -26.0
PHI_PI,    PSI_PI         = -57.0, -70.0

# ---- Coiled-coil (Crick) ----
CC_N_HELICES  = 3
CC_LENGTHS    = [28, 28, 28]
CC_R0         = 6.0       # Å (半径；fallback 时作为摆位半径)
CC_OMEGA0     = 102.0     # deg / residue
CC_Z0         = 1.5       # Å / residue
# 等间隔相位（为空则自动）：
CC_PHASES_DEG = []        # e.g. [0,120,240]
CC_INVERT     = []        # e.g. [False, False, False]

# ---- β ----
BETA_LEN      = 8
BETA_AA       = "V"
PHI_BETA, PSI_BETA = -135.0, 135.0

# β-hairpin
HAIRPIN_STRAND_LEN = 6
HAIRPIN_TURN_LEN   = 4
HAIRPIN_TURN_TYPE  = "I"   # "I" or "II"
HAIRPIN_STRAND_AA  = "V"
HAIRPIN_TURN_AA    = "D"

# β-meander (3 strands: β - turn - β - turn - β)
MEANDER_STRAND_LEN = 6
MEANDER_TURN_LEN   = 3
MEANDER_STRAND_AA  = "V"
MEANDER_TURN_AA    = "D"

# Greek key (4 strands)
GREEK_STRAND_LEN = 5
GREEK_TURN_LEN   = 3
GREEK_STRAND_AA  = "V"
GREEK_TURN_AA    = "D"

# ---- α/β /Combination ----
# HTH
HTH_H1_LEN   = 10
HTH_LOOP_LEN = 4
HTH_H2_LEN   = 10
HTH_HELIX_AA = "A"
HTH_LOOP_AA  = "D"

# HCH
HCH_H1_LEN   = 10
HCH_COIL_LEN = 6
HCH_H2_LEN   = 10
HCH_HELIX_AA = "A"
HCH_COIL_AA  = "S"

# β-α-β
BAB_B1_LEN   = 5
BAB_H_LEN    = 8
BAB_B2_LEN   = 5
BAB_BETA_AA  = "V"
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

# ---------- builders ----------

def build_alpha(phi, psi, length, aa="A"):
    pose = _pose_from_seq(aa * length)
    _set_phi_psi_omega(pose, 1, length, phi, psi, OMG)
    _relax_if_needed(pose)
    return pose

def build_coiled_coil():
    """
    先用 RosettaScripts <MakeBundle> + <Helix/> 子标签（use_degrees="true"）。
    若返回空 pose，则 fallback：用 α-helix 多股按圆周摆位。
    """
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    n = CC_N_HELICES
    lengths = CC_LENGTHS if CC_LENGTHS else [28] * n
    phases  = CC_PHASES_DEG if CC_PHASES_DEG else [i * 360.0 / n for i in range(n)]
    invert  = CC_INVERT if CC_INVERT else [False] * n
    if not (len(lengths) == len(phases) == len(invert) == n):
        raise ValueError("CC_LENGTHS / CC_PHASES_DEG / CC_INVERT must all match CC_N_HELICES")

    helix_lines = []
    for L, ph, inv in zip(lengths, phases, invert):
        inv_str = "true" if inv else "false"
        helix_lines.append(
            f'<Helix helix_length="{L}" residue_name="VAL" '
            f'r0="{CC_R0:.3f}" omega0="{CC_OMEGA0:.3f}" delta_omega0="{ph:.3f}" '
            f'z1="{CC_Z0:.3f}" invert="{inv_str}" />'
        )
    helix_block = "\n        ".join(helix_lines)

    xml = f"""
<ROSETTASCRIPTS>
  <MOVERS>
    <MakeBundle name="mb" use_degrees="true" reset="true">
        {helix_block}
    </MakeBundle>
  </MOVERS>
  <PROTOCOLS>
    <Add mover_name="mb"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
""".strip()

    objs = XmlObjects.create_from_string(xml)
    mb = objs.get_mover("mb")
    pose = rosetta.core.pose.Pose()
    mb.apply(pose)

    # Fallback：若构建失败或空 pose，手工搭一个圈
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

    _relax_if_needed(pose)
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
