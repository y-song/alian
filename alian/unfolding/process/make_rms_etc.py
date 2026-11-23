#!/usr/bin/env python3
"""
Build EEC response matrices from the per-jet parquet produced by script 1.

Notes:
  groomed version   uses (groomed jet pt,   RL, weight)
  ungroomed version uses (ungroomed jet pt, RL, weight)
  The weight is normalized by the groomed (radiator) pt in both versions, so
  groomed pt enters the ungroomed version through the weight axis. No 4D
  unfolding is needed.

Correction strategy (no Fakes are filled anywhere):
  * EEC:    purity correction of the raw distribution
            (pair_match_rec_pur_num / pair_all_rec_pur_den, or the 3D
            {v}_{lab}_reco_matched / {v}_{lab}_reco_all), then unfolding with
            Miss() handling the pair efficiency.
  * Jet pT: external purity and efficiency corrections (jet_* histograms).

Processing:
  Each event's jets are analyzed once (recluster C/A, Lund declustering, SD
  split, EEC pairs). Then:
    - every det jet  fills purity denominators/numerators and *_reco_all
    - every part jet fills efficiency denominators/numerators, *_gen_all, Misses
    - every matched det/part pair fills the responses and *_matched histograms

Run in the heppyy environment (cppyy fastjet / fjcontrib / ecorrel).
"""

import sys
import math
import os
import argparse
from collections import defaultdict

import numpy as np
import awkward as ak
import pandas as pd

import fastjet as fj
import fjcontrib
import ecorrel
import ROOT
import array

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

ROOT.gSystem.Load("libRooUnfold")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JET_R      = 0.4
SD_ZCUT    = 0.1          # soft drop zcut (beta = 0 implied by z() cut)
TRK_THRD   = 1.0          # track pt threshold used inside EEC
DPHI_CUT   = -9999        # off
DETA_CUT   = -9999        # off
POWER      = 1
NMAX       = 2
PION_MASS  = 0.13957

# Reported jet pT range. Purity diagnostics select on det pT,
# efficiency diagnostics select on part pT (each on its own level).
PT_LO, PT_HI = 20.0, 200.0

# If True, all truth pairs from truth jets that have no matched det jet
# (for a given version) are filled as Misses, so the unfolded EEC represents
# ALL truth jets. Set False if you correct the EEC jet-level efficiency
# externally instead (never do both).
MISS_FROM_UNMATCHED_TRUTH_JETS = False

# ---- response matrix binning ----
JETPT_BINS    = np.array([10, 20, 40, 60, 80, 100, 120, 150, 200, 500], dtype=np.float64)
JETPT_BINS_TR = np.array([10, 20, 40, 60, 80, 100, 120, 150, 200, 500], dtype=np.float64)

RL_NBINS = 24
RL_MIN, RL_MAX = 0.01, 1.0
RL_BINS = np.logspace(np.log10(RL_MIN), np.log10(RL_MAX), RL_NBINS + 1)

W_NBINS = 20
W_MIN, W_MAX = 1e-5, 10**-0.5
W_BINS = np.logspace(-5, -0.5, W_NBINS + 1)

# Lund plane binning (log10)
LUND_KT_BINS = np.linspace(np.log10(0.01), np.log10(1000), 40)   # log10(kt):   -2 -> 3
LUND_RD_BINS = np.linspace(np.log10(1), np.log10(1e4), 40)       # log10(R/dR):  0 -> 4

EEC_LABELS = ["full_ungroomed", "AA", "AB", "BB", "rad"]
VERSIONS = ["groomed", "ungroomed"]


# ---------------------------------------------------------------------------
# Helpers: jet building and EECs
# ---------------------------------------------------------------------------
def build_pseudojets(const_pt, const_eta, const_phi, const_label):
    """Rebuild a vectorPJ from stored constituents; label -> user_index."""
    v = fj.vectorPJ()
    for pt, eta, phi, lab in zip(const_pt, const_eta, const_phi, const_label):
        px, py = pt * math.cos(phi), pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        E = math.sqrt(px * px + py * py + pz * pz + PION_MASS * PION_MASS)
        pj = fj.PseudoJet(px, py, pz, E)
        pj.set_user_index(int(lab))
        v.push_back(pj)
    return v


def recluster_ca(constituents):
    """Recluster constituents with Cambridge/Aachen (R=1.0) into a single jet.
    Returns (jet, cs); cs must be kept alive while the jet is used."""
    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
    cs = fj.ClusterSequence(constituents, jet_def_ca)
    jets = fj.sorted_by_pt(cs.inclusive_jets(0.0))
    if len(jets) == 0:
        return None, None
    return jets[0], cs


def select_split_sd(lund_plane_elements, zcut):
    """First splitting (angular-ordered, from LundGenerator) with z > zcut."""
    for d in lund_plane_elements:
        if d.z() > zcut:
            return d
    return None


def selected_constituents(sj, trk_thrd):
    """Constituents of a subjet above the track threshold, pt-sorted."""
    out = fj.vectorPJ()
    for c in fj.sorted_by_pt(sj.constituents()):
        if c.pt() < trk_thrd:
            break
        out.append(c)
    return out


def compute_eec_pairs(c_select, scale, c_select_B=None):
    """
    Run CorrelatorBuilder and return a list of pairs:
        (RL, weight, label_i, label_j)

    Two-collection (A x B) case: the C++ CorrelatorBuilder stores every A-B
    pair twice, alternating AB (idx1->A, idx2->B) and BA (idx1->B, idx2->A).
    Each index is read from the collection it belongs to.
    """
    if c_select_B is None:
        eec = ecorrel.CorrelatorBuilder(c_select, scale, NMAX, POWER, DPHI_CUT, DETA_CUT)
    else:
        eec = ecorrel.CorrelatorBuilder(c_select, c_select_B, scale, NMAX, POWER, DPHI_CUT, DETA_CUT)

    corr = eec.correlator(2)
    rs   = corr.rs()
    ws   = corr.weights()
    idx1 = corr.indices1()
    idx2 = corr.indices2()
    n = rs.size()

    pairs = []
    if c_select_B is None:
        for k in range(n):
            i, j = int(idx1[k]), int(idx2[k])
            pairs.append((rs[k], ws[k], c_select[i].user_index(), c_select[j].user_index()))
    else:
        assert n % 2 == 0, (
            f"AxB correlator returned an odd number of entries ({n}); "
            "the AB/BA alternation assumption is violated."
        )
        for k in range(n):
            i, j = int(idx1[k]), int(idx2[k])
            if k % 2 == 0:
                lab_i, lab_j = c_select[i].user_index(), c_select_B[j].user_index()
            else:
                lab_i, lab_j = c_select_B[i].user_index(), c_select[j].user_index()
            pairs.append((rs[k], ws[k], lab_i, lab_j))
    return pairs


def match_splittings(det_B, part_B):
    """Match SD splittings via the softer branch: dR < 0.1 and
    det softer-branch pt >= 50% of part softer-branch pt."""
    if det_B.delta_R(part_B) >= 0.1:
        return False
    if det_B.pt() < 0.5 * part_B.pt():
        return False
    return True


def match_eec_pairs(det_pairs, part_pairs):
    """
    Pair-level matching by constituent labels (unordered label sets).
    Returns (matched, tr_unmatched, det_unmatched), where matched is a list of
    (det_pair, part_pair). AB/BA duplicates are consumed one at a time so the
    double counting stays consistent between levels.
    """
    part_map = {}
    for p in part_pairs:
        part_map.setdefault(frozenset((p[2], p[3])), []).append(p)

    matched, tr_unmatched, det_unmatched = [], [], []
    used = {}
    for d in det_pairs:
        key = frozenset((d[2], d[3]))
        candidates = part_map.get(key)
        if not candidates:
            det_unmatched.append(d)
            continue
        c = used.get(key, 0)
        if c < len(candidates):
            matched.append((d, candidates[c]))
            used[key] = c + 1
        else:
            det_unmatched.append(d)

    for key, candidates in part_map.items():
        tr_unmatched.extend(candidates[used.get(key, 0):])

    return matched, tr_unmatched, det_unmatched


# ---------------------------------------------------------------------------
# Helpers: per-jet analysis and small utilities
# ---------------------------------------------------------------------------
def analyze_jet(cols, r, lund_gen, zcut):
    """Recluster one jet, find its SD split, and compute all EEC pair sets.
    SD-dependent EEC labels are absent from info['eec'] if SD fails."""
    consts = build_pseudojets(cols["const_pt"][r], cols["const_eta"][r],
                              cols["const_phi"][r], cols["const_label"][r])
    jet, cs = recluster_ca(consts)
    if jet is None:
        return None

    info = {
        "level":       str(cols["level"][r]),
        "weight":      float(cols["mc_weight"][r]),
        "is_matched":  bool(cols["is_matched"][r]),
        "match_index": int(cols["match_index"][r]),
        "jet": jet, "cs": cs,                 # cs kept alive for constituents()/pieces()
        "pt": jet.perp(),                     # ungroomed jet pt
        "pt_g": None,                         # groomed (radiator) pt, None if SD fails
        "lund": lund_gen.result(jet),
        "split": None, "rad": None, "A": None, "B": None,
        "eec": {},
        "partner": None,                      # filled when linking matched jets
        "pair_match": {},                     # lab -> (matched, tr_unmatched, det_unmatched)
    }

    info["eec"]["full_ungroomed"] = compute_eec_pairs(
        selected_constituents(jet, TRK_THRD), jet.perp())

    split = select_split_sd(info["lund"], zcut)
    if split is None:
        return info
    rad = split.pair()
    subs = sorted(rad.pieces(), key=lambda x: x.pt(), reverse=True)
    if len(subs) != 2:
        return info
    A, B = subs
    scale = rad.perp()
    info.update(split=split, rad=rad, A=A, B=B, pt_g=scale)

    cA = selected_constituents(A, TRK_THRD)
    cB = selected_constituents(B, TRK_THRD)
    info["eec"]["AA"]  = compute_eec_pairs(cA, scale)
    info["eec"]["BB"]  = compute_eec_pairs(cB, scale)
    info["eec"]["AB"]  = compute_eec_pairs(cA, scale, cB)
    info["eec"]["rad"] = compute_eec_pairs(selected_constituents(rad, TRK_THRD), scale)
    return info


def axis_pt(info, v):
    """Jet pt used on the response pt axis for version v (None if unavailable)."""
    if info is None:
        return None
    return info["pt_g"] if v == "groomed" else info["pt"]


def pt_in_range(pt):
    return PT_LO <= pt < PT_HI


def pt_bin(pt):
    """Index into JETPT_BINS for the pT-differential histograms, or None."""
    b = int(np.digitize(pt, JETPT_BINS)) - 1
    return b if 0 <= b < len(JETPT_BINS) - 1 else None


def lund_coords(s):
    return math.log10(JET_R / s.Delta()), math.log10(s.kt())


def splittings_match(info):
    """True if this jet and its partner both have SD splits that match."""
    p = info["partner"]
    if p is None or info["B"] is None or p["B"] is None:
        return False
    det, part = (info, p) if info["level"] == "det" else (p, info)
    return match_splittings(det["B"], part["B"])


def version_match(d, p, v, lab):
    """
    Pair matching valid for version v and EEC label lab.
    Returns None if the jets are not matched for this version (no partner, or
    the version's pt axis is unavailable at either level, i.e. SD failed for
    the groomed version). Otherwise returns (matched, tr_unmatched, det_unmatched).
    """
    if d is None or p is None or axis_pt(d, v) is None or axis_pt(p, v) is None:
        return None
    if lab in d["pair_match"]:
        return d["pair_match"][lab]
    # jets matched for this version, but the label's pairs exist on one side only
    return [], list(p["eec"].get(lab) or []), list(d["eec"].get(lab) or [])


def integral(h):
    """Weighted integral including under/overflow (TH1 or TH2)."""
    if isinstance(h, ROOT.TH2):
        return h.Integral(0, h.GetNbinsX() + 1, 0, h.GetNbinsY() + 1)
    return h.Integral(0, h.GetNbinsX() + 1)


def ratio(a, b):
    return a / b if b > 0 else 0.0


def zcut_tag(zcut):
    """0.1 -> 'zcut0p1', 0.2 -> 'zcut0p2', 0.15 -> 'zcut0p15'."""
    return "zcut" + f"{zcut:g}".replace(".", "p")

def tagged_output(path, zcut):
    """Insert the z_cut tag before the extension: response.root -> response_zcut0p2.root."""
    base, ext = os.path.splitext(path)
    return f"{base}_{zcut_tag(zcut)}{ext or '.root'}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input parquet file (from script 1)")
    ap.add_argument("-o", "--output", default="response.root",
                    help="output ROOT file base name; the z_cut tag is appended automatically")
    ap.add_argument("--zcut", type=float, default=SD_ZCUT,
                    help="soft drop z_cut (e.g. 0.1 or 0.2)")
    ap.add_argument("--corr-rootfile", default="",
                    help="Corresponding input root file (used as a merge check on the event count)")
    args = ap.parse_args()

    if not 0.0 < args.zcut < 0.5:
        sys.exit(f"--zcut must be between 0 and 0.5, got {args.zcut}")

    output_name = tagged_output(args.output, args.zcut)
    output_diff_name = os.path.splitext(output_name)[0] + "_ptdifferential.root"
    print(f"Soft drop z_cut = {args.zcut}")
    print(f"Output files: {output_name}, {output_diff_name}")


    # -----------------------------------------------------------------------
    # Output histograms
    # -----------------------------------------------------------------------
    n_bins_pt = len(JETPT_BINS) - 1
    jetpt_edges = array.array("d", JETPT_BINS)
    n_bins_pt_tr = len(JETPT_BINS_TR) - 1
    jetpt_edges_tr = array.array("d", JETPT_BINS_TR)
    pt_ranges = list(zip(JETPT_BINS, JETPT_BINS[1:]))

    hNevents = ROOT.TH1D("hNevents", "hNevents", 2, -0.5, 1.5)

    # ---- jet pt responses ----
    h_resp_jetpt, h_res_jetpt = {}, {}
    h1_reco, h1_gen = {}, {}
    response1D, response1D_oldversion = {}, {}
    for v in VERSIONS:
        title = "groomed jet p_{T} det vs part" if v == "groomed" else "jet p_{T} det vs part"
        xaxis = "p_{T,g}^{det}" if v == "groomed" else "p_{T}^{det}"
        yaxis = "p_{T,g}^{part}" if v == "groomed" else "p_{T}^{part}"
        h_resp_jetpt[v] = ROOT.TH2D(f"resp_jetpt_{v}", f"{title};{xaxis};{yaxis}",
                                    n_bins_pt, jetpt_edges, n_bins_pt_tr, jetpt_edges_tr)
        h_res_jetpt[v] = ROOT.TH2D(
            f"res_jetpt_{v}",
            f"jetpt ((part-det)/part) {v};p_{{T}}^{{part}};(p_{{T}}^{{part}}-p_{{T}}^{{det}})/p_{{T}}^{{part}}",
            50, 0, 500, 200, -1.0, 1.0)

        label_suffix = "gr. ch jet" if v == "groomed" else "ch jet"
        h1_reco[v] = ROOT.TH1D(f"h1jetpt_reco_{v}", f"h1jetpt_reco_{v}", n_bins_pt, jetpt_edges)
        h1_reco[v].GetXaxis().SetTitle(f"p^{{det}}_{{T,{label_suffix}}}")
        h1_reco[v].GetYaxis().SetTitle("Counts")
        h1_gen[v] = ROOT.TH1D(f"h1jetpt_gen_{v}", f"h1jetpt_gen_{v}", n_bins_pt_tr, jetpt_edges_tr)
        h1_gen[v].GetXaxis().SetTitle(f"p^{{part}}_{{T,{label_suffix}}}")
        h1_gen[v].GetYaxis().SetTitle("Counts")

        response1D[v] = ROOT.RooUnfoldResponse(h1_reco[v], h1_gen[v],
                                               f"roounfold_response_1D_{v}", f"roounfold_response_1D_{v}")
        response_matrix = ROOT.TH2D(f"roounfold_response_TH2D_{v}", f"roounfold_response_TH2D_{v}",
                                    n_bins_pt, jetpt_edges, n_bins_pt_tr, jetpt_edges_tr)
        response1D_oldversion[v] = ROOT.RooUnfoldResponse(
            h1_reco[v], h1_gen[v], response_matrix,
            f"roounfold_response_1D_{v}_oldversion", f"roounfold_response_1D_{v}_oldversion")

    # ---- 6D THnSparse + 3D RooUnfold responses per EEC ----
    nbins6 = [n_bins_pt, n_bins_pt_tr, RL_NBINS, RL_NBINS, W_NBINS, W_NBINS]
    xmin6 = [JETPT_BINS[0], JETPT_BINS_TR[0], RL_MIN, RL_MIN, W_MIN, W_MIN]
    xmax6 = [JETPT_BINS[-1], JETPT_BINS_TR[-1], RL_MAX, RL_MAX, W_MAX, W_MAX]

    resp6 = {v: {} for v in VERSIONS}
    roounfold_resp6 = {v: {} for v in VERSIONS}
    # RENAMED: *_matched = matched pairs only, *_all = every pair from every jet
    reco_matched_3 = {v: {} for v in VERSIONS}
    reco_all_3 = {v: {} for v in VERSIONS}
    gen_matched_3 = {v: {} for v in VERSIONS}
    gen_all_3 = {v: {} for v in VERSIONS}

    def make_h3(name, level, label_suffix):
        if level == "det":
            h3 = ROOT.TH3D(name, name, n_bins_pt, jetpt_edges, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
        else:
            h3 = ROOT.TH3D(name, name, n_bins_pt_tr, jetpt_edges_tr, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
        h3.GetXaxis().SetTitle(f"p^{{{level}}}_{{T,{label_suffix}}}")
        h3.GetYaxis().SetTitle(f"R_{{L}}^{{{level}}}")
        h3.GetZaxis().SetTitle(f"weight^{{{level}}}")
        return h3

    for v in VERSIONS:
        label_suffix = "gr. ch jet" if v == "groomed" else "ch jet"
        for lab in EEC_LABELS:
            hs = ROOT.THnSparseD(
                f"resp6_{v}_{lab}",
                f"6D response {v} {lab};p_{{T}}^{{det}};p_{{T}}^{{part}};R_{{L}}^{{det}};R_{{L}}^{{part}};w^{{det}};w^{{part}}",
                6, array.array("i", nbins6), array.array("d", xmin6), array.array("d", xmax6))
            hs.GetAxis(0).Set(n_bins_pt, jetpt_edges)
            hs.GetAxis(1).Set(n_bins_pt_tr, jetpt_edges_tr)
            hs.GetAxis(2).Set(RL_NBINS, array.array("d", RL_BINS.tolist()))
            hs.GetAxis(3).Set(RL_NBINS, array.array("d", RL_BINS.tolist()))
            hs.GetAxis(4).Set(W_NBINS, array.array("d", W_BINS.tolist()))
            hs.GetAxis(5).Set(W_NBINS, array.array("d", W_BINS.tolist()))
            hs.Sumw2()
            resp6[v][lab] = hs

            reco_matched_3[v][lab] = make_h3(f"{v}_{lab}_reco_matched", "det", label_suffix)
            reco_all_3[v][lab]     = make_h3(f"{v}_{lab}_reco_all", "det", label_suffix)
            gen_matched_3[v][lab]  = make_h3(f"{v}_{lab}_gen_matched", "part", label_suffix)
            gen_all_3[v][lab]      = make_h3(f"{v}_{lab}_gen_all", "part", label_suffix)

            roounfold_resp6[v][lab] = ROOT.RooUnfoldResponse(
                reco_matched_3[v][lab], gen_matched_3[v][lab],
                f"{v}_{lab}_roounfold_response", f"{v}_{lab}_roounfold_response")

    # ---- jet-level efficiency / purity ----
    h_match_gen_jet_eff_num = {v: ROOT.TH1D(f"jet_match_gen_eff_num_{v}", f"Matched Gen, {v.capitalize()} Jet Eff Num", n_bins_pt_tr, jetpt_edges_tr) for v in VERSIONS}
    h_all_gen_jet_eff_den   = {v: ROOT.TH1D(f"jet_all_gen_eff_den_{v}",   f"Total Gen, {v.capitalize()} Jet Eff Den",   n_bins_pt_tr, jetpt_edges_tr) for v in VERSIONS}
    h_match_rec_jet_pur_num = {v: ROOT.TH1D(f"jet_match_rec_pur_num_{v}", f"Matched Rec, {v.capitalize()} Jet Pur Num", n_bins_pt, jetpt_edges) for v in VERSIONS}
    h_all_rec_jet_pur_den   = {v: ROOT.TH1D(f"jet_all_rec_pur_den_{v}",   f"Total Rec, {v.capitalize()} Jet Pur Den",   n_bins_pt, jetpt_edges) for v in VERSIONS}

    # ---- SD splitting efficiency / purity (log10 Lund coordinates) ----
    lund_axes = ";log_{10}(R/#DeltaR);log_{10}(k_{T})"
    def make_lund(name, title):
        return ROOT.TH2D(name, title + lund_axes,
                         len(LUND_RD_BINS) - 1, LUND_RD_BINS, len(LUND_KT_BINS) - 1, LUND_KT_BINS)

    h_lund_matched_gen = make_lund("lund_matched_gen", "SD splitting, matched gen")
    h_lund_matched_rec = make_lund("lund_matched_rec", "SD splitting, matched rec")
    h_lund_all_gen     = make_lund("lund_all_gen", "SD splitting, all gen")
    h_lund_all_rec     = make_lund("lund_all_rec", "SD splitting, all rec")

    # ---- pT-differential full Lund planes (all declusterings) ----
    # "matched" here means the JET is matched; the splittings themselves are not matched.
    h_lund_all_gen_pt     = {a: make_lund(f"h_lund_all_gen_pt{b}-{c}",     f"Lund all gen pt:{b}-{c}")         for a, (b, c) in enumerate(pt_ranges)}
    h_lund_all_rec_pt     = {a: make_lund(f"h_lund_all_rec_pt{b}-{c}",     f"Lund all rec pt:{b}-{c}")         for a, (b, c) in enumerate(pt_ranges)}
    h_lund_matched_gen_pt = {a: make_lund(f"h_lund_matched_gen_pt{b}-{c}", f"Lund jet-matched gen pt:{b}-{c}") for a, (b, c) in enumerate(pt_ranges)}
    h_lund_matched_rec_pt = {a: make_lund(f"h_lund_matched_rec_pt{b}-{c}", f"Lund jet-matched rec pt:{b}-{c}") for a, (b, c) in enumerate(pt_ranges)}

    # ---- track efficiency / purity / residuals (within matched jets) ----
    h_trk_eff_num = ROOT.TH1D("trk_eff_num", "Track Efficiency Num;p_{T}", 250, 0, 250)
    h_trk_eff_den = ROOT.TH1D("trk_eff_den", "Track Efficiency Den;p_{T}", 250, 0, 250)
    h_trk_pur_num = ROOT.TH1D("trk_pur_num", "Track Purity Num;p_{T}", 250, 0, 250)
    h_trk_pur_den = ROOT.TH1D("trk_pur_den", "Track Purity Den;p_{T}", 250, 0, 250)
    h_trk_res_pt  = ROOT.TH2D("trk_res_pt", "Track pT Residual;p_{T}^{part};(p_{T}^{part}-p_{T}^{det})/p_{T}^{part}", 250, 0, 250, 100, -1, 1)

    # ---- pair efficiency / purity / residuals ----
    h_match_gen_pair_eff_num, h_all_gen_pair_eff_den = {}, {}
    h_match_rec_pair_pur_num, h_all_rec_pair_pur_den = {}, {}
    h_res_rl, h_res_w = {}, {}
    for lab in EEC_LABELS:
        h_match_gen_pair_eff_num[lab] = ROOT.TH1D(f"pair_match_gen_eff_num_{lab}", f"Matched Gen, Pair Eff Num {lab}", RL_NBINS, RL_BINS)
        h_all_gen_pair_eff_den[lab]   = ROOT.TH1D(f"pair_all_gen_eff_den_{lab}",   f"Total Gen, Pair Eff Den {lab}",   RL_NBINS, RL_BINS)
        h_match_rec_pair_pur_num[lab] = ROOT.TH1D(f"pair_match_rec_pur_num_{lab}", f"Matched Rec, Pair Pur Num {lab}", RL_NBINS, RL_BINS)
        h_all_rec_pair_pur_den[lab]   = ROOT.TH1D(f"pair_all_rec_pur_den_{lab}",   f"Total Rec, Pair Pur Den {lab}",   RL_NBINS, RL_BINS)
        h_res_rl[lab] = ROOT.TH2D(f"res_rl_{lab}", f"RL residual {lab};R_{{L}}^{{part}};(R_{{L}}^{{part}}-R_{{L}}^{{det}})/R_{{L}}^{{part}}", RL_NBINS, RL_BINS, 100, -1, 1)
        h_res_w[lab]  = ROOT.TH2D(f"res_w_{lab}",  f"Weight residual {lab};w^{{part}};(w^{{part}}-w^{{det}})/w^{{part}}", W_NBINS, W_BINS, 100, -1, 1)

    def make_pair_pt(prefix, title):
        return {a: {lab: ROOT.TH1D(f"{prefix}_pt{b}-{c}_{lab}", f"{title} pt:{b}-{c} {lab}", RL_NBINS, RL_BINS)
                    for lab in EEC_LABELS} for a, (b, c) in enumerate(pt_ranges)}

    h_match_gen_pair_eff_num_pt = make_pair_pt("pair_match_gen_eff_num", "Matched Gen, Pair Eff Num")
    h_all_gen_pair_eff_den_pt   = make_pair_pt("pair_all_gen_eff_den",   "Total Gen, Pair Eff Den")
    h_match_rec_pair_pur_num_pt = make_pair_pt("pair_match_rec_pur_num", "Matched Rec, Pair Pur Num")
    h_all_rec_pair_pur_den_pt   = make_pair_pt("pair_all_rec_pur_den",   "Total Rec, Pair Pur Den")

    # -----------------------------------------------------------------------
    # Fill functions
    # -----------------------------------------------------------------------
    def fill_jet_level(info, h_den, h_num):
        """Jet efficiency (part) or purity (det).
        Ungroomed: matched = jet-matched.
        Groomed:   matched = jet-matched AND SD passes at both levels."""
        w, p = info["weight"], info["partner"]
        h_den["ungroomed"].Fill(info["pt"], w)
        if p is not None:
            h_num["ungroomed"].Fill(info["pt"], w)
        if info["pt_g"] is not None:
            h_den["groomed"].Fill(info["pt_g"], w)
            if p is not None and p["pt_g"] is not None:
                h_num["groomed"].Fill(info["pt_g"], w)

    def fill_lund_ptdiff(info, h_all, h_jetmatched):
        b = pt_bin(info["pt"])
        if b is None:
            return
        w = info["weight"]
        for s in info["lund"]:
            x, y = lund_coords(s)
            h_all[b].Fill(x, y, w)
            if info["partner"] is not None:
                h_jetmatched[b].Fill(x, y, w)

    def fill_sd_splitting(info, h_all, h_matched):
        """Denominator: all jets in range (own level) with an SD split.
        Numerator: those whose partner also has an SD split that matches."""
        if info["split"] is None or not pt_in_range(info["pt"]):
            return
        x, y = lund_coords(info["split"])
        h_all.Fill(x, y, info["weight"])
        if splittings_match(info):
            h_matched.Fill(x, y, info["weight"])

    def fill_pair_1d(info, h_all, h_match, h_all_pt, h_match_pt):
        """Pair purity (det jets, selected on det pt) or efficiency (part jets,
        selected on part pt). The denominator includes EVERY jet of that level
        in range: matched or not, partner passing SD or not."""
        if not pt_in_range(info["pt"]):
            return
        b = pt_bin(info["pt"])
        w = info["weight"]
        side = 0 if info["level"] == "det" else 1
        for lab in EEC_LABELS:
            pairs = info["eec"].get(lab)
            if pairs is None:
                continue
            matched = info["pair_match"][lab][0] if lab in info["pair_match"] else []
            for pr in pairs:
                h_all[lab].Fill(pr[0], w)
                if b is not None:
                    h_all_pt[b][lab].Fill(pr[0], w)
            for m in matched:
                rl = m[side][0]
                h_match[lab].Fill(rl, w)
                if b is not None:
                    h_match_pt[b][lab].Fill(rl, w)

    def fill_all_3d(info, h3):
        """Every pair from every jet, in the response binning (3D purity/efficiency denominators)."""
        w = info["weight"]
        for v in VERSIONS:
            apt = axis_pt(info, v)
            if apt is None:
                continue
            for lab in EEC_LABELS:
                pairs = info["eec"].get(lab)
                if pairs is None:
                    continue
                for pr in pairs:
                    h3[v][lab].Fill(apt, pr[0], pr[1], w)

    def fill_misses(p):
        """Truth pairs without a det partner -> RooUnfold Miss."""
        w, d = p["weight"], p["partner"]
        for v in VERSIONS:
            ppt = axis_pt(p, v)
            if ppt is None:
                continue
            for lab in EEC_LABELS:
                ppairs = p["eec"].get(lab)
                if ppairs is None:
                    continue
                vm = version_match(d, p, v, lab)
                if vm is not None:
                    missed = vm[1]                      # pair-level inefficiency in matched jets
                elif MISS_FROM_UNMATCHED_TRUTH_JETS:
                    missed = ppairs                     # whole truth jet unmatched for this version
                else:
                    continue
                for pp in missed:
                    roounfold_resp6[v][lab].Miss(ppt, pp[0], pp[1], w)

    def fill_det(d):
        fill_jet_level(d, h_all_rec_jet_pur_den, h_match_rec_jet_pur_num)
        fill_lund_ptdiff(d, h_lund_all_rec_pt, h_lund_matched_rec_pt)
        fill_sd_splitting(d, h_lund_all_rec, h_lund_matched_rec)
        fill_pair_1d(d, h_all_rec_pair_pur_den, h_match_rec_pair_pur_num,
                     h_all_rec_pair_pur_den_pt, h_match_rec_pair_pur_num_pt)
        fill_all_3d(d, reco_all_3)

    def fill_part(p):
        fill_jet_level(p, h_all_gen_jet_eff_den, h_match_gen_jet_eff_num)
        fill_lund_ptdiff(p, h_lund_all_gen_pt, h_lund_matched_gen_pt)
        fill_sd_splitting(p, h_lund_all_gen, h_lund_matched_gen)
        fill_pair_1d(p, h_all_gen_pair_eff_den, h_match_gen_pair_eff_num,
                     h_all_gen_pair_eff_den_pt, h_match_gen_pair_eff_num_pt)
        fill_all_3d(p, gen_all_3)
        fill_misses(p)

    def fill_matched(d, p):
        """Responses and matched histograms for one matched det/part jet pair."""
        w = d["weight"]
        det_in, part_in = pt_in_range(d["pt"]), pt_in_range(p["pt"])

        # ---- track efficiency/purity/residuals via MC labels ----
        part_by_label = {c.user_index(): c for c in p["jet"].constituents()}
        det_labels = {c.user_index() for c in d["jet"].constituents()}
        if det_in:
            for dtrk in d["jet"].constituents():
                h_trk_pur_den.Fill(dtrk.perp(), w)
                ptrk = part_by_label.get(dtrk.user_index())
                if ptrk is not None:
                    h_trk_pur_num.Fill(dtrk.perp(), w)
                    h_trk_res_pt.Fill(ptrk.perp(), (ptrk.perp() - dtrk.perp()) / ptrk.perp(), w)
        if part_in:
            for ptrk in p["jet"].constituents():
                h_trk_eff_den.Fill(ptrk.perp(), w)
                if ptrk.user_index() in det_labels:
                    h_trk_eff_num.Fill(ptrk.perp(), w)

        # ---- pair residuals (once per matched pair, not once per version) ----
        if part_in:
            for lab, (matched, _, _) in d["pair_match"].items():
                for dp, pp in matched:
                    if pp[0] > 0:
                        h_res_rl[lab].Fill(pp[0], (pp[0] - dp[0]) / pp[0], w)
                    if pp[1] > 0:
                        h_res_w[lab].Fill(pp[1], (pp[1] - dp[1]) / pp[1], w)

        # ---- responses ----
        for v in VERSIONS:
            dpt, ppt = axis_pt(d, v), axis_pt(p, v)
            if dpt is None or ppt is None:
                continue        # groomed version requires SD at both levels

            h_resp_jetpt[v].Fill(dpt, ppt, w)
            h1_reco[v].Fill(dpt, w)
            h1_gen[v].Fill(ppt, w)
            response1D[v].Fill(dpt, ppt, w)
            response1D_oldversion[v].Fill(dpt, ppt, w)
            if ppt > 0:
                h_res_jetpt[v].Fill(ppt, (ppt - dpt) / ppt, w)

            for lab in EEC_LABELS:
                matched = version_match(d, p, v, lab)[0]
                for dp, pp in matched:
                    resp6[v][lab].Fill(array.array("d", [dpt, ppt, dp[0], pp[0], dp[1], pp[1]]), w)
                    roounfold_resp6[v][lab].Fill(dpt, dp[0], dp[1], ppt, pp[0], pp[1], w)
                    reco_matched_3[v][lab].Fill(dpt, dp[0], dp[1], w)
                    gen_matched_3[v][lab].Fill(ppt, pp[0], pp[1], w)

    # -----------------------------------------------------------------------
    # Event count (merge check only)
    # -----------------------------------------------------------------------
    total_num_events = 0
    if args.corr_rootfile != "":
        corr_file = ROOT.TFile.Open(args.corr_rootfile, "READ")
        total_num_events = corr_file.Get("eventTree").GetEntries()
        corr_file.Close()
    hNevents.SetBinContent(1, total_num_events)
    hNevents.SetBinError(1, total_num_events ** 0.5)
    hNevents.SetEntries(total_num_events)

    # -----------------------------------------------------------------------
    # Read parquet once into plain Python lists (fast row access)
    # -----------------------------------------------------------------------
    jets = ak.from_parquet(args.input)
    if len(jets) == 0:
        print(f"Warning: File '{args.input}' contains 0 jet entries. Exiting process.")
        return

    fields = ["event", "level", "jet_index", "is_matched", "match_index", "mc_weight",
              "const_pt", "const_eta", "const_phi", "const_label"]
    cols = {f: ak.to_list(jets[f]) for f in fields}

    rows_by_event = defaultdict(list)
    for r, ev in enumerate(cols["event"]):
        rows_by_event[ev].append(r)

    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
    lund_gen = fjcontrib.LundGenerator(jet_def_ca)

    # -----------------------------------------------------------------------
    # Event loop
    # -----------------------------------------------------------------------
    n_pairs = 0
    for ev, rlist in rows_by_event.items():
        # 1) analyze every jet in the event once
        infos = {}
        for r in rlist:
            info = analyze_jet(cols, r, lund_gen, args.zcut)
            if info is not None:
                infos[(info["level"], int(cols["jet_index"][r]))] = info

        # 2) link matched det/part jets and do pair-level matching once per pair
        for (level, _), d in infos.items():
            if level != "det" or not d["is_matched"]:
                continue
            p = infos.get(("part", d["match_index"]))
            if p is None:
                continue
            d["partner"], p["partner"] = p, d
            for lab in EEC_LABELS:
                dpairs, ppairs = d["eec"].get(lab), p["eec"].get(lab)
                if dpairs is not None and ppairs is not None:
                    d["pair_match"][lab] = match_eec_pairs(dpairs, ppairs)
            p["pair_match"] = d["pair_match"]

        # 3) fill
        for (level, _), info in infos.items():
            if level == "det":
                fill_det(info)
                if info["partner"] is not None:
                    fill_matched(info, info["partner"])
                    n_pairs += 1
            else:
                fill_part(info)

    print(f"Filled responses from {n_pairs} matched jet pairs.")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    fout = ROOT.TFile(output_name, "RECREATE")
    fout.cd()
    hNevents.Write()
    ROOT.TParameter("double")("sd_zcut", args.zcut).Write()

    for v in VERSIONS:
        h_resp_jetpt[v].Write()
        h_res_jetpt[v].Write()
        response1D[v].Write()
        response1D_oldversion[v].Write()
        for lab in EEC_LABELS:
            resp6[v][lab].Write()
            roounfold_resp6[v][lab].Write()
            reco_matched_3[v][lab].Write()
            reco_all_3[v][lab].Write()
            gen_matched_3[v][lab].Write()
            gen_all_3[v][lab].Write()

    for v in VERSIONS:
        h_match_gen_jet_eff_num[v].Write()
        h_all_gen_jet_eff_den[v].Write()
        h_match_rec_jet_pur_num[v].Write()
        h_all_rec_jet_pur_den[v].Write()

    h_lund_matched_gen.Write()
    h_lund_matched_rec.Write()
    h_lund_all_gen.Write()
    h_lund_all_rec.Write()

    h_trk_eff_num.Write()
    h_trk_eff_den.Write()
    h_trk_pur_num.Write()
    h_trk_pur_den.Write()
    h_trk_res_pt.Write()

    for lab in EEC_LABELS:
        h_match_gen_pair_eff_num[lab].Write()
        h_all_gen_pair_eff_den[lab].Write()
        h_match_rec_pair_pur_num[lab].Write()
        h_all_rec_pair_pur_den[lab].Write()
        h_res_rl[lab].Write()
        h_res_w[lab].Write()

    # ---- pT-differential output ----
    fout_diff = ROOT.TFile(output_diff_name, "RECREATE")
    fout_diff.cd()
    hNevents.Write()
    ROOT.TParameter("double")("sd_zcut", args.zcut).Write()
    for b in range(n_bins_pt):
        h_lund_all_gen_pt[b].Write()
        h_lund_all_rec_pt[b].Write()
        h_lund_matched_gen_pt[b].Write()
        h_lund_matched_rec_pt[b].Write()
        for lab in EEC_LABELS:
            h_match_gen_pair_eff_num_pt[b][lab].Write()
            h_all_gen_pair_eff_den_pt[b][lab].Write()
            h_match_rec_pair_pur_num_pt[b][lab].Write()
            h_all_rec_pair_pur_den_pt[b][lab].Write()
    fout_diff.Close()

    # ---- weighted scalar summary ----
    fout.cd()
    jet_eff = ratio(integral(h_match_gen_jet_eff_num["ungroomed"]), integral(h_all_gen_jet_eff_den["ungroomed"]))
    jet_pur = ratio(integral(h_match_rec_jet_pur_num["ungroomed"]), integral(h_all_rec_jet_pur_den["ungroomed"]))
    split_eff = ratio(integral(h_lund_matched_gen), integral(h_lund_all_gen))
    split_pur = ratio(integral(h_lund_matched_rec), integral(h_lund_all_rec))

    h_summary = ROOT.TH1D("summary_efficiencies", "Global Efficiency and Purity (weighted);Metric;Value", 20, 0, 20)
    summary = [("Jet Eff", jet_eff), ("Jet Pur", jet_pur),
               ("Split Eff", split_eff), ("Split Pur", split_pur)]
    for lab in EEC_LABELS:
        summary.append((f"Pair {lab} Eff", ratio(integral(h_match_gen_pair_eff_num[lab]), integral(h_all_gen_pair_eff_den[lab]))))
        summary.append((f"Pair {lab} Pur", ratio(integral(h_match_rec_pair_pur_num[lab]), integral(h_all_rec_pair_pur_den[lab]))))
    for i, (name, val) in enumerate(summary, start=1):
        h_summary.SetBinContent(i, val)
        h_summary.GetXaxis().SetBinLabel(i, name)
    h_summary.Write()

    fout.Close()
    print(f"Wrote response matrices and metrics to {output_name} and {output_diff_name}")
    for name, val in summary:
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()