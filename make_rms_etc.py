'''
Add: single  particle efFICIENCY
Add: pt differential
NOTE: if we wanted to look in bins of ungroomed jet pt, but calculate weight with groomed jet pt, would 4D unfolding be needed??
In this script:
groomed version uses (groomed jet pt, RL, radiator for weight)
ungroomed version uses (ungroomed jet pt, RL, radiator for weight) (so groomed jet pt is included inherently in the weight)
'''

#!/usr/bin/env python3
"""
Build EEC response matrices from the per-jet parquet produced by script 1.

For each matched det/part jet pair:
  - recluster constituents (C/A), run LundGenerator, find first SD zcut split
  - get radiator + subjet A, subjet B
  - compute EECs (AA, BB, AxB, radiator) with radiator.perp() weighting
  - pair-level match det EEC pairs to part EEC pairs via constituent labels
  - fill:
      * 2D  jet pt (det vs part)         -- one per event pair
      * 2D  groomed(radiator) pt (det vs part)
      * 6D  (pt_det, pt_part, RL_det, RL_part, w_det, w_part) for each of
            AA, AB, BB, radiator

Run in the heppyy environment (cppyy fastjet / fjcontrib / ecorrel).

    pip / env deps: uproot, awkward, numpy, ROOT, heppyy
"""

import sys
import math
import argparse
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
TRK_THRD   = 1.0          # track pt threshold used inside EEC (self.trk_thrd)
DPHI_CUT   = -9999        # off
DETA_CUT   = -9999        # off
POWER      = 1
NMAX       = 2
PION_MASS  = 0.13957

# ---- response matrix binning ----
JETPT_BINS = np.array([10, 20, 40, 60, 80, 100, 120, 150, 200, 500], dtype=np.float64)

RL_NBINS = 25
RL_MIN, RL_MAX = 0.01, 1.0
RL_BINS = np.logspace(np.log10(RL_MIN), np.log10(RL_MAX), RL_NBINS + 1)

W_NBINS = 20 #30 #100
W_MIN, W_MAX = 0.0, 0.3 #1.0 #0.3
W_BINS = np.logspace(-5,-0.5,W_NBINS+1) #0.00001 - 0.316227766 # W_BINS = np.linspace(W_MIN, W_MAX, W_NBINS + 1)

# Lund Plane binning
LUND_KT_BINS = np.linspace(np.log10(0.01), np.log10(1000), 40) # log10(kt) (linearly: -2 --> 3)
LUND_RD_BINS = np.linspace(np.log10(1), np.log10(1e4), 40)    # log10(R/dR) (linearly: 0 --> 4)

EEC_LABELS = ["full_ungroomed", "AA", "AB", "BB", "rad"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_pseudojets(const_pt, const_eta, const_phi, const_label):
    """Rebuild a vectorPJ from stored constituents; label -> user_index."""
    v = fj.vectorPJ()
    for pt, eta, phi, lab in zip(const_pt, const_eta, const_phi, const_label):
        pj = fj.PseudoJet()
        pj.reset_PtYPhiM(float(pt), float(eta), float(phi), PION_MASS)
        pj.set_user_index(int(lab))
        v.push_back(pj)
    return v


def recluster_ca(constituents):
    """Recluster constituents with Cambridge/Aachen into a single jet."""
    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0) #5.0)  # large R to catch all
    cs = fj.ClusterSequence(constituents, jet_def_ca)
    jets = fj.sorted_by_pt(cs.inclusive_jets(0.0))
    if len(jets) == 0:
        return None, None
    # keep the ClusterSequence alive by returning it too
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
    where label_i/label_j are the user_index (particle labels) of the two
    constituents forming the pair.

    Two-collection (A x B) case:
        The C++ CorrelatorBuilder stores every A-B pair TWICE, in strict
        alternating order:
            even k -> AB entry:  indices1 = A-index, indices2 = B-index
            odd  k -> BA entry:  indices1 = B-index, indices2 = A-index
        (see the two consecutive addwr(...) calls in the constructor.)
        Both orderings are kept intentionally, but each index must be read
        from the collection it actually belongs to -- otherwise indices from
        the BA entries overflow the wrong vector (IndexError) or mislabel.

        The dphi/deta cuts reject pairs BEFORE both addwr calls, so rejection
        always removes AB and BA together and never breaks the alternation.
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

    if c_select_B is None: #AA/BB/groomed jet
        # single collection: indices1/indices2 both index into c_select
        for k in range(n):
            i = int(idx1[k])
            j = int(idx2[k])
            lab_i = c_select[i].user_index()
            lab_j = c_select[j].user_index()
            pairs.append((rs[k], ws[k], lab_i, lab_j))
    else: #AB
        # two collections: entries alternate AB, BA, AB, BA, ...
        # so the total count must be even.
        assert n % 2 == 0, (
            f"AxB correlator returned an odd number of entries ({n}); "
            "the AB/BA alternation assumption is violated."
        )
        for k in range(n):
            i = int(idx1[k])
            j = int(idx2[k])
            if k % 2 == 0:
                # AB entry: idx1 -> A (c_select), idx2 -> B (c_select_B)
                lab_i = c_select[i].user_index()
                lab_j = c_select_B[j].user_index()
            else:
                # BA entry: idx1 -> B (c_select_B), idx2 -> A (c_select)
                lab_i = c_select_B[i].user_index()
                lab_j = c_select[j].user_index()
            pairs.append((rs[k], ws[k], lab_i, lab_j))

    return pairs


def match_splittings(det_B, part_B):
    """
    Match two splittings based on deltaR of the softer branch and momentum fraction of the softer branch.
    Returns True if matched, False otherwise.
    Input is the softer branch of the splitting for the detector and particle level splittings.
    NOTE: b/c of this analysis setup, there is no check for bidirectional/uniqueness of the matching
    """
    # 1. Check if deltaR < 0.1
    # Match the softer branch (det_B) to the corresponding particle branch (part_B)
    if det_B.delta_R(part_B) >= 0.1:
        return False

    # 2. Softer branch of detector-level splitting carries at least 50% of the
    # momentum of the corresponding particle-level branch
    if det_B.pt() < 0.5 * part_B.pt():
        return False

    return True


def match_eec_pairs(det_pairs, part_pairs):

    """
    Pair-level matching by constituent labels.
    A det pair (li, lj) matches a part pair (li, lj) if the *unordered*
    label sets are identical (both particles are the same at both levels).
    Returns list of (det_pair, part_pair).

    Note: because both AB and BA orderings are stored, each physical pair
    appears twice at each level. The used-counter below consumes candidates
    one at a time, so the double-counting stays consistent between det and
    part (each of the two det entries matches one of the two part entries).

    det_pairs/part_pairs are lists of tuples: (RL, weight, label_i, label_j)
    """
    # index particle pairs by frozenset of the two labels
    part_map = {}
    # print("IN FUNCTION MATCH EEC PAIRS")
    for p in part_pairs:
        key = frozenset((p[2], p[3]))
        # a jet can have the pair (i,j) and (j,i) counted twice;
        # store a list so both orderings can be matched
        part_map.setdefault(key, []).append(p)

    matched = []
    tr_unmatched = []
    det_unmatched = []
    used = {}  # key -> count already consumed
    for d in det_pairs:
        key = frozenset((d[2], d[3]))
        candidates = part_map.get(key) #returns the pair (RL, weight, label_i, label_j) or None
        if not candidates:
            det_unmatched.append(d)
            continue
        c = used.get(key, 0) #Retrieves how many candidates under this key have already been assigned. Defaults to 0 if it's the first time seeing this key.
        if c < len(candidates):
            matched.append((d, candidates[c]))
            used[key] = c + 1
        else:
            det_unmatched.append(d)

    # Identify truth pairs that were never matched
    for key, candidates in part_map.items():
        consumed = used.get(key, 0)
        for i in range(consumed, len(candidates)):
            tr_unmatched.append(candidates[i])

    return matched, tr_unmatched, det_unmatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input parquet file (from script 1)")
    ap.add_argument("-o", "--output", default="response.root", help="output ROOT file")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Output histograms
    # -----------------------------------------------------------------------
    VERSIONS = ["groomed", "ungroomed"]
    n_bins_pt = len(JETPT_BINS) - 1
    jetpt_edges = array.array("d", JETPT_BINS)


    h_resp_jetpt = {}
    h_res_jetpt = {}
    for v in VERSIONS:
        title = "groomed jet p_{T} det vs part" if v == "groomed" else "jet p_{T} det vs part"
        xaxis = "p_{T,g}^{det}" if v == "groomed" else "p_{T}^{det}"
        yaxis = "p_{T,g}^{part}" if v == "groomed" else "p_{T}^{part}"
        h_resp_jetpt[v] = ROOT.TH2D(f"resp_jetpt_{v}", f"{title};{xaxis};{yaxis}",
                                     len(JETPT_BINS) - 1, jetpt_edges,
                                     len(JETPT_BINS) - 1, jetpt_edges)

        # JES/JER: pT_part vs (pT_part - pT_det / pT_part)
        h_res_jetpt[v] = ROOT.TH2D(f"res_jetpt_{v}", f"jetpt ((part-det)/part) {v};p_{{T}}^{{part}};(p_{{T}}^{{part}}-p_{{T}}^{{det}})/p_{{T}}^{{part}}",
                                      50, 0, 500, 200, -1.0, 1.0)

    # 1D response matrix
    h1_reco = {}
    h1_gen = {}
    response1D = {}
    for v in VERSIONS:
        label_suffix = "gr. ch jet" if v == "groomed" else "ch jet"
        h1_reco[v] = ROOT.TH1D(f"h1jetpt_reco_{v}", f"h1jetpt_reco_{v}", len(JETPT_BINS) - 1, jetpt_edges)
        h1_reco[v].GetXaxis().SetTitle(f'p^{{det}}_{{T,{label_suffix}}}')
        h1_reco[v].GetYaxis().SetTitle('Counts')

        h1_gen[v] = ROOT.TH1D(f"h1jetpt_gen_{v}", f"h1jetpt_gen_{v}", len(JETPT_BINS) - 1, jetpt_edges)
        h1_gen[v].GetXaxis().SetTitle(f'p^{{part}}_{{T,{label_suffix}}}')
        h1_gen[v].GetYaxis().SetTitle('Counts')

        res = ROOT.RooUnfoldResponse(h1_reco[v], h1_gen[v])
        res.SetName(f"roounfold_response_1D_{v}")
        response1D[v] = res

    # 6D THnSparse per EEC:  axes = pt_det, pt_part, RL_det, RL_part, w_det, w_part
    nbins6 = np.array([len(JETPT_BINS) - 1, len(JETPT_BINS) - 1,
                       RL_NBINS, RL_NBINS, W_NBINS, W_NBINS], dtype=np.int32)
    xmin6 = np.array([JETPT_BINS[0], JETPT_BINS[0], RL_MIN, RL_MIN, W_MIN, W_MIN], dtype=np.float64)
    xmax6 = np.array([JETPT_BINS[-1], JETPT_BINS[-1], RL_MAX, RL_MAX, W_MAX, W_MAX], dtype=np.float64)

    resp6 = {v: {} for v in VERSIONS}
    roounfold_resp6 = {v: {} for v in VERSIONS}
    reco_3 = {v: {} for v in VERSIONS}
    reco_unmatched_3 = {v: {} for v in VERSIONS}
    gen_3 = {v: {} for v in VERSIONS}
    gen_unmatched_3 = {v: {} for v in VERSIONS}

    for v in VERSIONS:
        label_suffix = "gr. ch jet" if v == "groomed" else "ch jet"
        for lab in EEC_LABELS:
            hs = ROOT.THnSparseD(
                f"resp6_{v}_{lab}",
                f"6D response {v} {lab};p_{{T}}^{{det}};p_{{T}}^{{part}};R_{{L}}^{{det}};R_{{L}}^{{part}};w^{{det}};w^{{part}}",
                6,
                array.array("i", nbins6.tolist()),
                array.array("d", xmin6.tolist()),
                array.array("d", xmax6.tolist()),
            )
            # set variable bin edges on the axes that need them
            hs.GetAxis(0).Set(len(JETPT_BINS) - 1, jetpt_edges)  # pt_det
            hs.GetAxis(1).Set(len(JETPT_BINS) - 1, jetpt_edges)  # pt_part
            hs.GetAxis(2).Set(RL_NBINS, array.array("d", RL_BINS.tolist()))  # RL_det (log)
            hs.GetAxis(3).Set(RL_NBINS, array.array("d", RL_BINS.tolist()))  # RL_part (log)
            hs.GetAxis(4).Set(W_NBINS, array.array("d", W_BINS.tolist()))    # w_det
            hs.GetAxis(5).Set(W_NBINS, array.array("d", W_BINS.tolist()))    # w_part
            hs.Sumw2()
            resp6[v][lab] = hs

            # same response matrices, but in ROOT.RooUnfoldResponse form, order: jetpt, RL, weight (det), jetpt, RL, weight (part)
            h3_reco = ROOT.TH3D(f"{v}_{lab}_reco", f"{v}_{lab}_reco", len(JETPT_BINS) - 1, jetpt_edges, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
            h3_gen = ROOT.TH3D(f"{v}_{lab}_gen", f"{v}_{lab}_gen", len(JETPT_BINS) - 1, jetpt_edges, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
            h3_reco.GetXaxis().SetTitle(f'p^{{det}}_{{T,{label_suffix}}}'); h3_reco.GetYaxis().SetTitle('R_{L}^{det}'); h3_reco.GetZaxis().SetTitle('weight^{det}')
            h3_gen.GetXaxis().SetTitle(f'p^{{part}}_{{T,{label_suffix}}}'); h3_gen.GetYaxis().SetTitle('R_{L}^{part}'); h3_gen.GetZaxis().SetTitle('weight^{part}')
            hresponse = ROOT.RooUnfoldResponse(h3_reco, h3_gen, f"{v}_{lab}_roounfold_response", f"{v}_{lab}_roounfold_response")
            roounfold_resp6[v][lab] = hresponse

            h3_reco_unmatched = ROOT.TH3D(f"{v}_{lab}_reco_unmatched", f"{v}_{lab}_reco_unmatched", len(JETPT_BINS) - 1, jetpt_edges, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
            h3_gen_unmatched = ROOT.TH3D(f"{v}_{lab}_gen_unmatched", f"{v}_{lab}_gen_unmatched", len(JETPT_BINS) - 1, jetpt_edges, RL_NBINS, RL_BINS, W_NBINS, W_BINS)
            h3_reco_unmatched.GetXaxis().SetTitle(f'p^{{det}}_{{T,{label_suffix}}}'); h3_reco_unmatched.GetYaxis().SetTitle('R_{L}^{det}'); h3_reco_unmatched.GetZaxis().SetTitle('weight^{det}')
            h3_gen_unmatched.GetXaxis().SetTitle(f'p^{{part}}_{{T,{label_suffix}}}'); h3_gen_unmatched.GetYaxis().SetTitle('R_{L}^{part}'); h3_gen_unmatched.GetZaxis().SetTitle('weight^{part}')

            reco_3[v][lab] = h3_reco
            reco_unmatched_3[v][lab] = h3_reco_unmatched
            gen_3[v][lab] = h3_gen
            gen_unmatched_3[v][lab] = h3_gen_unmatched

    # --- pT-differential versions ---
    n_bins_pt = len(JETPT_BINS) - 1

    # --- Differential Efficiency/Purity Histograms ---
    # Jet level (pT)
    h_match_gen_jet_eff_num = {v: ROOT.TH1D(f"jet_match_gen_eff_num_{v}", f"Matched Gen, {v.capitalize()} Jet Eff Num", len(JETPT_BINS)-1, jetpt_edges) for v in VERSIONS}
    h_all_gen_jet_eff_den = {v: ROOT.TH1D(f"jet_all_gen_eff_den_{v}", f"Total Gen, {v.capitalize()} Jet Eff Den", len(JETPT_BINS)-1, jetpt_edges) for v in VERSIONS}
    h_match_rec_jet_pur_num = {v: ROOT.TH1D(f"jet_match_rec_pur_num_{v}", f"Matched Rec, {v.capitalize()} Jet Pur Num", len(JETPT_BINS)-1, jetpt_edges) for v in VERSIONS}
    h_all_rec_jet_pur_den = {v: ROOT.TH1D(f"jet_all_rec_pur_den_{v}", f"Total Rec, {v.capitalize()} Jet Pur Den", len(JETPT_BINS)-1, jetpt_edges) for v in VERSIONS}

    # Splitting Efficiency/Purity & Lund Plane (2D: ln(kT) vs ln(R/dR))
    h_lund_matched_gen = ROOT.TH2D("lund_matched_gen", "Lund Plane Matched Gen;ln(R/#Delta R);ln(k_{T})", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS)
    h_lund_matched_rec = ROOT.TH2D("lund_matched_rec", "Lund Plane Matched Rec;ln(R/#Delta R);ln(k_{T})", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS)
    h_lund_all_gen = ROOT.TH2D("lund_all_gen", "Lund Plane All Gen;ln(R/#Delta R);ln(k_{T})", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS)
    h_lund_all_rec = ROOT.TH2D("lund_all_rec", "Lund Plane All Rec;ln(R/#Delta R);ln(k_{T})", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS)

    # Pair level (RL)
    h_match_gen_pair_eff_num = {}
    h_all_gen_pair_eff_den = {}
    h_match_rec_pair_pur_num = {}
    h_all_rec_pair_pur_den = {}
    h_res_rl = {}
    h_res_w = {}
    for lab in EEC_LABELS:
        h_match_gen_pair_eff_num[lab] = ROOT.TH1D(f"pair_match_gen_eff_num_{lab}", f"Matched Gen, Pair Eff Num {lab}", RL_NBINS, RL_BINS)
        h_all_gen_pair_eff_den[lab] = ROOT.TH1D(f"pair_all_gen_eff_den_{lab}", f"Total Gen, Pair Eff Den {lab}", RL_NBINS, RL_BINS)
        h_match_rec_pair_pur_num[lab] = ROOT.TH1D(f"pair_match_rec_pur_num_{lab}", f"Matched Rec, Pair Pur Num {lab}", RL_NBINS, RL_BINS)
        h_all_rec_pair_pur_den[lab] = ROOT.TH1D(f"pair_all_rec_pur_den_{lab}", f"Total Rec, Pair Pur Den {lab}", RL_NBINS, RL_BINS)


    # Lund plane differential
    h_lund_all_gen_pt = {a: ROOT.TH2D(f"h_lund_all_gen_pt{b}-{c}", f"Lund all gen pt:{b}-{c}", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS) for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_lund_all_rec_pt = {a: ROOT.TH2D(f"h_lund_all_rec_pt{b}-{c}", f"Lund all rec pt:{b}-{c}", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS) for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_lund_matched_gen_pt = {a: ROOT.TH2D(f"h_lund_matched_gen_pt{b}-{c}", f"Lund matched gen pt:{b}-{c}", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS) for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_lund_matched_rec_pt = {a: ROOT.TH2D(f"h_lund_matched_rec_pt{b}-{c}", f"Lund matched rec pt:{b}-{c}", len(LUND_RD_BINS)-1, LUND_RD_BINS, len(LUND_KT_BINS)-1, LUND_KT_BINS) for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}

    # Tracking efficiency/purity/residuals
    h_trk_eff_num = ROOT.TH1D("trk_eff_num", "Track Efficiency Num;p_{T}", 250, 0, 250)
    h_trk_eff_den = ROOT.TH1D("trk_eff_den", "Track Efficiency Den;p_{T}", 250, 0, 250)
    h_trk_pur_num = ROOT.TH1D("trk_pur_num", "Track Purity Num;p_{T}", 250, 0, 250)
    h_trk_pur_den = ROOT.TH1D("trk_pur_den", "Track Purity Den;p_{T}", 250, 0, 250)
    h_trk_res_pt = ROOT.TH2D("trk_res_pt", "Track pT Residual;p_{T}^{part};(p_{T}^{part}-p_{T}^{det})/p_{T}^{part}", 250, 0, 250, 100, -1, 1)
    
    # Pair efficiency/purity/residuals
    h_match_gen_pair_eff_num_pt = {a: {lab: ROOT.TH1D(f"pair_match_gen_eff_num_pt{b}-{c}_{lab}", f"Matched Gen, Pair Eff Num pt:{b}-{c} {lab}", RL_NBINS, RL_BINS) for lab in EEC_LABELS} for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_all_gen_pair_eff_den_pt = {a: {lab: ROOT.TH1D(f"pair_all_gen_eff_den_pt{b}-{c}_{lab}", f"Total Gen, Pair Eff Den pt:{b}-{c} {lab}", RL_NBINS, RL_BINS) for lab in EEC_LABELS} for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_match_rec_pair_pur_num_pt = {a: {lab: ROOT.TH1D(f"pair_match_rec_pur_num_pt{b}-{c}_{lab}", f"Matched Rec, Pair Pur Num pt:{b}-{c} {lab}", RL_NBINS, RL_BINS) for lab in EEC_LABELS} for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    h_all_rec_pair_pur_den_pt = {a: {lab: ROOT.TH1D(f"pair_all_rec_pur_den_pt{b}-{c}_{lab}", f"Total Rec, Pair Pur Den pt:{b}-{c} {lab}", RL_NBINS, RL_BINS) for lab in EEC_LABELS} for a,(b,c) in enumerate(zip(JETPT_BINS, JETPT_BINS[1:]))}
    for lab in EEC_LABELS:
        h_res_rl[lab] = ROOT.TH2D(f"res_rl_{lab}", f"RL residual {lab};R_{{L}}^{{part}};(R_{{L}}^{{part}}-R_{{L}}^{{det}})/R_{{L}}^{{part}}", RL_NBINS, RL_BINS, 100, -1, 1)
        h_res_w[lab] = ROOT.TH2D(f"res_w_{lab}", f"Weight residual {lab};w^{{part}};(w^{{part}}-w^{{det}})/w^{{part}}", W_NBINS, W_BINS, 100, -1, 1)


    # -----------------------------------------------------------------------
    # Read parquet, regroup jets by (event, level)
    # -----------------------------------------------------------------------
    jets = ak.from_parquet(args.input)

    # Exit early if the input file has no entries
    if len(jets) == 0:
        print(f"Warning: File '{args.input}' contains 0 jet entries. Exiting process.")
        return

    # --- Global Jet Totals for scalar eff/pur (keeping the previous requested feature) ---
    n_part_jets_total = len(jets[jets.level == "part"])
    n_det_jets_total = len(jets[jets.level == "det"])
    n_matched_jets = len(jets[(jets.level == "det") & (jets.is_matched)])

    # --- Pair Totals for scalar eff/pur ---
    n_part_pairs_total = {lab: 0 for lab in EEC_LABELS}
    n_det_pairs_total = {lab: 0 for lab in EEC_LABELS}
    n_matched_pairs = {lab: 0 for lab in EEC_LABELS}

    # We need, per event, the matched det/part jet pairs. In the parquet each
    # jet row has: event, level, jet_index, is_matched, match_index, ...
    # Build lookup: (event, level, jet_index) -> row
    events = np.asarray(jets.event)
    levels = np.asarray(jets.level)
    jidx   = np.asarray(jets.jet_index)

    # index rows for fast retrieval
    row_of = {}
    for r in range(len(jets)):
        row_of[(int(events[r]), str(levels[r]), int(jidx[r]))] = r

    n_pairs = 0
    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)

    counter1 = 0; counter2 = 0; counter3 = 0; counter4 = 0 #(counter2/counter1 should be eff, 4/3 should be purity)

    for r in range(len(jets)):

        # make cuts on jets
        # don't need because you can just see the groomed jet pt distribution in the histogram
        # if jets[r].jetpt < 10 or jets[r].jetpt > 200: #Limit jets binned in ungroomed jet pt to 10-200 gev

        # Fill jet eff/pur before looking at matched jets
        if "ungroomed" in VERSIONS:
            if str(levels[r]) == "part":
                h_all_gen_jet_eff_den["ungroomed"].Fill(jets[r].jet_pt, float(jets[r].mc_weight))
                counter1 += 1
                if bool(jets[r].is_matched):
                    h_match_gen_jet_eff_num["ungroomed"].Fill(jets[r].jet_pt, float(jets[r].mc_weight))
                    counter2 += 1
            if str(levels[r]) == "det":
                h_all_rec_jet_pur_den["ungroomed"].Fill(jets[r].jet_pt, float(jets[r].mc_weight))
                counter3 += 1
                if bool(jets[r].is_matched):
                    h_match_rec_jet_pur_num["ungroomed"].Fill(jets[r].jet_pt, float(jets[r].mc_weight))
                    counter4 += 1

        # For groomed versions, we only fill denominators for unmatched jets here
        # to avoid rebuilding matched jets twice.
        if "groomed" in VERSIONS and not bool(jets[r].is_matched):
            jet_consts = build_pseudojets(jets[r].const_pt, jets[r].const_eta, jets[r].const_phi, jets[r].const_label)
            jet, cs = recluster_ca(jet_consts)
            if jet is not None:
                lund_gen = fjcontrib.LundGenerator(jet_def_ca)
                jet_lund = lund_gen.result(jet)
                jet_splitting = select_split_sd(jet_lund, SD_ZCUT)
                if jet_splitting is not None:
                    groomed_jet = jet_splitting.pair()
                    if str(levels[r]) == "part":
                        h_all_gen_jet_eff_den["groomed"].Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                    if str(levels[r]) == "det":
                        h_all_rec_jet_pur_den["groomed"].Fill(groomed_jet.perp(), float(jets[r].mc_weight))

        # ---- pT-differential Lund plane ---
        # use ungroomed pT for binning
        bin_pt = np.digitize(jets[r].jet_pt, JETPT_BINS) - 1
        if 0 <= bin_pt < n_bins_pt:
            # Rebuild for Lund plane
            jet_consts_lund = build_pseudojets(jets[r].const_pt, jets[r].const_eta, jets[r].const_phi, jets[r].const_label)
            jet_l, _ = recluster_ca(jet_consts_lund)
            if jet_l is not None:
                l_gen = fjcontrib.LundGenerator(jet_def_ca)
                l_lund = l_gen.result(jet_l)
                for d in l_lund:
                    val_rd = math.log10(JET_R/d.Delta())
                    val_kt = math.log10(d.kt())
                    if str(levels[r]) == "part":
                        h_lund_all_gen_pt[bin_pt].Fill(val_rd, val_kt, float(jets[r].mc_weight))
                    if str(levels[r]) == "det":
                        h_lund_all_rec_pt[bin_pt].Fill(val_rd, val_kt, float(jets[r].mc_weight))

                # If matched, also fill matched Lund planes
                if bool(jets[r].is_matched):
                    if str(levels[r]) == "part":
                        for d in l_lund:
                            h_lund_matched_gen_pt[bin_pt].Fill(math.log10(JET_R/d.Delta()), math.log10(d.kt()), float(jets[r].mc_weight))
                    if str(levels[r]) == "det":
                        for d in l_lund:
                            h_lund_matched_rec_pt[bin_pt].Fill(math.log10(JET_R/d.Delta()), math.log10(d.kt()), float(jets[r].mc_weight))


        # Now continue to looking for matched jets
        if str(levels[r]) != "det":
            continue
        det = jets[r]
        if not bool(det.is_matched):
            continue
        ev = int(det.event)
        part_jidx = int(det.match_index)
        pr = row_of.get((ev, "part", part_jidx))
        if pr is None:
            continue
        part = jets[pr]

        # ---- event weight from the parquet row ----
        mc_weight = float(det.mc_weight)

        # ---- rebuild constituents ----
        det_consts = build_pseudojets(det.const_pt, det.const_eta,
                                      det.const_phi, det.const_label)
        part_consts = build_pseudojets(part.const_pt, part.const_eta,
                                       part.const_phi, part.const_label)

        # ---- recluster C/A ----
        det_jet, det_cs = recluster_ca(det_consts)
        part_jet, part_cs = recluster_ca(part_consts)
        if det_jet is None or part_jet is None:
            continue

        # ---- LundGenerator + SD split ----
        lund_gen = fjcontrib.LundGenerator(jet_def_ca)

        det_lund = lund_gen.result(det_jet)
        part_lund = lund_gen.result(part_jet)

        det_d = select_split_sd(det_lund, SD_ZCUT)
        part_d = select_split_sd(part_lund, SD_ZCUT)
        if det_d is None or part_d is None:
            continue  # both must pass SD to enter the response

        det_rad = det_d.pair()
        part_rad = part_d.pair()

        # Fill groomed efficiency/purity for matched jets
        if "groomed" in VERSIONS:
            h_match_gen_jet_eff_num["groomed"].Fill(part_rad.perp(), mc_weight)
            h_all_gen_jet_eff_den["groomed"].Fill(part_rad.perp(), mc_weight)
            h_match_rec_jet_pur_num["groomed"].Fill(det_rad.perp(), mc_weight)
            h_all_rec_jet_pur_den["groomed"].Fill(det_rad.perp(), mc_weight)

        # Get subjets of the chosen radiator
        det_sub = sorted(det_rad.pieces(), key=lambda x: x.pt(), reverse=True)
        part_sub = sorted(part_rad.pieces(), key=lambda x: x.pt(), reverse=True)
        if len(det_sub) != 2 or len(part_sub) != 2:
            continue
        det_A, det_B = det_sub
        part_A, part_B = part_sub

        # ---- weighting scale = radiator.perp() (per your request), also the groomed jet pt ----
        det_scale = det_rad.perp()
        part_scale = part_rad.perp()

        # jet pt used for response pt axes
        det_ptjet = det_jet.perp()
        part_ptjet = part_jet.perp()

        # ---- fill 2D jet pt and groomed pt responses (one per matched pair) ----
        for v in VERSIONS:
            d_pt = det_scale if v == "groomed" else det_ptjet
            p_pt = part_scale if v == "groomed" else part_ptjet
            h_resp_jetpt[v].Fill(d_pt, p_pt, mc_weight)
            h1_reco[v].Fill(d_pt, mc_weight)
            h1_gen[v].Fill(p_pt, mc_weight)
            response1D[v].Fill(d_pt, p_pt, mc_weight)
            if p_pt > 0:
                h_res_jetpt[v].Fill(p_pt, (p_pt - d_pt) / p_pt, mc_weight)

        # ---- pT-differential Response and Pair Metrics ----
        # use ungroomed pT for binning the differential file
        bin_pt = np.digitize(det_ptjet, JETPT_BINS) - 1
        if 0 <= bin_pt < n_bins_pt:
            for v in VERSIONS:
                fill_det_jetpt = det_scale if v == "groomed" else det_ptjet
                fill_part_jetpt = part_scale if v == "groomed" else part_ptjet
                for lab in EEC_LABELS:
                    # Note: Actual filling of these is done in the EEC loop below for efficiency
                    # but we can pre-calculate the bin here.
                    pass

        n_pairs += 1 # number of matched jets (det/part)


        # make cuts on matched jets - keep groomed jet pt between 10 and 200 gev
        if det_ptjet >= 10 and det_ptjet < 200 and part_ptjet >= 10 and part_ptjet < 200:

            # Study matched splittings and splittings eff/pur
            h_lund_all_gen.Fill(math.log10(JET_R/part_d.Delta()), math.log10(part_d.kt()), mc_weight)
            h_lund_all_rec.Fill(math.log10(JET_R/det_d.Delta()), math.log10(det_d.kt()), mc_weight)
            # print("split det:", math.log10(JET_R/part_d.Delta()), math.log10(part_d.kt()))
            matched_splitting = match_splittings(det_B, part_B)
            if matched_splitting:
                h_lund_matched_gen.Fill(math.log10(JET_R/part_d.Delta()), math.log10(part_d.kt()), mc_weight)
                h_lund_matched_rec.Fill(math.log10(JET_R/det_d.Delta()), math.log10(det_d.kt()), mc_weight)


        # ---- selected constituents per subjet ----
        det_cA = selected_constituents(det_A, TRK_THRD)
        det_cB = selected_constituents(det_B, TRK_THRD)
        part_cA = selected_constituents(part_A, TRK_THRD)
        part_cB = selected_constituents(part_B, TRK_THRD)

        # radiator constituents = full groomed jet constituents
        det_crad = selected_constituents(det_rad, TRK_THRD)
        part_crad = selected_constituents(part_rad, TRK_THRD)

        # ungroomed jet constituents
        det_full = selected_constituents(det_jet, TRK_THRD)
        part_full = selected_constituents(part_jet, TRK_THRD)


        # ---- track-level matching (efficiency, purity, residuals) ----
        # match each detector track to the closest generator particle
        matched_tracks = 0
        if det_ptjet >= 10 and det_ptjet < 200 and part_ptjet >= 10 and part_ptjet < 200:

            for dtrk in det_full:
                best_dr = 0.1 # matching window
                best_part = None
                for ptrk in part_full:
                    dr = dtrk.delta_R(ptrk)
                    if dr < best_dr:
                        best_dr = dr
                        best_part = ptrk

                h_trk_pur_den.Fill(dtrk.perp(), mc_weight)
                if best_part:
                    matched_tracks += 1
                    h_trk_pur_num.Fill(dtrk.perp(), mc_weight)
                    # track residual: (part - det) / part
                    if best_part.perp() > 0:
                        h_trk_res_pt.Fill(best_part.perp(), (best_part.perp() - dtrk.perp()) / best_part.perp(), mc_weight)

            # efficiency: match generator particles to detector tracks
            for ptrk in part_full:
                h_trk_eff_den.Fill(ptrk.perp(), mc_weight)
                is_matched = False
                for dtrk in det_full:
                    if dtrk.delta_R(ptrk) < 0.01: # tight window for efficiency
                        is_matched = True
                        break
                if is_matched:
                    h_trk_eff_num.Fill(ptrk.perp(), mc_weight)

        # ---- compute EEC pairs at both levels (radiator-pt weighted) ----
        eec_sets = {
            "full_ungroomed": (compute_eec_pairs(det_full, det_jet.perp()),
                               compute_eec_pairs(part_full, part_jet.perp())),
            "AA":  (compute_eec_pairs(det_cA, det_scale),
                    compute_eec_pairs(part_cA, part_scale)),
            "BB":  (compute_eec_pairs(det_cB, det_scale),
                    compute_eec_pairs(part_cB, part_scale)),
            "AB":  (compute_eec_pairs(det_cA, det_scale, det_cB),
                    compute_eec_pairs(part_cA, part_scale, part_cB)),
            "rad": (compute_eec_pairs(det_crad, det_scale),
                    compute_eec_pairs(part_crad, part_scale)),
        }

        # ---- pair-level match and fill 6D ----
        for lab, (det_pairs, part_pairs) in eec_sets.items():
            matched, tr_unmatched, det_unmatched = match_eec_pairs(det_pairs, part_pairs)
                
            # update pair counters
            n_det_pairs_total[lab] += len(det_pairs)
            n_part_pairs_total[lab] += len(part_pairs)
            n_matched_pairs[lab] += len(matched)
            
            if det_ptjet >= 10 and det_ptjet < 200 and part_ptjet >= 10 and part_ptjet < 200:
                
                # fill differential pair eff/pur
                for dpair in det_pairs:
                    h_all_rec_pair_pur_den[lab].Fill(dpair[0], mc_weight)
                for ppair in part_pairs:
                    h_all_gen_pair_eff_den[lab].Fill(ppair[0], mc_weight)
                for dpair, ppair in matched:
                    h_match_rec_pair_pur_num[lab].Fill(dpair[0], mc_weight)
                    h_match_gen_pair_eff_num[lab].Fill(ppair[0], mc_weight)

                # fill differential pair eff/pur
                bin_pt_pairs = np.digitize(det_ptjet, JETPT_BINS) - 1 # use det jet pt for binning
                if 0 <= bin_pt_pairs < n_bins_pt:
                    for dpair in det_pairs:
                        h_all_rec_pair_pur_den_pt[bin_pt_pairs][lab].Fill(dpair[0], mc_weight)
                    for ppair in part_pairs:
                        h_all_gen_pair_eff_den_pt[bin_pt_pairs][lab].Fill(ppair[0], mc_weight)
                    for dpair, ppair in matched:
                        h_match_rec_pair_pur_num_pt[bin_pt_pairs][lab].Fill(dpair[0], mc_weight)
                        h_match_gen_pair_eff_num_pt[bin_pt_pairs][lab].Fill(ppair[0], mc_weight)

            for v in VERSIONS:
                fill_det_jetpt = det_scale if v == "groomed" else det_ptjet
                fill_part_jetpt = part_scale if v == "groomed" else part_ptjet

                hs = resp6[v][lab]
                hru_response = roounfold_resp6[v][lab]
                for dpair, ppair in matched:
                    rl_det, w_det = dpair[0], dpair[1]
                    rl_part, w_part = ppair[0], ppair[1]
                    x = array.array("d", [fill_det_jetpt, fill_part_jetpt, # groomed jet pt
                                        rl_det, rl_part, w_det, w_part])
                    hs.Fill(x, mc_weight)

                    hru_response.Fill(fill_det_jetpt, rl_det, w_det, fill_part_jetpt, rl_part, w_part, mc_weight)
                    reco_3[v][lab].Fill(fill_det_jetpt, rl_det, w_det, mc_weight)
                    gen_3[v][lab].Fill(fill_part_jetpt, rl_part, w_part, mc_weight)
                    reco_unmatched_3[v][lab].Fill(fill_det_jetpt, rl_det, w_det, mc_weight)
                    gen_unmatched_3[v][lab].Fill(fill_part_jetpt, rl_part, w_part, mc_weight)

                    if det_ptjet >= 10 and det_ptjet < 200 and part_ptjet >= 10 and part_ptjet < 200:
                        if rl_part > 0:
                            h_res_rl[lab].Fill(rl_part, (rl_part - rl_det) / rl_part, mc_weight)
                        if w_part > 0:
                            h_res_w[lab].Fill(w_part, (w_part - w_det) / w_part, mc_weight)


                for dpair in det_unmatched:
                    rl_det, w_det = dpair[0], dpair[1]
                    reco_unmatched_3[v][lab].Fill(fill_det_jetpt, rl_det, w_det, mc_weight)

                    # bin_pt_resp = np.digitize(det_ptjet, JETPT_BINS) - 1
                    # if 0 <= bin_pt_resp < n_bins_pt:
                    #     reco_unmatched_3_pt[bin_pt_resp][v][lab].Fill(rl_det, w_det, mc_weight)

                for tpair in tr_unmatched:
                    rl_part, w_part = tpair[0], tpair[1]
                    # if rl_part >= 0 and rl_part < RL_MAX:
                    hru_response.Miss(fill_part_jetpt, rl_part, w_part, mc_weight)
                    gen_unmatched_3[v][lab].Fill(fill_part_jetpt, rl_part, w_part, mc_weight)

                    # bin_pt_resp = np.digitize(det_ptjet, JETPT_BINS) - 1
                    # if 0 <= bin_pt_resp < n_bins_pt:
                    #     gen_unmatched_3_pt[bin_pt_resp][v][lab].Fill(rl_part, w_part, mc_weight)

            
        

    print(f"Filled response from {n_pairs} matched jet pairs passing SD at both levels.")

    # # Divide appropriate histograms to get efficiency/purity
    # # A. jets
    # h_jet_eff_ungroomed = h_match_gen_jet_eff_num_ungroomed.Clone("jet_efficiency_ungroomed")
    # h_jet_eff_ungroomed.Divide(h_all_gen_jet_eff_den_ungroomed)
    # h_jet_eff_ungroomed.SetTitle("Ungroomed Jet Efficiency")
    
    # h_jet_pur_ungroomed = h_match_rec_jet_pur_num_ungroomed.Clone("jet_purity_ungroomed")
    # h_jet_pur_ungroomed.Divide(h_all_rec_jet_pur_den_ungroomed)
    # h_jet_pur_ungroomed.SetTitle("Ungroomed Jet Purity")

    # h_jet_eff_groomed = h_match_gen_jet_eff_num_groomed.Clone("jet_efficiency_groomed")
    # h_jet_eff_groomed.Divide(h_all_gen_jet_eff_den_groomed)
    # h_jet_eff_groomed.SetTitle("Groomed Jet Efficiency")

    # h_jet_pur_groomed = h_match_rec_jet_pur_num_groomed.Clone("jet_purity_groomed")
    # h_jet_pur_groomed.Divide(h_all_rec_jet_pur_den_groomed)
    # h_jet_pur_groomed.SetTitle("Groomed Jet Purity")

    # # B. splittings
    # h_lund_eff = h_lund_matched_gen.Clone("lund_split_efficiency")
    # h_lund_eff.Divide(h_lund_all_gen)
    # h_lund_eff.SetTitle("Lund Plane SD z_{cut}=0.1 Splittings Efficiency")

    # h_lund_pur = h_lund_matched_rec.Clone("lund_split_purity")
    # h_lund_pur.Divide(h_lund_all_rec)
    # h_lund_pur.SetTitle("Lund Plane SD z_{cut}=0.1 Splittings Purity")
    
    # # C. pairs
    # h_pair_eff = {}
    # h_pair_pur = {}
    # for lab in EEC_LABELS:
    #     h_pair_eff[lab] = h_match_gen_pair_eff_num[lab].Clone(f"pair_efficiency_{lab}")
    #     h_pair_eff[lab].Divide(h_all_gen_pair_eff_den[lab])
    #     h_pair_eff[lab].SetTitle(f"Pair Efficiency {lab}")

    #     h_pair_pur[lab] = h_match_rec_pair_pur_num[lab].Clone(f"pair_purity_{lab}")
    #     h_pair_pur[lab].Divide(h_all_rec_pair_pur_den[lab])
    #     h_pair_pur[lab].SetTitle(f"Pair Purity {lab}")


    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    fout = ROOT.TFile(args.output, "RECREATE")
    fout.cd()

    for v in VERSIONS:
        h_resp_jetpt[v].Write()
        h_res_jetpt[v].Write()
        response1D[v].Write()
        for lab in EEC_LABELS:
            resp6[v][lab].Write()
            roounfold_resp6[v][lab].Write()

            reco_3[v][lab].Write()
            reco_unmatched_3[v][lab].Write()
            gen_3[v][lab].Write()
            gen_unmatched_3[v][lab].Write()

    # Write differential efficiency and purity - ungroomed and groomed jet pt
    for v in VERSIONS:
        h_match_gen_jet_eff_num[v].Write()
        h_all_gen_jet_eff_den[v].Write()
        h_match_rec_jet_pur_num[v].Write()
        h_all_rec_jet_pur_den[v].Write()
        # h_jet_eff.Write()
        # h_jet_pur.Write()


    h_lund_matched_gen.Write()
    h_lund_matched_rec.Write()
    h_lund_all_gen.Write()
    h_lund_all_rec.Write()
    # h_lund_eff.Write()
    # h_lund_pur.Write()

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

    #     h_pair_eff[lab].Write()
    #     h_pair_pur[lab].Write()

    # -----------------------------------------------------------------------
    # Write pT-differential output
    # -----------------------------------------------------------------------
    # Strip .root extension if present and append _ptdifferential.root
    output_diff_name = args.output.rsplit(".", 1)[0] + "_ptdifferential.root" if "." in args.output else args.output + "_ptdifferential.root"
    fout_diff = ROOT.TFile(output_diff_name, "RECREATE")
    fout_diff.cd()

    for b in range(n_bins_pt):
        # Lund plane
        h_lund_all_gen_pt[b].Write()
        h_lund_all_rec_pt[b].Write()
        h_lund_matched_gen_pt[b].Write()
        h_lund_matched_rec_pt[b].Write()

        # Pair metrics
        for lab in EEC_LABELS:
            h_match_gen_pair_eff_num_pt[b][lab].Write()
            h_all_gen_pair_eff_den_pt[b][lab].Write()
            h_match_rec_pair_pur_num_pt[b][lab].Write()
            h_all_rec_pair_pur_den_pt[b][lab].Write()


    fout_diff.Close()

    # Ensure we are back in the main output file for the summary
    fout.cd()

    # Write scalar efficiency and purity
    # Jet level
    jet_eff = n_matched_jets / n_part_jets_total if n_part_jets_total > 0 else 0
    jet_pur = n_matched_jets / n_det_jets_total if n_det_jets_total > 0 else 0

    # Splitting level
    # Also add splitting efficiency/purity (integrated over the Lund plane)
    # Using the total counts of matched vs all splittings
    # Note: n_pairs is the count of matched jets that both pass SD.
    # To get the total number of jets that pass SD at each level:
    # We can use the integral of h_lund_all_gen and h_lund_all_rec
    total_gen_splits = h_lund_all_gen.Integral()
    total_rec_splits = h_lund_all_rec.Integral()
    total_matched_splits = h_lund_matched_gen.Integral()

    split_eff = total_matched_splits / total_gen_splits if total_gen_splits > 0 else 0
    split_pur = total_matched_splits / total_rec_splits if total_rec_splits > 0 else 0

    # Pair level
    for lab in EEC_LABELS:
        pair_eff = n_matched_pairs[lab] / n_part_pairs_total[lab] if n_part_pairs_total[lab] > 0 else 0
        pair_pur = n_matched_pairs[lab] / n_det_pairs_total[lab] if n_det_pairs_total[lab] > 0 else 0

    # Scalar efficiency/purity summary plot
    # We'll use a TH1D where the x-axis represents the different categories
    h_summary = ROOT.TH1D("summary_efficiencies", "Global Efficiency and Purity;Metric;Value", 20, 0, 20)

    # Map labels to bin indices
    # 1: Jet Eff, 2: Jet Pur
    h_summary.SetBinContent(1, jet_eff)
    h_summary.GetXaxis().SetBinLabel(1, "Jet Eff")
    h_summary.SetBinContent(2, jet_pur)
    h_summary.GetXaxis().SetBinLabel(2, "Jet Pur")

    h_summary.SetBinContent(3, split_eff)
    h_summary.GetXaxis().SetBinLabel(3, "Split Eff")
    h_summary.SetBinContent(4, split_pur)
    h_summary.GetXaxis().SetBinLabel(4, "Split Pur")

    bin_idx = 5
    for lab in EEC_LABELS:
        p_eff = n_matched_pairs[lab] / n_part_pairs_total[lab] if n_part_pairs_total[lab] > 0 else 0
        p_pur = n_matched_pairs[lab] / n_det_pairs_total[lab] if n_det_pairs_total[lab] > 0 else 0

        h_summary.SetBinContent(bin_idx, p_eff)
        h_summary.GetXaxis().SetBinLabel(bin_idx, f"Pair {lab} Eff")
        bin_idx += 1

        h_summary.SetBinContent(bin_idx, p_pur)
        h_summary.GetXaxis().SetBinLabel(bin_idx, f"Pair {lab} Pur")
        bin_idx += 1

    
    h_summary.Write()

    fout.Close()
    print(f"Wrote response matrices and differential metrics to {args.output}")
    print(f"Jet Efficiency: {jet_eff:.4f}, Purity: {jet_pur:.4f}")
    for lab in EEC_LABELS:
        print(f"Pair {lab} - Efficiency: {n_matched_pairs[lab]/n_part_pairs_total[lab] if n_part_pairs_total[lab]>0 else 0:.4f}, Purity: {n_matched_pairs[lab]/n_det_pairs_total[lab] if n_det_pairs_total[lab]>0 else 0:.4f}")

    print("counters:", counter1, counter2, counter3, counter4)


if __name__ == "__main__":
    main()