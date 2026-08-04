'''
The response matrices needed for this analysis: jet pt, RL, weight
sd? radiator pt?
how is the lund plane corrected?
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

W_NBINS = 100
W_MIN, W_MAX = 0.0, 1.0 #0.3
W_BINS = np.linspace(W_MIN, W_MAX, W_NBINS + 1)

EEC_LABELS = ["AA", "AB", "BB", "rad"]


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
    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 5.0)  # large R to catch all
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

    if c_select_B is None:
        # single collection: indices1/indices2 both index into c_select
        for k in range(n):
            i = int(idx1[k])
            j = int(idx2[k])
            lab_i = c_select[i].user_index()
            lab_j = c_select[j].user_index()
            pairs.append((rs[k], ws[k], lab_i, lab_j))
    else:
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
    """
    # index particle pairs by frozenset of the two labels
    part_map = {}
    for p in part_pairs:
        key = frozenset((p[2], p[3]))
        # a jet can have the pair (i,j) and (j,i) counted twice;
        # store a list so both orderings can be matched
        part_map.setdefault(key, []).append(p)

    matched = []
    used = {}  # key -> count already consumed
    for d in det_pairs:
        key = frozenset((d[2], d[3]))
        candidates = part_map.get(key)
        if not candidates:
            continue
        c = used.get(key, 0)
        if c < len(candidates):
            matched.append((d, candidates[c]))
            used[key] = c + 1
    return matched


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
    jetpt_edges = array.array("d", JETPT_BINS)

    h_resp_jetpt = ROOT.TH2D("resp_jetpt", "jet p_{T} det vs part;p_{T}^{det};p_{T}^{part}",
                             len(JETPT_BINS) - 1, jetpt_edges,
                             len(JETPT_BINS) - 1, jetpt_edges)
    h_resp_groomed = ROOT.TH2D("resp_groomed_jetpt",
                               "groomed jet p_{T} det vs part;p_{T,g}^{det};p_{T,g}^{part}",
                               len(JETPT_BINS) - 1, jetpt_edges,
                               len(JETPT_BINS) - 1, jetpt_edges)

    # 6D THnSparse per EEC:  axes = pt_det, pt_part, RL_det, RL_part, w_det, w_part
    nbins6 = np.array([len(JETPT_BINS) - 1, len(JETPT_BINS) - 1,
                       RL_NBINS, RL_NBINS, W_NBINS, W_NBINS], dtype=np.int32)
    xmin6 = np.array([JETPT_BINS[0], JETPT_BINS[0], RL_MIN, RL_MIN, W_MIN, W_MIN], dtype=np.float64)
    xmax6 = np.array([JETPT_BINS[-1], JETPT_BINS[-1], RL_MAX, RL_MAX, W_MAX, W_MAX], dtype=np.float64)

    resp6 = {}
    for lab in EEC_LABELS:
        hs = ROOT.THnSparseD(
            f"resp6_{lab}",
            f"6D response {lab};p_{{T}}^{{det}};p_{{T}}^{{part}};R_{{L}}^{{det}};R_{{L}}^{{part}};w^{{det}};w^{{part}}",
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
        resp6[lab] = hs

    # -----------------------------------------------------------------------
    # Read parquet, regroup jets by (event, level)
    # -----------------------------------------------------------------------
    jets = ak.from_parquet(args.input)

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

    for r in range(len(jets)):
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
        jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
        lund_gen = fjcontrib.LundGenerator(jet_def_ca)

        det_lund = lund_gen.result(det_jet)
        part_lund = lund_gen.result(part_jet)

        det_d = select_split_sd(det_lund, SD_ZCUT)
        part_d = select_split_sd(part_lund, SD_ZCUT)
        if det_d is None or part_d is None:
            continue  # both must pass SD to enter the response

        det_rad = det_d.pair()
        part_rad = part_d.pair()

        det_sub = sorted(det_rad.pieces(), key=lambda x: x.pt(), reverse=True)
        part_sub = sorted(part_rad.pieces(), key=lambda x: x.pt(), reverse=True)
        if len(det_sub) != 2 or len(part_sub) != 2:
            continue
        det_A, det_B = det_sub
        part_A, part_B = part_sub

        # ---- weighting scale = radiator.perp() (per your request) ----
        det_scale = det_rad.perp()
        part_scale = part_rad.perp()

        # jet pt used for response pt axes
        det_ptjet = det_jet.perp()
        part_ptjet = part_jet.perp()

        # ---- fill 2D jet pt and groomed pt responses (one per matched pair) ----
        h_resp_jetpt.Fill(det_ptjet, part_ptjet, mc_weight)
        h_resp_groomed.Fill(det_rad.perp(), part_rad.perp(), mc_weight)
        n_pairs += 1

        # ---- selected constituents per subjet ----
        det_cA = selected_constituents(det_A, TRK_THRD)
        det_cB = selected_constituents(det_B, TRK_THRD)
        part_cA = selected_constituents(part_A, TRK_THRD)
        part_cB = selected_constituents(part_B, TRK_THRD)

        # radiator constituents = full groomed jet constituents
        det_crad = selected_constituents(det_rad, TRK_THRD)
        part_crad = selected_constituents(part_rad, TRK_THRD)

        # ---- compute EEC pairs at both levels (radiator-pt weighted) ----
        eec_sets = {
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
            matched = match_eec_pairs(det_pairs, part_pairs)
            hs = resp6[lab]
            for dpair, ppair in matched:
                rl_det, w_det = dpair[0], dpair[1]
                rl_part, w_part = ppair[0], ppair[1]
                x = array.array("d", [det_rad.perp(), part_rad.perp(), # groomed jet pt
                                      rl_det, rl_part, w_det, w_part])
                hs.Fill(x, mc_weight)

    print(f"Filled response from {n_pairs} matched jet pairs passing SD at both levels.")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    fout = ROOT.TFile(args.output, "RECREATE")
    fout.cd()
    h_resp_jetpt.Write()
    h_resp_groomed.Write()
    for lab in EEC_LABELS:
        resp6[lab].Write()
    fout.Close()
    print(f"Wrote response matrices to {args.output}")


if __name__ == "__main__":
    main()