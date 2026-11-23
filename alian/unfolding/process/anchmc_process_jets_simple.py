#!/usr/bin/env python3

"""
In this script:
JUST doing groomed bins version, zcut = 0.1
Plotting the jet pt spectrum and EECs from the anchored MC
if ungroomed bins desired, add an argument (see data_process_jets_simple.py)
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
PT_SLICES = [(10, 20), (20, 40), (40, 60), (60, 80), (80, 100),
            (100, 120), (120, 150), (150, 200), (200, 500)]

RL_NBINS = 25
RL_MIN, RL_MAX = 0.01, 1.0
RL_BINS = np.logspace(np.log10(RL_MIN), np.log10(RL_MAX), RL_NBINS + 1)

EEC_LABELS = ["AA", "AB", "BB", "rad"]
VERSIONS = ["groomed"] #, "ungroomed"]
WEIGHT_LABELS = ["wwradpt"] #"wwjetpt", "wwnullpt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_pseudojets(const_pt, const_eta, const_phi, const_label):
    """Rebuild a vectorPJ from stored constituents; label -> user_index."""
    v = fj.vectorPJ()
    for pt, eta, phi, lab in zip(const_pt, const_eta, const_phi, const_label):
        # pj = fj.PseudoJet()
        # pj.reset_PtYPhiM(float(pt), float(eta), float(phi), PION_MASS)
        px, py = pt*math.cos(phi), pt*math.sin(phi)
        pz = pt*math.sinh(eta)
        E  = math.sqrt(px*px + py*py + pz*pz + PION_MASS*PION_MASS)
        pj = fj.PseudoJet(px, py, pz, E)
        # pj.reset_PtYPhiM(float(pt), float(eta), float(phi), PION_MASS)
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

# takes in a pt value, and returns pt bin that that pt value falls into - assuming no overlap in pt bins
def slices_for_pt(pt):
    """The slice a given pt falls into (slices may not overlap)."""
    if pt is None:
        return ()
    for (lo,hi) in PT_SLICES:
        if pt >= lo and pt < hi:
            return (lo, hi)
    return () #(lo, hi) for (lo, hi) in PT_SLICES if lo <= pt < hi

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


def fill_correlator(pairs, hist, weighted=True, mc_weight=1): #, hist_ptRL, avg_pt, 
    for rl_value, weight_value, _, _ in pairs:
        if weighted:
            hist.Fill(rl_value, weight_value*mc_weight)
        else:
            hist.Fill(rl_value, mc_weight)

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

def hist_names(obj, tag, wlabel):
        """Reproduce the original naming convention."""
        if obj == "full":
            if wlabel == "wwjetpt":
                return f"hist_full_{tag}", f"hist_full_ptRL_{tag}"
            return f"hist_full_{tag}_{wlabel}", f"hist_full_ptRL_{tag}_{wlabel}"
        return f"hist_{obj}_{tag}_{wlabel}", f"hist_{obj}_ptRL_{tag}_{wlabel}"

def book_hists(tag):
        nbins = RL_NBINS
        rl_log_bins = RL_BINS
        h = {}

        titles = {"AA": "AxA", "BB": "BxB", "AB": "AxB", "rad": "EEC"}
        for lab in EEC_LABELS:
            for wlabel in WEIGHT_LABELS:
                name_rl, _ = hist_names(lab, tag, wlabel)
                h[f"{lab}_{wlabel}"] = ROOT.TH1D(name_rl, f"{titles[lab]}; R_{{L}}", nbins, rl_log_bins)

        h['jetpt'] = ROOT.TH1D(f"hist_jetpt_{tag}", "jet p_{T}; p_{T,jet}",
                                   len(JETPT_BINS) - 1, JETPT_BINS)
        return h

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input parquet file (from script 1)")
    ap.add_argument("-o", "--output", default="anchmc_simple.root", help="output ROOT file")
    ap.add_argument("--corr-rootfile", default="", help="Corresponding input root file for the parquet file") # bc the parquet file only stores jets, not total number of events
    args = ap.parse_args()


    # -----------------------------------------------------------------------
    # Output histograms
    # -----------------------------------------------------------------------
    hNevents = ROOT.TH1D("hNevents", "hNevents", 2, -0.5, 1.5)

    hJetPtALL_det_matched_ungroomed = ROOT.TH1D("hJetPtALL_det_matched_ungroomed", "hJetPtALL_det_matched_ungroomed; jet pT", 500, 0, 500)
    hJetPtALL_det_all_ungroomed = ROOT.TH1D("hJetPtALL_det_all_ungroomed", "hJetPtALL_det_all_ungroomed; jet pT", 500, 0, 500)
    hJetPtALL_part_matched_ungroomed = ROOT.TH1D("hJetPtALL_part_matched_ungroomed", "hJetPtALL_part_matched_ungroomed; jet pT", 500, 0, 500)
    hJetPtALL_part_all_ungroomed = ROOT.TH1D("hJetPtALL_part_all_ungroomed", "hJetPtALL_part_all_ungroomed; jet pT", 500, 0, 500)

    hJetPtALL_det_matched_groomed = ROOT.TH1D("hJetPtALL_det_matched_groomed", "hJetPtALL_det_matched_groomed; jet pT", 500, 0, 500)
    hJetPtALL_det_all_groomed = ROOT.TH1D("hJetPtALL_det_all_groomed", "hJetPtALL_det_all_groomed; jet pT", 500, 0, 500)
    hJetPtALL_part_matched_groomed = ROOT.TH1D("hJetPtALL_part_matched_groomed", "hJetPtALL_part_matched_groomed; jet pT", 500, 0, 500)
    hJetPtALL_part_all_groomed = ROOT.TH1D("hJetPtALL_part_all_groomed", "hJetPtALL_part_all_groomed; jet pT", 500, 0, 500)

    inclusiveQA = {}
    inclusiveQA["allpt"] = ROOT.TH1D(f"QA_hist_full_ALL_wwjetpt", "Inclusive EEC; R_{L}", RL_NBINS, RL_BINS)
    hists_allpt_det_matched = book_hists("det_matched_allpt_sd0.1")
    hists_allpt_det_all = book_hists("det_all_allpt_sd0.1")
    hists_allpt_part_matched = book_hists("part_matched_allpt_sd0.1")
    hists_allpt_part_all = book_hists("part_all_allpt_sd0.1")

    # ---- book everything up front ----
    # root_outfile = ROOT.TFile(args.output, "RECREATE")
    hists_det_matched = {}
    hists_det_all = {}
    hists_part_matched = {}
    hists_part_all = {}
    for (lo, hi) in PT_SLICES:
        tag = f"jetpt{lo}_{hi}_sd0.1"
        hists_det_matched[(lo, hi)] = book_hists(f"det_matched_{tag}")
        hists_det_all[(lo, hi)] = book_hists(f"det_all_{tag}")
        hists_part_matched[(lo, hi)] = book_hists(f"part_matched_{tag}")
        hists_part_all[(lo, hi)] = book_hists(f"part_all_{tag}")
        inclusiveQA[(lo, hi)] = ROOT.TH1D(f"QA_hist_full_pt{lo}_{hi}_wwjetpt", "Inclusive EEC; R_{L}", RL_NBINS, RL_BINS)
        
    

    # -----------------------------------------------------------------------
    # Calculate number of events
    # -----------------------------------------------------------------------
    # Read number of events from corresponding rootfile
    total_num_events = 0
    if args.corr_rootfile != "":
        corr_file = ROOT.TFile.Open(args.corr_rootfile, "READ")
        t = corr_file.Get("eventTree")
        total_num_events = t.GetEntries()
        corr_file.Close()

    hNevents.Fill(0, total_num_events)
    b = hNevents.FindBin(0)
    hNevents.SetBinContent(b, total_num_events)
    hNevents.SetBinError(b, total_num_events ** 0.5)
    hNevents.SetEntries(total_num_events)

    
    # -----------------------------------------------------------------------
    # Read parquet, regroup jets by (event, level)
    # -----------------------------------------------------------------------
    jets = ak.from_parquet(args.input)

    # Exit early if the input file has no entries
    if len(jets) == 0:
        print(f"Warning: File '{args.input}' contains 0 jet entries. Exiting process.")
        return

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

    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)


    for r in range(len(jets)):

        # Fill ungroomed jet pt
        if str(levels[r]) == "part":
            hJetPtALL_part_all_ungroomed.Fill(jets[r].jet_pt, float(jets[r].mc_weight))
            if bool(jets[r].is_matched):
                hJetPtALL_part_matched_ungroomed.Fill(jets[r].jet_pt, float(jets[r].mc_weight))
        if str(levels[r]) == "det":
            hJetPtALL_det_all_ungroomed.Fill(jets[r].jet_pt, float(jets[r].mc_weight))
            if bool(jets[r].is_matched):
                hJetPtALL_det_matched_ungroomed.Fill(jets[r].jet_pt, float(jets[r].mc_weight))
                                    

        # Fill groomed jet pt
        jet_consts = build_pseudojets(jets[r].const_pt, jets[r].const_eta, jets[r].const_phi, jets[r].const_label)
        jet, _ = recluster_ca(jet_consts)
        if jet is not None:
            lund_gen = fjcontrib.LundGenerator(jet_def_ca)
            jet_lund = lund_gen.result(jet)
            jet_splitting = select_split_sd(jet_lund, SD_ZCUT)
            if jet_splitting is not None:
                groomed_jet = jet_splitting.pair()

                # Identify the pt slice that the groomed jet pt falls into
                ptslice = slices_for_pt(groomed_jet.perp())

                if str(levels[r]) == "part":
                    hJetPtALL_part_all_groomed.Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                    hists_allpt_part_all['jetpt'].Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                    if ptslice != ():
                        h_part_all = hists_part_all[ptslice]
                        h_part_all['jetpt'].Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                if str(levels[r]) == "det":
                    hJetPtALL_det_all_groomed.Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                    hists_allpt_det_all['jetpt'].Fill(groomed_jet.perp(), float(jets[r].mc_weight))
                    if ptslice != ():
                        h_det_all = hists_det_all[ptslice]
                        h_det_all['jetpt'].Fill(groomed_jet.perp(), float(jets[r].mc_weight))

        

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

        # Identify the pt slice that the groomed jet pt falls into
        det_ptslice = slices_for_pt(det_scale)
        part_ptslice = slices_for_pt(part_scale)

        # Fill groomed jet pt for the matches
        hJetPtALL_part_matched_groomed.Fill(part_scale, mc_weight)
        hJetPtALL_det_matched_groomed.Fill(det_scale, mc_weight)

        hists_allpt_det_matched['jetpt'].Fill(det_scale, mc_weight)
        hists_allpt_part_matched['jetpt'].Fill(part_scale, mc_weight)

        if det_ptslice != ():
            h_det_matched = hists_det_matched[det_ptslice]
            h_det_matched['jetpt'].Fill(det_scale, mc_weight)
        if part_ptslice != ():
            h_part_matched = hists_part_matched[part_ptslice]
            h_part_matched['jetpt'].Fill(part_scale, mc_weight)


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

        # Calculate EECs for full ungroomed jet
        inclusive_full_det_pairs = compute_eec_pairs(det_full, det_ptjet)
        inclusive_full_part_pairs = compute_eec_pairs(part_full, part_ptjet)
        fill_correlator(inclusive_full_det_pairs, inclusiveQA["allpt"])
        fill_correlator(inclusive_full_part_pairs, inclusiveQA["allpt"])
        if det_ptslice != ():
            fill_correlator(inclusive_full_det_pairs, inclusiveQA[det_ptslice])
        if part_ptslice != ():
            fill_correlator(inclusive_full_part_pairs, inclusiveQA[part_ptslice])

        # for wlabel, wval in weight_specs:
    
        # ---- compute EEC pairs at both levels (radiator-pt weighted) ----
        eec_sets = {
            "rad": (compute_eec_pairs(det_crad, det_scale),
                    compute_eec_pairs(part_crad, part_scale)),
            "AA":  (compute_eec_pairs(det_cA, det_scale),
                    compute_eec_pairs(part_cA, part_scale)),
            "BB":  (compute_eec_pairs(det_cB, det_scale),
                    compute_eec_pairs(part_cB, part_scale)),
            "AB":  (compute_eec_pairs(det_cA, det_scale, det_cB),
                    compute_eec_pairs(part_cA, part_scale, part_cB)),
        }

        # ---- pair-level match and fill 6D ----
        for lab, (det_pairs, part_pairs) in eec_sets.items():
            
            matched, tr_unmatched, det_unmatched = match_eec_pairs(det_pairs, part_pairs)

            # Fill unmatched pairs
            fill_correlator(det_pairs, hists_allpt_det_all[f"{lab}_wwradpt"])
            fill_correlator(part_pairs, hists_allpt_part_all[f"{lab}_wwradpt"])

            # Fill matched pairs
            det_pairs_matched  = [d for d, p in matched]
            part_pairs_matched = [p for d, p in matched]
            fill_correlator(det_pairs_matched, hists_allpt_det_matched[f"{lab}_wwradpt"])
            fill_correlator(part_pairs_matched, hists_allpt_part_matched[f"{lab}_wwradpt"])


            # Pick out and fill the histograms for that pt slice
            if det_ptslice != ():
                h_det_all = hists_det_all[det_ptslice]
                h_det_matched = hists_det_matched[det_ptslice]
                fill_correlator(det_pairs, h_det_all[f"{lab}_wwradpt"])
                fill_correlator(det_pairs_matched, h_det_matched[f"{lab}_wwradpt"])

            if part_ptslice != ():
                h_part_all = hists_part_all[part_ptslice]
                h_part_matched = hists_part_matched[part_ptslice]
                fill_correlator(part_pairs, h_part_all[f"{lab}_wwradpt"])
                fill_correlator(part_pairs_matched, h_part_matched[f"{lab}_wwradpt"])



    print("Filled pairs.")




    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    fout = ROOT.TFile(args.output, "RECREATE")
    fout.cd()

    hNevents.Write()


    hJetPtALL_det_matched_ungroomed.Write()
    hJetPtALL_det_all_ungroomed.Write()
    hJetPtALL_part_matched_ungroomed.Write()
    hJetPtALL_part_all_ungroomed.Write()

    hJetPtALL_det_matched_groomed.Write()
    hJetPtALL_det_all_groomed.Write()
    hJetPtALL_part_matched_groomed.Write()
    hJetPtALL_part_all_groomed.Write()

    for h in inclusiveQA.values():
        h.Write()
    
    # inclusiveQA["allpt"].Write()
    for h in hists_allpt_det_matched.values():
        h.Write()
    for h in hists_allpt_det_all.values():
        h.Write()
    for h in hists_allpt_part_matched.values():
        h.Write()
    for h in hists_allpt_part_all.values():
        h.Write()
        
    for sl in PT_SLICES:
        hists = hists_det_matched[sl]
        for h in hists.values():
            h.Write()
        hists = hists_det_all[sl]
        for h in hists.values():
            h.Write()
        hists = hists_part_matched[sl]
        for h in hists.values():
            h.Write()
        hists = hists_part_all[sl]
        for h in hists.values():
            h.Write()
        
        # inclusiveQA[(lo, hi)].Write()
    
    fout.Close()


if __name__ == "__main__":
    main()