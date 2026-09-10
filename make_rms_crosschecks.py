'''
Cross-check script for energy weights in the AxB subjet case.
Focuses on pairs with weights in the range [0.22, 0.25].
'''

#!/usr/bin/env python3
"""
Build cross-check histograms from the per-jet parquet produced by script 1.

For each matched det/part jet pair:
  - recluster constituents (C/A), run LundGenerator, find first SD zcut split
  - get radiator + subjet A, subjet B
  - filter for AB energy weights in [0.22, 0.25]
  - fill plots for particle pTs, jet constituents, jet pTs, PID, deltaR, and mass.

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

# Weight filter range
W_MIN_FILTER = 0.22
W_MAX_FILTER = 0.25

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
    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0) # large R to catch all
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


def compute_eec_pairs_indices(c_select, scale, c_select_B=None):
    """
    Return a list of pairs: (RL, weight, index_i, index_j)
    Where index_i indices into c_select and index_j indices into c_select_B (for AB).
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
            pairs.append((rs[k], ws[k], int(idx1[k]), int(idx2[k])))
    else:
        # entries alternate AB, BA, AB, BA, ...
        for k in range(n):
            i = int(idx1[k])
            j = int(idx2[k])
            if k % 2 == 0:
                # AB entry: idx1 -> A, idx2 -> B
                pairs.append((rs[k], ws[k], i, j))
            else:
                # BA entry: idx1 -> B, idx2 -> A
                pairs.append((rs[k], ws[k], j, i))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input parquet file (from script 1)")
    ap.add_argument("-o", "--output", default="crosschecks.root", help="output ROOT file")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Output histograms
    # -----------------------------------------------------------------------
    hNevents = ROOT.TH1D("hNevents", "Number of Events", 1, 0, 1)

    # We will create these for both det and part if available
    # Use a dictionary to store by version
    VERSIONS = ["det", "part"]

    h_pair_pt = {v: ROOT.TH2D(f"h_pair_pt_{v}", f"Pair p_{{T}}s {v};p_{{T,i}};p_{{T,j}}", 200, 0, 200, 200, 0, 200) for v in VERSIONS}
    h_n_const_orig = {v: ROOT.TH1D(f"h_n_const_orig_{v}", f"Original constituents {v};N", 100, 0, 100) for v in VERSIONS}
    h_n_const_groomed = {v: ROOT.TH1D(f"h_n_const_groomed_{v}", f"Groomed constituents {v};N", 100, 0, 100) for v in VERSIONS}
    h_n_const_removed = {v: ROOT.TH1D(f"h_n_const_removed_{v}", f"Removed constituents {v};N", 100, 0, 100) for v in VERSIONS}
    h_jet_pt_orig = {v: ROOT.TH1D(f"h_jet_pt_orig_{v}", f"Original jet p_{{T}} {v};p_{{T}}", 200, 0, 200) for v in VERSIONS}
    h_jet_pt_groomed = {v: ROOT.TH1D(f"h_jet_pt_groomed_{v}", f"Groomed jet p_{{T}} {v};p_{{T}}", 200, 0, 200) for v in VERSIONS}
    h_pair_pid = {v: ROOT.TH2D(f"h_pair_pid_{v}", f"Pair PID {v};abs(PID_{{i}});abs(PID_{{j}})", 401, 0, 401, 401, 0, 401) for v in VERSIONS}
    h_pair_dr = {v: ROOT.TH1D(f"h_pair_dr_{v}", f"Pair #Delta R {v};#Delta R", 100, 0, 1.0) for v in VERSIONS}
    h_pair_mass = {v: ROOT.TH1D(f"h_pair_mass_{v}", f"Pair Invariant Mass {v};Mass [GeV]", 100, 0, 10) for v in VERSIONS}

    
    # -----------------------------------------------------------------------
    # Read parquet
    # -----------------------------------------------------------------------
    jets = ak.from_parquet(args.input)

    if len(jets) == 0:
        print(f"Warning: File '{args.input}' contains 0 jet entries. Exiting.")
        return

    # We need matched det/part jet pairs.
    events = np.asarray(jets.event)
    levels = np.asarray(jets.level)
    jidx   = np.asarray(jets.jet_index)

    row_of = {}
    for r in range(len(jets)):
        row_of[(int(events[r]), str(levels[r]), int(jidx[r]))] = r

    jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)

    # Process each event once
    unique_events = np.unique(events)
    for ev in unique_events:
        hNevents.Fill(0)

        # Find det jet for this event that is matched
        # Note: Simplified to first matched det jet per event for cross-check
        det_row = None
        for r in range(len(jets)):
            if events[r] == ev and levels[r] == "det" and bool(jets[r].is_matched):
                det_row = r
                break

        if det_row is None:
            continue

        det = jets[det_row]
        part_jidx = int(det.match_index)
        pr = row_of.get((int(ev), "part", part_jidx))
        if pr is None:
            continue
        part = jets[pr]
        mc_weight = float(det.mc_weight)

        # Process both levels
        for v, jet_data in [("det", det), ("part", part)]:
            # 1. Build constituents
            consts = build_pseudojets(jet_data.const_pt, jet_data.const_eta, jet_data.const_phi, jet_data.const_label)

            # 2. Recluster C/A
            jet, cs = recluster_ca(consts)
            if jet is None: continue

            # 3. SD Split
            lund_gen = fjcontrib.LundGenerator(jet_def_ca)
            jet_lund = lund_gen.result(jet)
            split = select_split_sd(jet_lund, SD_ZCUT)
            if split is None: continue

            rad = split.pair()
            sub = sorted(rad.pieces(), key=lambda x: x.pt(), reverse=True)
            if len(sub) != 2: continue
            sjA, sjB = sub

            # 4. Energy weight cross-check (AxB)
            cA = selected_constituents(sjA, TRK_THRD)
            cB = selected_constituents(sjB, TRK_THRD)

            # Using radiator pt as scale for weights
            pairs = compute_eec_pairs_indices(cA, rad.perp(), cB)

            for rl, w, idxA, idxB in pairs:
                if W_MIN_FILTER <= w <= W_MAX_FILTER:
                    # We found a pair in the target weight range
                    pA = cA[idxA]
                    pB = cB[idxB]

                    # Particle pTs
                    h_pair_pt[v].Fill(pA.perp(), pB.perp(), mc_weight)

                    # Jet info
                    # Original constituents = everything that went into the jet (filtered by threshold if desired,
                    # but let's use the full list as provided in parquet/recluster)
                    n_orig = len(consts)
                    # Groomed constituents = everything in the radiator
                    crad = selected_constituents(rad, TRK_THRD)
                    n_groomed = len(crad)

                    h_n_const_orig[v].Fill(n_orig, mc_weight)
                    h_n_const_groomed[v].Fill(n_groomed, mc_weight)
                    h_n_const_removed[v].Fill(n_orig - n_groomed, mc_weight)

                    h_jet_pt_orig[v].Fill(jet.perp(), mc_weight)
                    h_jet_pt_groomed[v].Fill(rad.perp(), mc_weight)

                    # PID information
                    # We need to get the original index of the particle to look up PID in the parquet
                    # Since we use user_index = label from build_pseudojets, we can't easily map
                    # back to the original array index unless we store it.
                    # However, if we assume labels are consistent, we could use them.
                    # Better: find the particle in the original array.

                    # Let's try to find PID from the parquet if available
                    if "const_pid" in jet_data.fields:
                        # we need the index in the original const_pt array.
                        # build_pseudojets just iterates, so the i-th PJ in v is the i-th in the input.
                        # But cA is a subset of the radiator's constituents.
                        # The radiator's constituents are a subset of the original jet's.

                        # To get the correct PID, we should find the particle with the same user_index
                        # in the original const_label array.
                        labelA = pA.user_index()
                        labelB = pB.user_index()

                        # Find indices in the parquet array where label == user_index
                        # This is slow in a loop, but since we only do it for w in [0.22, 0.25], it's okay.
                        idxA_orig = np.where(jet_data.const_label == labelA)[0]
                        idxB_orig = np.where(jet_data.const_label == labelB)[0]

                        if len(idxA_orig) > 0 and len(idxB_orig) > 0:
                            pidA_abs = abs(int(jet_data.const_pid[idxA_orig[0]]))
                            pidB_abs = abs(int(jet_data.const_pid[idxB_orig[0]]))

                            # Clip values > 400 to 400 to capture protons (2212) and others at the edge
                            valA = pidA_abs if pidA_abs <= 400 else 400
                            valB = pidB_abs if pidB_abs <= 400 else 400
                            h_pair_pid[v].Fill(float(valA), float(valB), mc_weight)

                    # Angular separation and mass
                    dr = pA.delta_R(pB)
                    h_pair_dr[v].Fill(dr, mc_weight)

                    # Invariant mass
                    pA_vec = fj.LorentzVector(pA.perp(), pA.rapidity(), pA.phi(), PION_MASS)
                    pB_vec = fj.LorentzVector(pB.perp(), pB.rapidity(), pB.phi(), PION_MASS)
                    mass = (pA_vec + pB_vec).mass()
                    h_pair_mass[v].Fill(mass, mc_weight)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    fout = ROOT.TFile(args.output, "RECREATE")
    fout.cd()

    hNevents.Write()
    for v in VERSIONS:
        h_pair_pt[v].Write()
        h_n_const_orig[v].Write()
        h_n_const_groomed[v].Write()
        h_n_const_removed[v].Write()
        h_jet_pt_orig[v].Write()
        h_jet_pt_groomed[v].Write()
        h_pair_pid[v].Write()
        h_pair_dr[v].Write()
        h_pair_mass[v].Write()

    fout.Close()
    print(f"Wrote cross-check histograms to {args.output}")

if __name__ == "__main__":
    main()
