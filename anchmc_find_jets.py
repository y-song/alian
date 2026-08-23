#!/usr/bin/env python3
"""
Read eventTree from a ROOT file, cluster detector-level and particle-level
anti-kt R=0.4 jets, match them, apply pT/pThat outlier rejection, save
histograms to a ROOT file, and save a per-jet table (with constituents)
to a Parquet file.

Dependencies:
    pip install uproot awkward numpy fastjet pyarrow
"""

import argparse
import numpy as np
import uproot
import awkward as ak
import fastjet
import ROOT
import pyarrow as pa
import pyarrow.parquet as pq

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JET_R = 0.4
JET_ETA_MAX = 0.9 - JET_R          # fiducial jet |eta| acceptance
JET_PT_MIN = 8.0                   # minimum jet pt to keep (GeV) -- tune as needed
PTHAT_OUTLIER_FACTOR = 4.0         # reject events with pt_jet / pThat > this
MATCH_DR_MAX = 0.6 * JET_R         # geometric matching radius (dR)

PION_MASS = 0.13957                # assumed constituent mass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---- In script 1, modify make_pseudojets to carry the label as user_index ----

def make_pseudojets(pt, eta, phi, label, mass=PION_MASS):
    """Build PseudoJets; store the particle's label in user_index so it
    survives clustering and can be recovered from jet constituents."""
    pjets = []
    for i in range(len(pt)):
        pj = fastjet.PseudoJet()
        pj.reset_PtYPhiM(float(pt[i]), float(eta[i]), float(phi[i]), mass)
        pj.set_user_index(int(label[i]))   # <-- store LABEL, not position index
        pjets.append(pj)
    return pjets


def cluster_jets(pt, eta, phi, label, jet_def, pt_min, eta_max, highjetcut=False):
    consts = make_pseudojets(pt, eta, phi, label)
    if len(consts) == 0:
        return []
    cs = fastjet.ClusterSequence(consts, jet_def)
    jets = fastjet.sorted_by_pt(cs.inclusive_jets(pt_min))
    out = []
    for j in jets:
        j_eta = j.eta()
        if abs(j_eta) >= eta_max:
            continue
        highpttrack = False
        if highjetcut:
            for c in j.constituents():
                if c.pt() > 100:
                    highpttrack = True
                    break
        if highpttrack:
            continue
        cparts = j.constituents()
        out.append({
            "pt": j.pt(), "eta": j_eta, "phi": j.phi_std(),
            "nconst": len(cparts),
            "const_pt":    [c.pt() for c in cparts],
            "const_eta":   [c.eta() for c in cparts],
            "const_phi":   [c.phi_std() for c in cparts],
            "const_label": [c.user_index() for c in cparts],   # <-- NEW
        })
    return out


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta * deta + dphi * dphi)


def match_jets(det_jets, part_jets, dr_max):
    """
    Greedy one-to-one dR matching, starting from highest-pT detector jets.
    Returns a list of (det_index, part_index) tuples.
    """
    matches = []
    used_part = set()
    for id_, det in enumerate(det_jets):
        best_idx = -1
        best_dr = dr_max
        for k, part in enumerate(part_jets):
            if k in used_part:
                continue
            dr = delta_r(det["eta"], det["phi"], part["eta"], part["phi"])
            if dr < best_dr:
                best_dr = dr
                best_idx = k
        if best_idx >= 0:
            used_part.add(best_idx)
            matches.append((id_, best_idx))
    return matches


# ---------------------------------------------------------------------------
# Histogram container
# ---------------------------------------------------------------------------
class Hist1D:
    def __init__(self, name, title, nbins, xlow, xhigh):
        self.name = name
        self.title = title
        self.nbins = nbins
        self.xlow = xlow
        self.xhigh = xhigh
        self.edges = np.linspace(xlow, xhigh, nbins + 1)
        self.counts = np.zeros(nbins, dtype=np.float64)

    def fill(self, value, weight=1.0):
        if value < self.xlow or value >= self.xhigh:
            return
        idx = int((value - self.xlow) / (self.xhigh - self.xlow) * self.nbins)
        if 0 <= idx < self.nbins:
            self.counts[idx] += weight

    def fill_array(self, values, weight=1.0):
        for v in values:
            self.fill(v, weight)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input ROOT file containing eventTree")
    ap.add_argument("-o", "--output", default="jets_out.root",
                    help="output ROOT file (histograms)")
    ap.add_argument("-p", "--parquet", default="jets_out.parquet",
                    help="output Parquet file (per-jet table)")
    ap.add_argument("-t", "--tree", default="eventTree", help="tree name")
    ap.add_argument("-n", "--nevents", type=int, default=-1,
                    help="number of events to process (-1 = all)")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Histograms
    # -----------------------------------------------------------------------
    h = {
        "det_track_pt":  Hist1D("det_track_pt",  "Detector track p_{T};p_{T} (GeV);counts", 100, 0, 100),
        "det_track_eta": Hist1D("det_track_eta", "Detector track #eta;#eta;counts",         60, -1.5, 1.5),
        "det_track_phi": Hist1D("det_track_phi", "Detector track #phi;#phi;counts",         64, -np.pi, np.pi),
        "mc_part_pt":    Hist1D("mc_part_pt",    "MC particle p_{T};p_{T} (GeV);counts",    100, 0, 100),
        "mc_part_eta":   Hist1D("mc_part_eta",   "MC particle #eta;#eta;counts",             60, -1.5, 1.5),
        "mc_part_phi":   Hist1D("mc_part_phi",   "MC particle #phi;#phi;counts",             64, -np.pi, np.pi),
        "det_jet_pt":    Hist1D("det_jet_pt",    "Detector jet p_{T};p_{T} (GeV);counts",   100, 0, 200),
        "det_jet_eta":   Hist1D("det_jet_eta",   "Detector jet #eta;#eta;counts",            60, -1.0, 1.0),
        "det_jet_phi":   Hist1D("det_jet_phi",   "Detector jet #phi;#phi;counts",            64, 0, 2*np.pi),
        "part_jet_pt":   Hist1D("part_jet_pt",   "Particle jet p_{T};p_{T} (GeV);counts",   100, 0, 200),
        "part_jet_eta":  Hist1D("part_jet_eta",  "Particle jet #eta;#eta;counts",            60, -1.0, 1.0),
        "part_jet_phi":  Hist1D("part_jet_phi",  "Particle jet #phi;#phi;counts",            64, 0, 2*np.pi),
        "matched_det_pt":  Hist1D("matched_det_pt",  "Matched det jet p_{T};p_{T} (GeV);counts",  100, 0, 200),
        "matched_part_pt": Hist1D("matched_part_pt", "Matched part jet p_{T};p_{T} (GeV);counts", 100, 0, 200),
    }
    resp_bins = np.linspace(0, 200, 101)
    resp = np.zeros((len(resp_bins) - 1, len(resp_bins) - 1), dtype=np.float64)

    # -----------------------------------------------------------------------
    # Parquet accumulator: one row per jet.
    # We store constituents as variable-length (list) columns so awkward /
    # pyarrow can read them back as jagged arrays.
    # -----------------------------------------------------------------------
    rows = {
        "event": [],          # event index (running counter)
        "level": [],          # "det" or "part"
        "jet_index": [],      # index of jet within its level for this event
        "jet_pt": [],
        "jet_eta": [],
        "jet_phi": [],
        "nconst": [],
        "const_pt": [],       # list<float>
        "const_eta": [],      # list<float>
        "const_phi": [],      # list<float>
        "is_matched": [],     # bool
        "match_index": [],    # index of partner jet at the other level (-1 if none)
        "match_pt": [],       # pt of matched partner (-1 if none)
        "match_dr": [],       # dR to matched partner (-1 if none)
        "mc_weight": [],
        "mc_pThat": [],
        "const_label": [],    
    }

    def add_jet_row(event, level, jidx, jet, is_matched,
                    match_index, match_pt, match_dr, weight, pthat):
        rows["event"].append(event)
        rows["level"].append(level)
        rows["jet_index"].append(jidx)
        rows["jet_pt"].append(jet["pt"])
        rows["jet_eta"].append(jet["eta"])
        rows["jet_phi"].append(jet["phi"])
        rows["nconst"].append(jet["nconst"])
        rows["const_pt"].append(jet["const_pt"])
        rows["const_eta"].append(jet["const_eta"])
        rows["const_phi"].append(jet["const_phi"])
        rows["is_matched"].append(is_matched)
        rows["match_index"].append(match_index)
        rows["match_pt"].append(match_pt)
        rows["match_dr"].append(match_dr)
        rows["mc_weight"].append(weight)
        rows["mc_pThat"].append(pthat)
        rows["const_label"].append(jet["const_label"])
        

    # -----------------------------------------------------------------------
    # anti-kt R=0.4
    # -----------------------------------------------------------------------
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_R)

    branches = [
        "mc_weight", "mc_pThat", "event_selection",
        "track_data_pt", "track_data_eta", "track_data_phi",
        "track_data_mclabel", "track_data_tracksel",
        "mc_particle_pt", "mc_particle_eta", "mc_particle_phi",
        "mc_particle_partID",
    ]

    SEL8_BIT = 0
    GLOBAL_TRACK_BIT = 1        

    tree_path = f"{args.input}:{args.tree}"
    n_processed = 0
    n_rejected = 0

    for chunk in uproot.iterate(tree_path, branches, library="ak", step_size=10000):
        n_in_chunk = len(chunk["mc_pThat"])
        for ie in range(n_in_chunk):
            if args.nevents >= 0 and n_processed >= args.nevents:
                break

            weight = float(chunk["mc_weight"][ie])
            pthat = float(chunk["mc_pThat"][ie])

            # event selection (sel8)
            evsel = int(chunk["event_selection"][ie])
            if not (evsel & (1 << SEL8_BIT)):
                n_processed += 1
                continue
    
            # --- detector tracks with selection ---
            d_pt  = ak.to_numpy(chunk["track_data_pt"][ie])
            d_eta = ak.to_numpy(chunk["track_data_eta"][ie])
            d_phi = ak.to_numpy(chunk["track_data_phi"][ie])
            d_lab = ak.to_numpy(chunk["track_data_mclabel"][ie])
            d_sel = ak.to_numpy(chunk["track_data_tracksel"][ie]).astype(np.uint32)
            dmask = (d_pt >= 0.15) & (np.abs(d_eta) <= 0.9) & ((d_sel & (1 << GLOBAL_TRACK_BIT)) != 0)
            d_pt, d_eta, d_phi, d_lab = d_pt[dmask], d_eta[dmask], d_phi[dmask], d_lab[dmask]
    
            # --- MC particles with selection ---
            p_pt  = ak.to_numpy(chunk["mc_particle_pt"][ie])
            p_eta = ak.to_numpy(chunk["mc_particle_eta"][ie])
            p_phi = ak.to_numpy(chunk["mc_particle_phi"][ie])
            p_id  = ak.to_numpy(chunk["mc_particle_partID"][ie])
            pmask = (p_pt >= 0.15) & (np.abs(p_eta) <= 0.9)
            p_pt, p_eta, p_phi, p_id = p_pt[pmask], p_eta[pmask], p_phi[pmask], p_id[pmask]
    
            det_jets  = cluster_jets(d_pt, d_eta, d_phi, d_lab, jet_def, JET_PT_MIN, JET_ETA_MAX, highjetcut=True)
            part_jets = cluster_jets(p_pt, p_eta, p_phi, p_id, jet_def, JET_PT_MIN, JET_ETA_MAX)

            # ---- pt,jet / pThat outlier rejection (event-level veto) ----
            all_jet_pts = [j["pt"] for j in det_jets] + [j["pt"] for j in part_jets]
            if pthat > 0 and len(all_jet_pts) > 0:
                if max(all_jet_pts) / pthat > PTHAT_OUTLIER_FACTOR:
                    n_rejected += 1
                    n_processed += 1
                    continue

            # ---- constituent histograms ----
            h["det_track_pt"].fill_array(d_pt, weight)
            h["det_track_eta"].fill_array(d_eta, weight)
            h["det_track_phi"].fill_array(d_phi, weight)
            h["mc_part_pt"].fill_array(p_pt, weight)
            h["mc_part_eta"].fill_array(p_eta, weight)
            h["mc_part_phi"].fill_array(p_phi, weight)

            # ---- jet histograms ----
            for j in det_jets:
                h["det_jet_pt"].fill(j["pt"], weight)
                h["det_jet_eta"].fill(j["eta"], weight)
                h["det_jet_phi"].fill(j["phi"], weight)
            for j in part_jets:
                h["part_jet_pt"].fill(j["pt"], weight)
                h["part_jet_eta"].fill(j["eta"], weight)
                h["part_jet_phi"].fill(j["phi"], weight)

            # ---- matching ----
            matches = match_jets(det_jets, part_jets, MATCH_DR_MAX)
            # build lookup maps: det_idx -> part_idx and vice versa
            det_to_part = {}
            part_to_det = {}
            for id_, ip_ in matches:
                det_to_part[id_] = ip_
                part_to_det[ip_] = id_
                det, part = det_jets[id_], part_jets[ip_]
                h["matched_det_pt"].fill(det["pt"], weight)
                h["matched_part_pt"].fill(part["pt"], weight)
                ix = np.searchsorted(resp_bins, part["pt"], side="right") - 1
                iy = np.searchsorted(resp_bins, det["pt"], side="right") - 1
                if 0 <= ix < resp.shape[0] and 0 <= iy < resp.shape[1]:
                    resp[ix, iy] += weight

            # ---- fill parquet rows: detector jets ----
            for id_, det in enumerate(det_jets):
                if id_ in det_to_part:
                    ip_ = det_to_part[id_]
                    part = part_jets[ip_]
                    dr = delta_r(det["eta"], det["phi"], part["eta"], part["phi"])
                    add_jet_row(n_processed, "det", id_, det, True,
                                ip_, part["pt"], dr, weight, pthat)
                else:
                    add_jet_row(n_processed, "det", id_, det, False,
                                -1, -1.0, -1.0, weight, pthat)

            # ---- fill parquet rows: particle jets ----
            for ip_, part in enumerate(part_jets):
                if ip_ in part_to_det:
                    id_ = part_to_det[ip_]
                    det = det_jets[id_]
                    dr = delta_r(det["eta"], det["phi"], part["eta"], part["phi"])
                    add_jet_row(n_processed, "part", ip_, part, True,
                                id_, det["pt"], dr, weight, pthat)
                else:
                    add_jet_row(n_processed, "part", ip_, part, False,
                                -1, -1.0, -1.0, weight, pthat)

            n_processed += 1

        if args.nevents >= 0 and n_processed >= args.nevents:
            break

    print(f"Processed {n_processed} events, rejected {n_rejected} as pThat outliers.")

    # -----------------------------------------------------------------------
    # Write histograms
    # -----------------------------------------------------------------------
    with uproot.recreate(args.output) as fout:
        for hist in h.values():
            fout[hist.name] = (hist.counts, hist.edges)
        fout["response_jetpt_part_vs_det"] = (resp, resp_bins, resp_bins)
    print(f"Wrote histograms to {args.output}")

    # -----------------------------------------------------------------------
    # Write per-jet table to Parquet.
    # Using an awkward array preserves the jagged constituent lists cleanly,
    # then convert to an arrow table via ak.to_arrow_table.
    # -----------------------------------------------------------------------
    jets_ak = ak.Array({
        "event":       np.asarray(rows["event"], dtype=np.int64),
        "level":       rows["level"],
        "jet_index":   np.asarray(rows["jet_index"], dtype=np.int32),
        "jet_pt":      np.asarray(rows["jet_pt"], dtype=np.float32),
        "jet_eta":     np.asarray(rows["jet_eta"], dtype=np.float32),
        "jet_phi":     np.asarray(rows["jet_phi"], dtype=np.float32),
        "nconst":      np.asarray(rows["nconst"], dtype=np.int32),
        "const_pt":    ak.Array(rows["const_pt"]),
        "const_eta":   ak.Array(rows["const_eta"]),
        "const_phi":   ak.Array(rows["const_phi"]),
        "is_matched":  np.asarray(rows["is_matched"], dtype=bool),
        "match_index": np.asarray(rows["match_index"], dtype=np.int32),
        "match_pt":    np.asarray(rows["match_pt"], dtype=np.float32),
        "match_dr":    np.asarray(rows["match_dr"], dtype=np.float32),
        "mc_weight":   np.asarray(rows["mc_weight"], dtype=np.float64),
        "mc_pThat":    np.asarray(rows["mc_pThat"], dtype=np.float32),
        "const_label": ak.Array(rows["const_label"]),
    })

    arrow_table = ak.to_arrow_table(jets_ak)
    pq.write_table(arrow_table, args.parquet)
    print(f"Wrote {len(jets_ak)} jets to {args.parquet}")


if __name__ == "__main__":
    main()