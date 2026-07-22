#!/usr/bin/env python
import sys
import numpy as np
import uproot
import awkward as ak
import fastjet as fj
import pandas as pd
import argparse

# --- Selection bits ---
SEL8_BIT = 0
GLOBAL_TRACK_BIT = 1

# --- Kinematic cuts ---
PT_MIN  = 0.15   # GeV/c (150 MeV)
ETA_MAX = 0.9    # |eta| < 0.9

# --- Jet settings ---
JET_R       = 0.4
JET_PT_MIN  = 8.0                # GeV/c, minimum jet pt to save
JET_ETA_MAX = ETA_MAX - JET_R    # fiducial acceptance: |eta_jet| < 0.5

# --- RCT (Run Condition Table) flag definitions ---
rctdict = {
    "kCPVBad":            0,
    "kEMCBad":            1,
    "kEMCLimAccMCRepr":   2,
    "kFDDBad":            3,
    "kFT0Bad":            4,
    "kFV0Bad":            5,
    "kHMPBad":            6,
    "kITSBad":            7,
    "kITSLimAccMCRepr":   8,
    "kMCHBad":            9,
    "kMCHLimAccMCRepr":  10,
    "kMFTBad":           11,
    "kMFTLimAccMCRepr":  12,
    "kMIDBad":           13,
    "kMIDLimAccMCRepr":  14,
    "kPHSBad":           15,
    "kTOFBad":           16,
    "kTOFLimAccMCRepr":  17,
    "kTPCBadTracking":   18,
    "kTPCBadPID":        19,
    "kTPCLimAccMCRepr":  20,
    "kTRDBad":           21,
    "kZDCBad":           22,
    "kNRCTSelectionFlags": 23,
    "kDummy24":          24,
    "kDummy25":          25,
    "kDummy26":          26,
    "kDummy27":          27,
    "kDummy28":          28,
    "kDummy29":          29,
    "kDummy30":          30,
    "kCcdbObjectLoaded": 31,
}

def make_rct_mask(flags):
    """OR together the bits for the given list of flag names."""
    mask = 0
    for f in flags:
        mask |= (1 << rctdict[f])
    return mask

# CBT selection: event is BAD if ANY of these bits are set.
CBT_FLAGS = ["kFT0Bad", "kITSBad", "kTPCBadTracking", "kTPCBadPID"]
RCT_MASK_CBT = make_rct_mask(CBT_FLAGS)


def process(infile, outfile, jet_pt_min=JET_PT_MIN):

    # anti-kt jet definition
    jetdef = fj.JetDefinition(fj.antikt_algorithm, JET_R)

    # Bookkeeping
    n_events_total = 0
    n_events_sel   = 0
    n_events_rct   = 0
    n_tracks_sel   = 0
    n_jets_sel     = 0

    # Running indices over the whole file
    global_event_id = 0
    global_jet_id   = 0

    # Output accumulators (one row per constituent)
    rec_event_id   = []
    rec_jet_id     = []
    rec_run_number = []
    rec_jet_pt     = []
    rec_jet_eta    = []
    rec_jet_phi    = []
    rec_c_pt       = []
    rec_c_eta      = []
    rec_c_phi      = []

    for batch in uproot.iterate(
        f"{infile}:eventTree",
        ["run_number", "multiplicity", "centrality",
         "event_sel", "rct", "track_pt", "track_eta", "track_phi", "track_sel"],
        step_size="200 MB",
        library="ak",
    ):
        n_batch = len(batch)
        n_events_total += n_batch

        # --- Event selection: sel8 AND rct/CBT good ---
        em = (batch["event_sel"] & (1 << SEL8_BIT)) != 0
        n_events_sel += int(ak.sum(em))
        rct_good = (batch["rct"] & RCT_MASK_CBT) == 0
        em = em & rct_good
        n_events_rct += int(ak.sum(em))

        # event_id is a running index over ALL events in the file (before selection)
        ev_ids_all = np.arange(global_event_id, global_event_id + n_batch)
        global_event_id += n_batch

        em_np = ak.to_numpy(em)
        b = batch[em]
        ev_ids = ev_ids_all[em_np]
        if len(b) == 0:
            continue

        run_all = ak.to_numpy(b["run_number"])

        # --- Track selection: global track AND pt>=150 MeV AND |eta|<=0.9 ---
        tm = (
            ((b["track_sel"] & (1 << GLOBAL_TRACK_BIT)) != 0)
            & (b["track_pt"] >= PT_MIN)
            & (abs(b["track_eta"]) <= ETA_MAX)
        )

        sel_pt  = b["track_pt"][tm]
        sel_eta = b["track_eta"][tm]
        sel_phi = b["track_phi"][tm]

        n_tracks_sel += int(ak.sum(ak.num(sel_pt)))

        # --- Jet clustering, per event ---
        n_ev = len(sel_pt)
        for iev in range(n_ev):

            if iev % 10000 == 0:
                print(f"Clustering event {iev}/{n_ev}")

            ev_pt  = ak.to_numpy(sel_pt[iev])
            ev_eta = ak.to_numpy(sel_eta[iev])
            ev_phi = ak.to_numpy(sel_phi[iev])

            if len(ev_pt) == 0:
                continue

            # Massless four-vectors
            px = ev_pt * np.cos(ev_phi)
            py = ev_pt * np.sin(ev_phi)
            pz = ev_pt * np.sinh(ev_eta)
            E  = np.sqrt(px**2 + py**2 + pz**2)

            pj_particles = [
                fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
                for i in range(len(ev_pt))
            ]

            cluster = fj.ClusterSequence(pj_particles, jetdef)
            jets = fj.sorted_by_pt(cluster.inclusive_jets(jet_pt_min))

            for jet in jets:
                jeta = jet.eta()
                # fiducial cut on the anti-kt jet
                if abs(jeta) > JET_ETA_MAX:
                    continue

                constits = jet.constituents()
                # nc = len(constits)

                for c in constits:
                    rec_event_id.append(int(ev_ids[iev]))
                    rec_jet_id.append(global_jet_id)
                    rec_run_number.append(int(run_all[iev]))
                    rec_jet_pt.append(float(jet.pt()))
                    rec_jet_eta.append(float(jeta))
                    rec_jet_phi.append(float(jet.phi()))
                    rec_c_pt.append(float(c.pt()))
                    rec_c_eta.append(float(c.eta()))
                    rec_c_phi.append(float(c.phi()))

                global_jet_id += 1
                n_jets_sel += 1

    # --- Build DataFrame and save (one row per constituent) ---
    df = pd.DataFrame({
        "event_id":   rec_event_id,
        "jet_id":     rec_jet_id,
        "run_number": rec_run_number,
        "jet_pt":     rec_jet_pt,
        "jet_eta":    rec_jet_eta,
        "jet_phi":    rec_jet_phi,
        "c_pt":       rec_c_pt,
        "c_eta":      rec_c_eta,
        "c_phi":      rec_c_phi,
    })

    df.to_parquet(outfile, engine="pyarrow", index=False)

    print(f"Done: {infile} -> {outfile}")
    print(f"  events total:                {n_events_total}")
    print(f"  events sel8:                 {n_events_sel}")
    print(f"  events sel8+rct:             {n_events_rct}")
    print(f"  global trks (pt>={PT_MIN}, |eta|<={ETA_MAX}): {n_tracks_sel}")
    print(f"  jets saved (anti-kt R={JET_R}, pt>={jet_pt_min}, |eta|<{JET_ETA_MAX:.2f}): {n_jets_sel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and save jets from ALICE data.")
    parser.add_argument("infile", help="Input ROOT file")
    parser.add_argument("outfile", help="Output Parquet file")
    parser.add_argument("--ptmin", type=float, default=JET_PT_MIN,
                        help=f"Minimum jet pT to save (default: {JET_PT_MIN})")

    args = parser.parse_args()
    process(args.infile, args.outfile, jet_pt_min=args.ptmin)