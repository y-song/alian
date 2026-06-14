#!/usr/bin/env python
import sys
import numpy as np
import uproot
import awkward as ak
import fastjet as fj
import ROOT
import array as array

SEL8_BIT = 0
GLOBAL_TRACK_BIT = 1

# Kinematic cuts
PT_MIN  = 0.15   # GeV/c (150 MeV)
ETA_MAX = 0.9    # |eta| < 0.9

# Jet settings
JET_R       = 0.4          # resolution parameter
JET_PT_MIN  = 5.0          # GeV/c, minimum jet pt
JET_ETA_MAX = ETA_MAX - JET_R   # fiducial acceptance: |eta_jet| < 0.9 - R = 0.5
JET_TRIG_PT = 8.0   # GeV/c, trigger threshold for the jet-triggered track spectrum

# --- RCT (Run Condition Table) flag definitions ---
# Bit positions taken directly from the RCTSelectionFlags enum.
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

# CBT selection: setFlags({kFT0Bad, kITSBad, kTPCBadTracking, kTPCBadPID})
# Event is BAD if ANY of these bits are set.
CBT_FLAGS = ["kFT0Bad", "kITSBad", "kTPCBadTracking", "kTPCBadPID"]
RCT_MASK_CBT = make_rct_mask(CBT_FLAGS)

def process(infile, outfile):

    bins_pt_QA = array.array('d', [0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                    1.0, 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5,
                                    10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                                    20., 25., 30., 35., 40., 45., 50.,
                                    60., 70., 80., 90., 100.])

    # Output histograms
    h_pt  = ROOT.TH1D("h_pt",  "global track p_{T};p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    # h_pt_ext  = ROOT.TH1D("h_pt_ext",  "global track p_{T};p_{T} [GeV/c];counts", len(bins_pt_QA)-1, bins_pt_QA)
    h_eta = ROOT.TH1D("h_eta", "global track #eta;#eta;counts",            80, -1.0, 1.0)
    h_phi = ROOT.TH1D("h_phi", "global track #phi;#phi;counts",            80, 0, 2*np.pi) #-np.pi, np.pi)

    # Jet histograms
    h_jet_pt  = ROOT.TH1D("h_jet_pt",  "jet p_{T};p_{T}^{jet} [GeV/c];counts", 1000, 0.0, 100.0)
    h_jet_eta = ROOT.TH1D("h_jet_eta", "jet #eta;#eta^{jet};counts",            80, -1.0, 1.0)
    h_jet_phi = ROOT.TH1D("h_jet_phi", "jet #phi;#phi^{jet};counts",            80, 0, 2*np.pi)
    h_jet_n   = ROOT.TH1D("h_jet_n",   "number of jets per event;N_{jets};events", 20, 0, 20)

    h_trackpt_jetpt = ROOT.TH2D("h_trackpt_jetpt", "track p_{T} vs jet p_{T};p_{T}^{jet} [GeV/c];p_{T}^{track} [GeV/c]", 200, 0, 20, 200, 0, 20)
    # h_trackpt_jetpt_ext = ROOT.TH2D("h_trackpt_jetpt_ext", "track p_{T} vs jet p_{T};p_{T}^{jet} [GeV/c];p_{T}^{track} [GeV/c]", 100, 0, 100, 100, 0, 100)

    # Track pt spectrum for events containing a jet > JET_TRIG_PT
    h_pt_jettrig = ROOT.TH1D("h_pt_jettrig", "track p_{T} (events with jet > 8 GeV);p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    h_jet_n_jettrig = ROOT.TH1D("h_jet_n_jettrig", "number of selected jets per event (events with jet > 8 GeV);N_{jets};events", 20, 0, 20) #total selected jet multiplicity, but only for the triggered events (those with ≥1 jet above 8 GeV)
    h_jet_n_above_below_jettrig = ROOT.TH2D("h_jet_n_above_below", "jet multiplicity: above vs below 8 GeV;N_{jets}^{>8 GeV};N_{jets}^{<8 GeV}", 20, 0, 20, 20, 0, 20)
    

    for h in (h_pt, h_eta, h_phi, h_jet_pt, h_jet_eta, h_jet_phi, h_jet_n, h_trackpt_jetpt, h_pt_jettrig, h_jet_n_above_below_jettrig): #h_pt_ext, h_trackpt_jetpt_ext
        h.Sumw2()

    # Bookkeeping counters stored as a 1-bin histogram (mergeable with hadd)
    h_cuts = ROOT.TH1D("h_cuts", "cut flow;;events or tracks", 5, 0, 5)
    h_cuts.GetXaxis().SetBinLabel(1, "events_total")
    h_cuts.GetXaxis().SetBinLabel(2, "events_sel8")
    h_cuts.GetXaxis().SetBinLabel(3, "events_sel8_rct")
    h_cuts.GetXaxis().SetBinLabel(4, "tracks_global")
    h_cuts.GetXaxis().SetBinLabel(5, "jets_sel")

    n_events_total = 0
    n_events_sel   = 0
    n_events_rct   = 0
    n_tracks_sel   = 0
    n_jets_sel     = 0

    # anti-kt jet definition
    jetdef = fj.JetDefinition(fj.antikt_algorithm, JET_R)

    for batch in uproot.iterate(
        f"{infile}:eventTree",
        ["event_sel", "rct", "track_pt", "track_eta", "track_phi", "track_sel"],
        step_size="200 MB",
        library="ak",
    ):
        n_events_total += len(batch)

        em = (batch["event_sel"] & (1 << SEL8_BIT)) != 0
        n_events_sel += int(ak.sum(em))   # events passing sel8
        # RCT / CBT: event is GOOD only if none of the bad bits are set
        rct_good = (batch["rct"] & RCT_MASK_CBT) == 0
        em = em & rct_good
        n_events_rct += int(ak.sum(em))   # events passing sel8 AND rct

        b = batch[em]
        if len(b) == 0:
            continue

        # Combined track mask: global track AND pt >= 150 MeV AND |eta| <= 0.9
        tm = (
            ((b["track_sel"] & (1 << GLOBAL_TRACK_BIT)) != 0)
            & (b["track_pt"] >= PT_MIN)
            & (abs(b["track_eta"]) <= ETA_MAX)
        )

        # --- Jagged (per-event) selected tracks, kept for jet clustering ---
        sel_pt  = b["track_pt"][tm]
        sel_eta = b["track_eta"][tm]
        sel_phi = b["track_phi"][tm]

        # --- Track QA fills (flattened across events) ---
        pt  = ak.to_numpy(ak.flatten(sel_pt))
        eta = ak.to_numpy(ak.flatten(sel_eta))
        phi = ak.to_numpy(ak.flatten(sel_phi))
        n_tracks_sel += len(pt)

        # FillN is the fast vectorized fill in PyROOT
        if len(pt):
            w = np.ones(len(pt), dtype=np.float64)
            h_pt.FillN(len(pt),   pt.astype(np.float64),  w)
            # h_pt_ext.FillN(len(pt), pt.astype(np.float64),  w)
            h_eta.FillN(len(eta), eta.astype(np.float64), w)
            h_phi.FillN(len(phi), phi.astype(np.float64), w)

        # --- Jet clustering (per event, looping over events) ---
        n_ev = len(sel_pt)
        for iev in range(n_ev):

            if iev % 10000 == 0:
                print(f"Processing event {iev}/{n_ev}")

            ev_pt  = ak.to_numpy(sel_pt[iev])
            ev_eta = ak.to_numpy(sel_eta[iev])
            ev_phi = ak.to_numpy(sel_phi[iev])

            if len(ev_pt) == 0:
                continue

            # Build massless four-vectors for this event
            px = ev_pt * np.cos(ev_phi)
            py = ev_pt * np.sin(ev_phi)
            pz = ev_pt * np.sinh(ev_eta)
            E  = np.sqrt(px**2 + py**2 + pz**2)   # massless

            # FastJet wants a std::vector<PseudoJet> -> use a Python list
            pj_particles = [
                fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
                for i in range(len(ev_pt))
            ]

            cluster = fj.ClusterSequence(pj_particles, jetdef)
            jets = fj.sorted_by_pt(cluster.inclusive_jets(JET_PT_MIN))

            njet_ev = 0
            njet_above_jettrig = 0   # jets above JET_TRIG_PT in this event
            njet_below_jettrig = 0   # jets below (or equal to) JET_TRIG_PT
            max_jet_pt = 0.0   # highest accepted-jet pt in this event
            for jet in jets:
                # Fiducial acceptance cut on jet
                if abs(jet.eta()) > JET_ETA_MAX:
                    continue

                njet_ev += 1
                n_jets_sel += 1

                jpt  = jet.pt()
                jeta = jet.eta()
                jphi = jet.phi()   # FastJet phi() is in [0, 2pi)

                if jpt > max_jet_pt:
                    max_jet_pt = jpt
                if jpt > JET_TRIG_PT:
                    njet_above_jettrig += 1
                else:
                    njet_below_jettrig += 1

                h_jet_pt.Fill(jpt)
                h_jet_eta.Fill(jeta)
                h_jet_phi.Fill(jphi)

                # Constituent track pt vs parent jet pt
                for c in jet.constituents():
                    h_trackpt_jetpt.Fill(jpt, c.pt())
                    # h_trackpt_jetpt_ext.Fill(jpt, c.pt())

            h_jet_n.Fill(njet_ev)
            h_jet_n_above_below_jettrig.Fill(njet_above_jettrig, njet_below_jettrig)

            # Jet-triggered track spectrum: if this event has any accepted jet
            # above the trigger threshold, fill ALL selected tracks in the event.
            if max_jet_pt > JET_TRIG_PT:
                h_jet_n_jettrig.Fill(njet_ev)   # total selected jets in triggered events

                w_ev = np.ones(len(ev_pt), dtype=np.float64)
                h_pt_jettrig.FillN(len(ev_pt), ev_pt.astype(np.float64), w_ev)

    h_cuts.SetBinContent(1, n_events_total)
    h_cuts.SetBinContent(2, n_events_sel)
    h_cuts.SetBinContent(3, n_events_rct)
    h_cuts.SetBinContent(4, n_tracks_sel)
    h_cuts.SetBinContent(5, n_jets_sel)


    fout = ROOT.TFile(outfile, "RECREATE")
    h_pt.Write(); h_eta.Write(); h_phi.Write() #
    h_jet_pt.Write(); h_jet_eta.Write(); h_jet_phi.Write(); h_jet_n.Write()
    h_trackpt_jetpt.Write(); #h_trackpt_jetpt_ext.Write()
    h_pt_jettrig.Write(); h_jet_n_jettrig.Write(); h_jet_n_above_below_jettrig.Write()
    h_cuts.Write()
    fout.Close()

    print(f"Done: {infile}")
    print(f"  events total: {n_events_total}")
    print(f"  events sel8:  {n_events_sel}")
    print(f"  global trks (pt>={PT_MIN}, |eta|<={ETA_MAX}):  {n_tracks_sel}")
    print(f"  jets (anti-kt R={JET_R}, pt>={JET_PT_MIN}, |eta|<={JET_ETA_MAX:.2f}):  {n_jets_sel}")

if __name__ == "__main__":
    infile, outfile = sys.argv[1], sys.argv[2]
    process(infile, outfile)