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
JET_R_06    = 0.6          # resolution parameter for R=0.6 checks
JET_PT_MIN  = 5.0          # GeV/c, minimum jet pt
JET_ETA_MAX = ETA_MAX - JET_R   # fiducial acceptance: |eta_jet| < 0.9 - R = 0.5
JET_ETA_MAX_R06 = 0.9 - 0.6   # = 0.3, fiducial |eta_jet| cut for R=0.6
JET_TRIG_PT = 8.0   # GeV/c, trigger threshold for the jet-triggered track spectrum


# High-pt non-jet track settings
HIGHPT_TRACK_MIN = 10.0     # GeV/c, threshold for "high pt track not in a jet"
CONSTITUENT_MATCH_TOL = 1e-4  # eta/phi tolerance for matching a track to a jet constituent
N_EVENT_DISPLAYS = 6        # number of high-pt-non-jet events to draw

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


def delta_phi(phi1, phi2):
    """Signed delta phi wrapped to (-pi, pi)."""
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


def draw_event_display(iev_global, ev_eta, ev_phi, ev_pt, jets,
                       highpt_idx, outdir="event_displays"):
    """
    Draw an eta-phi event display:
      - all tracks as arrows whose length is proportional to track pt
      - jets drawn as circles of radius JET_R centered on (jet_eta, jet_phi)
      - high-pt non-jet tracks highlighted in red
    Saved as a PNG.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    c = ROOT.TCanvas(f"c_evdisp_{iev_global}", "event display", 800, 700)

    # Frame in (eta, phi). phi from fastjet is in [0, 2pi).
    frame = ROOT.TH2F(f"frame_{iev_global}",
                      f"Event {iev_global} display;#eta;#varphi",
                      100, -1.2, 1.2, 100, 0.0, 2 * np.pi)
    frame.SetStats(0)
    frame.Draw()

    # Scale arrow length by pt. Pick a scale so the highest-pt track is visible.
    max_pt = max(ev_pt) if len(ev_pt) else 1.0
    # Arrow length in eta-phi units; tune so longest arrow ~0.5 in plot coords.
    length_scale = 0.5 / max_pt if max_pt > 0 else 0.0

    keep = []  # keep python references so ROOT objects aren't garbage collected

    highpt_set = set(highpt_idx)

    for i in range(len(ev_pt)):
        eta0 = ev_eta[i]
        phi0 = ev_phi[i]
        # Draw arrow pointing in +phi direction with length proportional to pt.
        # (direction is purely cosmetic; magnitude encodes pt)
        dlen = ev_pt[i] * length_scale
        arrow = ROOT.TArrow(eta0, phi0, eta0, phi0 + dlen, 0.01, "|>")
        if i in highpt_set:
            arrow.SetLineColor(ROOT.kRed)
            arrow.SetLineWidth(3)
        else:
            arrow.SetLineColor(ROOT.kBlue)
            arrow.SetLineWidth(1)
        arrow.Draw()
        keep.append(arrow)

    # Draw jets as circles (ellipses) of radius JET_R
    for jet in jets:
        jeta = jet.eta()
        jphi = jet.phi()
        ell = ROOT.TEllipse(jeta, jphi, JET_R, JET_R)
        ell.SetFillStyle(0)
        ell.SetLineColor(ROOT.kGreen + 2)
        ell.SetLineWidth(2)
        ell.Draw()
        keep.append(ell)
        # mark jet center
        m = ROOT.TMarker(jeta, jphi, 29)
        m.SetMarkerColor(ROOT.kGreen + 2)
        m.SetMarkerSize(1.5)
        m.Draw()
        keep.append(m)

    leg = ROOT.TLegend(0.70, 0.78, 0.98, 0.92)
    leg.AddEntry(keep[0] if keep else frame, "track (len #propto p_{T})", "l")
    leg.SetTextSize(0.03)
    leg.Draw()
    keep.append(leg)

    c.Update()
    c.SaveAs(f"{outdir}/event_{iev_global}.png")
    c.Close()


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
    h_phi = ROOT.TH1D("h_phi", "global track #varphi;#varphi;counts",            80, 0, 2*np.pi) #-np.pi, np.pi)

    # Jet histograms
    h_jet_pt  = ROOT.TH1D("h_jet_pt",  "jet p_{T};p_{T}^{jet} [GeV/c];counts", 1000, 0.0, 100.0)
    h_jet_eta = ROOT.TH1D("h_jet_eta", "jet #eta;#eta^{jet};counts",            80, -1.0, 1.0)
    h_jet_phi = ROOT.TH1D("h_jet_phi", "jet #varphi;#varphi^{jet};counts",            80, 0, 2*np.pi)
    h_jet_n   = ROOT.TH1D("h_jet_n",   "number of jets per event;N_{jets};events", 20, 0, 20)

    h_trackpt_jetpt = ROOT.TH2D("h_trackpt_jetpt", "track p_{T} vs jet p_{T};p_{T}^{jet} [GeV/c];p_{T}^{track} [GeV/c]", 200, 0, 20, 200, 0, 20)
    h_tracketa_jetpt = ROOT.TH2D("h_tracketa_jetpt", "track #eta vs jet p_{T};p_{T}^{jet} [GeV/c];#eta^{track} [GeV/c]", 200, 0, 20, 80, -1.0, 1.0)
    # h_trackpt_jetpt_ext = ROOT.TH2D("h_trackpt_jetpt_ext", "track p_{T} vs jet p_{T};p_{T}^{jet} [GeV/c];p_{T}^{track} [GeV/c]", 100, 0, 100, 100, 0, 100)

    h_trackpt_jetpt_edge = ROOT.TH2D("h_trackpt_jetpt_edge", "track p_{T} vs jet p_{T};p_{T}^{jet} [GeV/c];p_{T}^{track} [GeV/c]", 200, 0, 20, 200, 0, 20)
    h_tracketa_jetpt_edge = ROOT.TH2D("h_tracketa_jetpt_edge", "track #eta vs jet p_{T};p_{T}^{jet} [GeV/c];#eta^{track} [GeV/c]", 200, 0, 20, 80, -1.0, 1.0)

    # Track pt spectrum for events containing a jet > JET_TRIG_PT (jets in jet_eta < 0.5)
    h_pt_jettrig = ROOT.TH1D("h_pt_jettrig", "track p_{T} (events with jet > 8 GeV);p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    h_eta_jettrig = ROOT.TH1D("h_eta_jettrig", "track #eta (events with jet > 8 GeV);#eta;counts", 80, -1.0, 1.0)
    h_jet_n_jettrig = ROOT.TH1D("h_jet_n_jettrig", "number of selected jets per event (events with jet > 8 GeV);N_{jets};events", 20, 0, 20) #total selected jet multiplicity, but only for the triggered events (those with ≥1 jet above 8 GeV)
    h_jet_n_above_below_jettrig = ROOT.TH2D("h_jet_n_above_below", "jet multiplicity: above vs below 8 GeV;N_{jets}^{>8 GeV};N_{jets}^{<8 GeV}", 20, 0, 20, 20, 0, 20)
    
    # Track pt spectrum for events containing a jet > JET_TRIG_PT (no eta restriction)
    h_pt_jettrig_noetarestr = ROOT.TH1D("h_pt_jettrig_noetarestr", "track p_{T} (events with jet > 8 GeV at any #eta);p_{T} [GeV/c];counts", 200, 0.0, 20.0)

    # --- R=0.6 Checks ---
    h_pt_R06_trig = ROOT.TH1D("h_pt_R06_trig", "track p_{T} (events with R=0.6 jet > 8 GeV);p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    h_pt_R06_trig_fid = ROOT.TH1D("h_pt_R06_trig_fid", "track p_{T} (events with R=0.6 jet > 8 GeV, fiducial);p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    h_jet_pt_R06 = ROOT.TH1D("h_jet_pt_R06", "R=0.6 jet p_{T};p_{T} [GeV/c];counts", 100, 0.0, 100.0)
    h_jet_pt_R04_with_R06 = ROOT.TH1D("h_jet_pt_R04_with_R06", "R=0.4 jet p_{T} (events with R=0.6 jet);p_{T} [GeV/c];counts", 100, 0.0, 100.0)
    # fiducial versions: same as above but jets restricted to |eta_jet| < 0.3
    h_jet_pt_R06_fid = ROOT.TH1D("h_jet_pt_R06_fid",f"R=0.6 jet p_{{T}} (|#eta^{{jet}}| < {JET_ETA_MAX_R06:.1f});p_{{T}} [GeV/c];counts",100, 0.0, 100.0)
    h_jet_pt_R04_with_R04_fid = ROOT.TH1D("h_jet_pt_R04_with_R04_fid", f"R=0.4 (|#eta^{{jet}}| < {JET_ETA_MAX:.1f}),  jet p_{{T}} (events with R=0.6 jet);p_{{T}} [GeV/c];counts", 100, 0.0, 100.0)
    h_jet_pt_R04_with_R06_fid = ROOT.TH1D("h_jet_pt_R04_with_R06_fid", f"R=0.4 (|#eta^{{jet}}| < {JET_ETA_MAX_R06:.1f}),  jet p_{{T}} (events with R=0.6 jet);p_{{T}} [GeV/c];counts", 100, 0.0, 100.0)

    # track eta vs track pt for jet constituents
    h_tracketa_trackpt_fid  = ROOT.TH2D("h_tracketa_trackpt_fid", "track #eta vs track p_{T} (constituents, fiducial jets);p_{T}^{track} [GeV/c];#eta^{track}", 200, 0, 20, 80, -1.0, 1.0)
    h_tracketa_trackpt_jettrig_fid  = ROOT.TH2D("h_tracketa_trackpt_jettrig_fid", "track #eta vs track p_{T} (constituents, fiducial jets);p_{T}^{track} [GeV/c];#eta^{track}", 200, 0, 20, 80, -1.0, 1.0)
    h_tracketa_trackpt_edge = ROOT.TH2D("h_tracketa_trackpt_edge", "track #eta vs track p_{T} (constituents, jets>8 GeV outside fiducial);p_{T}^{track} [GeV/c];#eta^{track}", 200, 0, 20, 80, -1.0, 1.0)

    # look at tracks in R=0.6 jets
    h_trackpt_R06_trig = ROOT.TH1D("h_trackpt_R06_trig", "track p_{T} in R=0.6 jets > 8 GeV;p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    h_trackpt_R06_trig_fid = ROOT.TH1D("h_trackpt_R06_trig_fid", "track p_{T} in R=0.6 jets > 8 GeV, fiducial;p_{T} [GeV/c];counts", 200, 0.0, 20.0)
    
    # --- NEW: high-pt (> 10 GeV) tracks NOT in any jet ---
    h_highpt_nojet_eta_pt = ROOT.TH2D(
        "h_highpt_nojet_eta_pt",
        "high-p_{T} non-jet track #eta vs p_{T};p_{T}^{track} [GeV/c];#eta^{track}",
        200, 0, 20, 80, -1.0, 1.0)
    h_highpt_nojet_eta_phi = ROOT.TH2D(
        "h_highpt_nojet_eta_phi",
        "high-p_{T} non-jet track #eta vs #varphi;#varphi^{track};#eta^{track}",
        80, 0, 2 * np.pi, 80, -1.0, 1.0)
    # Delta R of high-pt non-jet track to leading jet and to closest jet
    h_highpt_nojet_dR_leadjet = ROOT.TH1D(
        "h_highpt_nojet_dR_leadjet",
        "#DeltaR(high-p_{T} non-jet track, leading jet);#DeltaR;counts",
        100, 0, 6.0)
    h_highpt_nojet_dR_closestjet = ROOT.TH1D(
        "h_highpt_nojet_dR_closestjet",
        "#DeltaR(high-p_{T} non-jet track, closest jet);#DeltaR;counts",
        100, 0, 6.0)

    # hCHECK_trackpt_jetsabove8gev_fid = ROOT.TH1D("hCHECK_trackpt_jetsabove8gev_fid",  "jet track p_{T};p_{T} [GeV/c], jets > 8 GeV, fiducial;counts", 200, 0.0, 20.0) # red
    # hCHECK_trackpt_jetsabove8gev_edge = ROOT.TH1D("hCHECK_trackpt_jetsabove8gev_edge",  "jet track p_{T};p_{T} [GeV/c], jets > 8 GeV, edge;counts", 200, 0.0, 20.0) # green
    # hCHECK_trackpt_jetsabove8gev_all = ROOT.TH1D("hCHECK_trackpt_jetsabove8gev_all",  "jet track p_{T};p_{T} [GeV/c], jets > 8 GeV, all;counts", 200, 0.0, 20.0) # grey

    # hCHECK_trackpt_evt_withjetsabove8gev_all = ROOT.TH1D("hCHECK_trackpt_evt_withjetsabove8gev_all",  "track p_{T};p_{T} [GeV/c], events with jets > 8 GeV, all;counts", 200, 0.0, 20.0) # magenta

    for h in (h_pt, h_eta, h_phi, h_jet_pt, h_jet_eta, h_jet_phi, h_jet_n, h_trackpt_jetpt, h_tracketa_jetpt, h_trackpt_jetpt_edge, h_tracketa_jetpt_edge, h_pt_jettrig, h_eta_jettrig, h_jet_n_above_below_jettrig, h_pt_jettrig_noetarestr, h_tracketa_trackpt_fid, h_tracketa_trackpt_jettrig_fid, h_tracketa_trackpt_edge, h_trackpt_R06_trig, h_trackpt_R06_trig_fid, h_highpt_nojet_eta_pt, h_highpt_nojet_eta_phi, h_highpt_nojet_dR_leadjet, h_highpt_nojet_dR_closestjet, h_pt_R06_trig, h_pt_R06_trig_fid, h_jet_pt_R06, h_jet_pt_R04_with_R06, h_jet_pt_R06_fid, h_jet_pt_R04_with_R04_fid, h_jet_pt_R04_with_R06_fid): #h_pt_ext, h_trackpt_jetpt_ext
        h.Sumw2()

    # for h in (hCHECK_trackpt_jetsabove8gev_fid, hCHECK_trackpt_jetsabove8gev_edge, hCHECK_trackpt_jetsabove8gev_all, hCHECK_trackpt_evt_withjetsabove8gev_all):
    #     h.Sumw2()

    # Bookkeeping counters stored as a 1-bin histogram (mergeable with hadd)
    h_cuts = ROOT.TH1D("h_cuts", "cut flow;;events or tracks", 6, 0, 6)
    h_cuts.GetXaxis().SetBinLabel(1, "events_total")
    h_cuts.GetXaxis().SetBinLabel(2, "events_sel8")
    h_cuts.GetXaxis().SetBinLabel(3, "events_sel8_rct")
    h_cuts.GetXaxis().SetBinLabel(4, "tracks_global")
    h_cuts.GetXaxis().SetBinLabel(5, "jets_sel")
    h_cuts.GetXaxis().SetBinLabel(6, "highpt_nojet_tracks")

    n_events_total = 0
    n_events_sel   = 0
    n_events_rct   = 0
    n_tracks_sel   = 0
    n_jets_sel     = 0
    n_highpt_nojet = 0

    n_displays_made = 0           # how many event displays drawn so far
    iev_global = 0                # global event counter (across batches)

    # anti-kt jet definition
    jetdef = fj.JetDefinition(fj.antikt_algorithm, JET_R)
    jetdef_06 = fj.JetDefinition(fj.antikt_algorithm, JET_R_06)

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

            # if iev_global == 100000:
            #     break

            iev_global += 1

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

            # --- R=0.6 Checks ---
            cluster_06 = fj.ClusterSequence(pj_particles, jetdef_06)
            jets_06 = fj.sorted_by_pt(cluster_06.inclusive_jets(JET_PT_MIN))

            has_R06_gt_8 = any(j.pt() > JET_TRIG_PT for j in jets_06)
            if has_R06_gt_8:
                w_ev = np.ones(len(ev_pt), dtype=np.float64)
                h_pt_R06_trig.FillN(len(ev_pt), ev_pt.astype(np.float64), w_ev)
                for j06 in jets_06:
                    h_jet_pt_R06.Fill(j06.pt())
                    for c_R06 in j06.constituents(): # fill in track pt here
                        h_trackpt_R06_trig.Fill(c_R06.pt())
                    if abs(j06.eta()) < JET_ETA_MAX_R06:
                        h_jet_pt_R06_fid.Fill(j06.pt())
                        for c_R06 in j06.constituents():
                            h_trackpt_R06_trig_fid.Fill(c_R06.pt())
                for j04 in jets:
                    h_jet_pt_R04_with_R06.Fill(j04.pt())
                    if abs(j04.eta()) < JET_ETA_MAX:
                        h_jet_pt_R04_with_R04_fid.Fill(j04.pt())
                    if abs(j04.eta()) < JET_ETA_MAX_R06:
                        h_jet_pt_R04_with_R06_fid.Fill(j04.pt())
            has_R06_gt_8_fid = any(j.pt() > JET_TRIG_PT and abs(j.eta()) < JET_ETA_MAX_R06 for j in jets_06)
            if has_R06_gt_8_fid:
                w_ev = np.ones(len(ev_pt), dtype=np.float64)
                h_pt_R06_trig_fid.FillN(len(ev_pt), ev_pt.astype(np.float64), w_ev)

            # has_greater = any(x > 8.0 for x in [jet.pt() for jet in jets])
            # if iev_global < 50 and has_greater:
            #     print()
            #     print("===== EVENT NUMBER", iev_global, "=====")
            #     print("Number of selected tracks:", len(ev_pt))
            #     print("Number of jets found:", len(jets))
            #     print("  with jet pts", [jet.pt() for jet in jets], "and jet etas", [jet.eta() for jet in jets])
            #     print("  and number of particles in each jet:", [len(jet.constituents()) for jet in jets])
            #     print("  with particle pts ", [[c.pt() for c in jet.constituents()] for jet in jets])
            #     print("  with particle etas", [[c.eta() for c in jet.constituents()] for jet in jets])
            #     print("    ev pts", len(ev_pt), ev_pt)
            #     print("    pj pts", len(pj_particles),[c.pt() for c in pj_particles])
            #     jet_const_list = [c for jet in jets for c in jet.constituents()]
            #     uncommon = list(set([c.pt() for c in pj_particles]) ^ set([c.pt() for c in jet_const_list]))
            #     print("    uncommon pts", len(uncommon), uncommon)

            #     print("   **, num tracks in jets:", len(jet_const_list))
            #     print("   **, num tracks in evts:", len(pj_particles))
            
            # if has_greater:
            #     for evpt in ev_pt:
            #         hCHECK_trackpt_evt_withjetsabove8gev_all.Fill(evpt)
            #     for jet in jets:
            #         if jet.pt() > JET_TRIG_PT:
            #             for c in jet.constituents():
            #                 hCHECK_trackpt_jetsabove8gev_all.Fill(c.pt())

            #             if abs(jet.eta()) <= JET_ETA_MAX:
            #                 for c in jet.constituents():
            #                     hCHECK_trackpt_jetsabove8gev_fid.Fill(c.pt())
            #             else:
            #                 for c in jet.constituents():
            #                     hCHECK_trackpt_jetsabove8gev_edge.Fill(c.pt())

            # ----- Collect (eta, phi) of all jet constituents in this event,
            #       used to flag tracks that ARE in a jet. -----
            constituent_etaphi = []
            constituent_pt = []
            for jet in jets:
                if jet.pt() > JET_TRIG_PT:          # match the >8 GeV definition
                    for c in jet.constituents():
                        constituent_etaphi.append((c.eta(), c.phi()))
                        constituent_pt.append(c.pt())

            njet_ev = 0
            njet_above_jettrig = 0   # jets above JET_TRIG_PT in this event
            njet_below_jettrig = 0   # jets below (or equal to) JET_TRIG_PT
            max_jet_pt = 0.0   # highest accepted-jet pt in this event (fiducial)
            max_jet_pt_noeta = 0.0   # highest jet pt in this event (any eta)
            for jet in jets:
                jpt  = jet.pt()
                jeta = jet.eta()
                jphi = jet.phi()

                if jpt > max_jet_pt_noeta:
                    max_jet_pt_noeta = jpt

                if abs(jeta) <= JET_ETA_MAX:
                    # ----- fiducial jet (all your usual cuts) -----
                    njet_ev += 1
                    n_jets_sel += 1

                    if jpt > max_jet_pt:
                        max_jet_pt = jpt
                    if jpt > JET_TRIG_PT:
                        njet_above_jettrig += 1
                    else:
                        njet_below_jettrig += 1

                    h_jet_pt.Fill(jpt)
                    h_jet_eta.Fill(jeta)
                    h_jet_phi.Fill(jphi)

                    for c in jet.constituents():
                        h_trackpt_jetpt.Fill(jpt, c.pt())
                        h_tracketa_jetpt.Fill(jpt, c.eta())
                        h_tracketa_trackpt_fid.Fill(c.pt(), c.eta())
                        if jpt > JET_TRIG_PT:
                            h_tracketa_trackpt_jettrig_fid.Fill(c.pt(), c.eta())
                    # if iev_global < 50:
                    #     print("   **,", iev_global, "DOING red num tracks in jets:", len(jet.constituents()))

                else:
                    # ----- edge jet: outside fiducial -----
                    for c in jet.constituents():
                        h_trackpt_jetpt_edge.Fill(jpt, c.pt())
                        h_tracketa_jetpt_edge.Fill(jpt, c.eta())
                        
                        # ----- edge jet: outside fiducial, only if > 8 GeV -----
                        if jpt > JET_TRIG_PT:
                            h_tracketa_trackpt_edge.Fill(c.pt(), c.eta())
                    # if iev_global < 50:
                    #     print("   **,", iev_global, "DOING green num tracks in jets:", len(jet.constituents()))


            h_jet_n.Fill(njet_ev)
            h_jet_n_above_below_jettrig.Fill(njet_above_jettrig, njet_below_jettrig)

            # ===== NEW: high-pt (>10 GeV) tracks NOT in any jet =====
            # A track is "in a jet" if its (eta, phi) matches a jet constituent.
            highpt_idx = []  # indices (into ev_*) of high-pt non-jet tracks
            ev_etaphi = []
            for e,p in zip(ev_eta, ev_phi):
                    ev_etaphi.append((e, p))

            # high_constituent_pt = [pt for pt in constituent_pt if pt > 10]
            # high_ev_pt = [pt for pt in ev_pt if pt > 10]
            # print("JET const (>10):", len(high_constituent_pt), high_constituent_pt)
            # print("EVT const (>10):", len(high_ev_pt), high_ev_pt)
            # # print("JET const:", len(constituent_pt), constituent_pt)
            # # print("EVT const:", len(ev_pt), ev_pt)
            for i in range(len(ev_pt)):
                if ev_pt[i] <= HIGHPT_TRACK_MIN:
                    continue
                t_eta = ev_eta[i]
                t_phi = ev_phi[i]
                in_jet = False
                for (c_eta, c_phi) in constituent_etaphi:
                    if (abs(c_eta - t_eta) < CONSTITUENT_MATCH_TOL and
                            abs(delta_phi(c_phi, t_phi)) < CONSTITUENT_MATCH_TOL):
                        in_jet = True
                        break
                if in_jet:
                    continue

                # This is a high-pt track not in any jet.
                highpt_idx.append(i)
                n_highpt_nojet += 1

                h_highpt_nojet_eta_pt.Fill(ev_pt[i], t_eta)
                # normalize track phi to [0, 2pi) for consistency with jet phi
                t_phi_norm = t_phi % (2 * np.pi)
                h_highpt_nojet_eta_phi.Fill(t_phi_norm, t_eta)

                # ----- Delta R to leading jet and to closest jet -----
                if len(jets) > 0:
                    lead_jet = jets[0]   # jets sorted by pt, so [0] is leading
                    dR_lead = delta_r(t_eta, t_phi, lead_jet.eta(), lead_jet.phi())
                    h_highpt_nojet_dR_leadjet.Fill(dR_lead)

                    dR_closest = min(
                        delta_r(t_eta, t_phi, jet.eta(), jet.phi())
                        for jet in jets
                    )
                    h_highpt_nojet_dR_closestjet.Fill(dR_closest)

            # ----- Event display for a few of these events -----
            if highpt_idx and n_displays_made < N_EVENT_DISPLAYS:
                draw_event_display(iev_global, ev_eta, ev_phi, ev_pt,
                                   jets, highpt_idx)
                n_displays_made += 1

            # Jet-triggered track spectrum (within jet eta): if this event has any accepted jet
            # above the trigger threshold, fill ALL selected tracks in the event.
            if max_jet_pt > JET_TRIG_PT:
                h_jet_n_jettrig.Fill(njet_ev)   # total selected jets in triggered events

                w_ev = np.ones(len(ev_pt), dtype=np.float64)
                h_pt_jettrig.FillN(len(ev_pt), ev_pt.astype(np.float64), w_ev)
                h_eta_jettrig.FillN(len(ev_eta), ev_eta.astype(np.float64), w_ev)
                # if iev_global < 50:
                #     print("--Blue Filled for event here!--", iev_global)

            # Jet-triggered track spectrum (any eta): if this event has any jet
            # above the trigger threshold anywhere, fill ALL selected tracks.
            if max_jet_pt_noeta > JET_TRIG_PT:
                w_ev = np.ones(len(ev_pt), dtype=np.float64)
                h_pt_jettrig_noetarestr.FillN(len(ev_pt), ev_pt.astype(np.float64), w_ev)
                # if iev_global < 50:
                #     print("   **, DOING num tracks in evts:", len(ev_pt))
                #     print("--Magenta Filled for event here!--", iev_global)


    h_cuts.SetBinContent(1, n_events_total)
    h_cuts.SetBinContent(2, n_events_sel)
    h_cuts.SetBinContent(3, n_events_rct)
    h_cuts.SetBinContent(4, n_tracks_sel)
    h_cuts.SetBinContent(5, n_jets_sel)
    h_cuts.SetBinContent(6, n_highpt_nojet)


    fout = ROOT.TFile(outfile, "RECREATE")
    h_pt.Write(); h_eta.Write(); h_phi.Write() #
    h_jet_pt.Write(); h_jet_eta.Write(); h_jet_phi.Write(); h_jet_n.Write()
    h_trackpt_jetpt.Write(); h_tracketa_jetpt.Write() #h_trackpt_jetpt_ext.Write()
    h_trackpt_jetpt_edge.Write(); h_tracketa_jetpt_edge.Write()
    h_pt_jettrig.Write(); h_eta_jettrig.Write(); h_jet_n_jettrig.Write(); h_jet_n_above_below_jettrig.Write()
    h_pt_jettrig_noetarestr.Write()
    h_tracketa_trackpt_fid.Write(); h_tracketa_trackpt_jettrig_fid.Write(); h_tracketa_trackpt_edge.Write()
    h_trackpt_R06_trig.Write(); h_trackpt_R06_trig_fid.Write()
    h_highpt_nojet_eta_pt.Write(); h_highpt_nojet_eta_phi.Write()
    h_highpt_nojet_dR_leadjet.Write(); h_highpt_nojet_dR_closestjet.Write()
    h_pt_R06_trig.Write(); h_pt_R06_trig_fid.Write();h_jet_pt_R06.Write(); h_jet_pt_R04_with_R06.Write()
    h_jet_pt_R06_fid.Write(); h_jet_pt_R04_with_R04_fid.Write(); h_jet_pt_R04_with_R06_fid.Write()
    h_cuts.Write()
    # hCHECK_trackpt_jetsabove8gev_fid.Write(); hCHECK_trackpt_jetsabove8gev_edge.Write(); hCHECK_trackpt_jetsabove8gev_all.Write(); hCHECK_trackpt_evt_withjetsabove8gev_all.Write()
    fout.Close()

    print(f"Done: {infile}")
    print(f"  events total: {n_events_total}")
    print(f"  events sel8:  {n_events_sel}")
    print(f"  global trks (pt>={PT_MIN}, |eta|<={ETA_MAX}):  {n_tracks_sel}")
    print(f"  jets (anti-kt R={JET_R}, pt>={JET_PT_MIN}, |eta|<={JET_ETA_MAX:.2f}):  {n_jets_sel}")
    print(f"  high-pt (>{HIGHPT_TRACK_MIN}) tracks NOT in a jet:  {n_highpt_nojet}")
    print(f"  event displays written:  {n_displays_made}")

if __name__ == "__main__":
    infile, outfile = sys.argv[1], sys.argv[2]
    process(infile, outfile)