#!/usr/bin/env python

''' HOW TO RUN:
  - Default (z=0.1, 0.2 and maxkt), binned in the UNGROOMED jet pt:
  python data_process_jets.py input.parquet output.root
  - Same, but binned in the GROOMED (radiator) jet pt:
  python data_process_jets.py input.parquet output.root --binning groomed
  - Specific z-cuts (e.g., 0.15 and 0.25) and maxkt:
  python data_process_jets.py input.parquet output.root --zcuts 0.15 0.25
  - Only z=0.1, no maxkt:
  python data_process_jets.py input.parquet output.root --zcuts 0.1 --no-maxkt

  NOTE on --binning groomed:
    * The groomed pt is the pt of the radiator (the parent of the selected
      splitting), so it depends on the cut configuration: each (slice, cut)
      combination gets its own jet population and its own <pt>.
    * Jets that fail the cut have no groomed pt and are therefore not assigned
      to any slice. In groomed mode num_jets == num_jets_passed_cut and
      hist_jetpt_all only contains jets that passed the cut.
    * Slice tags are prefixed with "gjetpt" instead of "jetpt" (override with
      --slice-prefix).
'''

import math
import numpy as np
import pandas as pd
import fastjet as fj
import fjcontrib
import ecorrel
import ROOT
import argparse
from collections import defaultdict

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()


class DataAnalysis:
    def __init__(self):
        self.jet_R = 0.4
        self.jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
        self.trk_thrd = 1
        self.dphi_cut = -9999
        self.deta_cut = -9999
        # Jet-pt slices (lower edges); each pair = [low, high) in GeV.
        # NOTE: these may overlap, a jet is filled into every slice it matches.
        self.pt_slices = [
            (10, 20), (20, 40), (40, 60), (60, 80), (80, 100),
            (100, 120), (120, 150), (150, 200), (200, 500), (50, 60)
        ]

        # binning mode: "ungroomed" (jet pt) or "groomed" (radiator pt)
        self.binning = "ungroomed"
        self.slice_prefix = "jetpt"
        self.unweighted = False

        # histogram axes
        self.RL_NBINS = 25  # 75
        RL_MIN, RL_MAX = 0.01, 1.0
        self.RL_BINS = np.logspace(np.log10(RL_MIN), np.log10(RL_MAX), self.RL_NBINS + 1)
        
        ptrl_xmin, ptrl_xmax = 0.1, 100
        self.ptrl_log_bins = np.logspace(np.log10(ptrl_xmin), np.log10(ptrl_xmax), self.RL_NBINS + 1)
        self.pt_bins = np.linspace(0, 500, 501)       # flat 0-500 GeV jet pt
        self.radpt_bins = np.linspace(0, 500, 501)    # flat 0-500 GeV radiator pt
        self.radkt_bins = np.linspace(-5, 5, self.RL_NBINS + 1)

        # RAW DATA BINS FOR UNFOLDING
        self.JETPT_UNF_BINS = np.array([10, 20, 40, 60, 80, 100, 120, 150, 200, 500], dtype=np.float64)

        W_NBINS = 20 #30 #100
        W_MIN, W_MAX = 0.0, 0.3 #1.0 #0.3
        W_BINS = np.logspace(-5,-0.5,W_NBINS+1) #0.00001 - 0.316227766 # W_BINS = np.linspace(W_MIN, W_MAX, W_NBINS + 1)


        # (object, weight) combinations that are actually computed/stored
        self.objects = ["full", "rad", "AA", "BB", "AB"]
        self.weight_labels = ["wwjetpt", "wwradpt", "wwnullpt"]

    # ---------- helpers ----------
    def FormatHist(self, hist, norm_factor, pt_rl=False, avg_pt=None):
        if pt_rl:
            if avg_pt is None:
                raise ValueError("avg_pt required for ptRL scaling")
            hist.Scale(math.log(avg_pt) / avg_pt)
        if norm_factor not in (-1, 0):
            hist.Scale(1 / norm_factor, "width")

    def GetCABHist(self, hist_AA, hist_BB, hist_AB, name):
        hist_CAB = hist_AB.Clone(name)
        hist_CAB.SetTitle("C_{AB};R_{L};AxB / #sqrt{AxA #times BxB}")
        hist_CAB.Reset()
        for i in range(1, hist_CAB.GetNbinsX() + 1):
            aa, bb, ab = hist_AA.GetBinContent(i), hist_BB.GetBinContent(i), hist_AB.GetBinContent(i)
            sig_aa, sig_bb, sig_ab = hist_AA.GetBinError(i), hist_BB.GetBinError(i), hist_AB.GetBinError(i)
            if aa > 0 and bb > 0:
                denom = math.sqrt(aa * bb)
                result = ab / denom
                term_ab = sig_ab / denom
                term_aa = (result * sig_aa) / (2 * aa)
                term_bb = (result * sig_bb) / (2 * bb)
                hist_CAB.SetBinContent(i, result)
                hist_CAB.SetBinError(i, np.sqrt(term_ab**2 + term_aa**2 + term_bb**2))
            else:
                hist_CAB.SetBinContent(i, 0)
                hist_CAB.SetBinError(i, 0)
        return hist_CAB

    def format_cut_tag(self, cut_mode, cut_value):
        return f"sd{cut_value}" if cut_mode == "sd" else "maxkt"

    def slice_tag(self, lo, hi):
        return f"{self.slice_prefix}{lo}_{hi}"

    # takes in a pt value, and returns which pt bins that pt value falls into (multiple bins possible if they overlap)
    def slices_for_pt(self, pt):
        """All slices a given pt falls into (slices may overlap)."""
        if pt is None:
            return []
        return [(lo, hi) for (lo, hi) in self.pt_slices if lo <= pt < hi]

    def select_split(self, lund_plane_elements, cut_mode, cut_value):
        if cut_mode == "sd":
            for d in lund_plane_elements:
                if d.z() > cut_value:
                    return d
            return None
        if cut_mode == "maxkt":
            filtered = [d for d in lund_plane_elements if d.kt() > 1.0]
            return max(filtered, key=lambda d: d.kt(), default=None) if filtered else None
        return None

    def get_split(self, lund_plane_elements, cut_mode, cut_value):
        """Return (lund_element, radiator, [subjet_a, subjet_b]) or (None, None, None)."""
        d = self.select_split(lund_plane_elements, cut_mode, cut_value)
        if d is None:
            return None, None, None
        parent_radiator = d.pair()
        subjets = sorted(parent_radiator.pieces(), key=lambda x: x.pt(), reverse=True)
        if len(subjets) != 2:
            return None, None, None
        return d, parent_radiator, subjets

    def cluster_jet(self, jet_constituents):
        """Recluster the constituents with C/A. Returns (cluster_sequence, jet).
        The cluster sequence must be kept alive while the jet is used."""
        c_pt = jet_constituents['c_pt'].to_numpy()
        c_eta = jet_constituents['c_eta'].to_numpy()
        c_phi = jet_constituents['c_phi'].to_numpy()
        px = c_pt * np.cos(c_phi)
        py = c_pt * np.sin(c_phi)
        pz = c_pt * np.sinh(c_eta)
        E = np.sqrt(px**2 + py**2 + pz**2)
        pj_particles = [fj.PseudoJet(float(px[k]), float(py[k]), float(pz[k]), float(E[k]))
                        for k in range(len(c_pt))]
        cs = fj.ClusterSequence(pj_particles, self.jet_def_ca)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        if not jets:
            return cs, None
        return cs, jets[0]

    def count_constituents(self, sj):
        n = 0
        for c in fj.sorted_by_pt(sj.constituents()):
            if c.pt() < self.trk_thrd:
                break
            n += 1
        return n

    def select_constituents(self, sj):
        c_select = fj.vectorPJ()
        for c in fj.sorted_by_pt(sj.constituents()):
            if c.pt() < self.trk_thrd:
                break
            c_select.append(c)
        return c_select

    def compute_correlator(self, sj_A, weight, sj_B=None):
        """Compute the 2-point correlator once; returns a list of (RL, weight)."""
        c_select = self.select_constituents(sj_A)
        if sj_B is not None:
            c_select_B = self.select_constituents(sj_B)
            eec_result = ecorrel.CorrelatorBuilder(c_select, c_select_B, weight, 2, 1,
                                                   self.dphi_cut, self.deta_cut)
        else:
            eec_result = ecorrel.CorrelatorBuilder(c_select, weight, 2, 1,
                                                   self.dphi_cut, self.deta_cut)
        rs = eec_result.correlator(2).rs()
        ws = eec_result.correlator(2).weights()
        return [(rs[i], ws[i]) for i in range(rs.size())]

    def fill_correlator(self, pairs, hist, hist_ptRL, avg_pt, weighted=True):
        for rl_value, weight_value in pairs:
            if weighted:
                hist.Fill(rl_value, weight_value)
                if hist_ptRL is not None:
                    hist_ptRL.Fill(rl_value * avg_pt, weight_value)
            else:
                hist.Fill(rl_value)
                if hist_ptRL is not None:
                    hist_ptRL.Fill(rl_value * avg_pt)

    # ---------- histogram booking ----------
    def hist_names(self, obj, tag, wlabel):
        """Reproduce the original naming convention."""
        if obj == "full":
            if wlabel == "wwjetpt":
                return f"hist_full_{tag}", f"hist_full_ptRL_{tag}"
            return f"hist_full_{tag}_{wlabel}", f"hist_full_ptRL_{tag}_{wlabel}"
        return f"hist_{obj}_{tag}_{wlabel}", f"hist_{obj}_ptRL_{tag}_{wlabel}"

    def book_hists(self, tag, cut_mode):
        nbins = self.RL_NBINS
        rl_log_bins = self.RL_BINS
        ptrl_log_bins = self.ptrl_log_bins
        h = {}

        titles = {"full": "EEC", "rad": "EEC", "AA": "AxA", "BB": "BxB", "AB": "AxB"}
        for obj in self.objects:
            for wlabel in self.weight_labels:
                if obj == "full" and wlabel == "wwradpt":
                    continue  # the full jet is never radiator-pt weighted
                name_rl, name_ptrl = self.hist_names(obj, tag, wlabel)
                h[f"{obj}_{wlabel}"] = ROOT.TH1D(name_rl, f"{titles[obj]}; R_{{L}}", nbins, rl_log_bins)
                h[f"{obj}_ptRL_{wlabel}"] = ROOT.TH1D(
                    name_ptrl, f"{titles[obj]}; <p_{{T}}>R_{{L}} [GeV/c]", nbins, ptrl_log_bins)

        h['jetpt_all'] = ROOT.TH1D(f"hist_jetpt_all_{tag}", "jet p_{T}; p_{T,jet}",
                                   len(self.pt_bins) - 1, self.pt_bins)
        h['jetpt_cut'] = ROOT.TH1D(f"hist_jetpt_{tag}", "jet p_{T} (cut); p_{T,jet}",
                                   len(self.pt_bins) - 1, self.pt_bins)
        h['radiatorpt'] = ROOT.TH1D(f"radiator_pt_{tag}", "radiator p_{T}; p_{T,radiator}",
                                    len(self.radpt_bins) - 1, self.radpt_bins)
        h['radiatorkt'] = ROOT.TH1D(f"radiator_lnkt_{tag}", "radiator ln k_{T}; ln(k_{T,radiator})",
                                    nbins, self.radkt_bins)
        if cut_mode == "sd":
            h['rg'] = ROOT.TH1D(f"hist_rg_{tag}", "R_{g} = #DeltaR_{AB}; R_{g}", nbins, rl_log_bins)

        n_max = 60
        h['nA_nB'] = ROOT.TH2D(f"hist_nA_nB_{tag}", "N in A vs B; N_{A}; N_{B}",
                               n_max, -0.5, n_max - 0.5, n_max, -0.5, n_max - 0.5)
        h['nTotalUnGroomed'] = ROOT.TH1D(f"hist_nTotalUnGroomed_{tag}", "N ungroomed; N",
                                         n_max, -0.5, n_max - 0.5)
        h['nTotalGroomed'] = ROOT.TH1D(f"hist_nTotalGroomed_{tag}", "N groomed; N",
                                       n_max, -0.5, n_max - 0.5)
        comb_max = n_max * n_max
        h['combAA'] = ROOT.TH1D(f"hist_combAA_{tag}", "AxA comb", 200, -0.5, comb_max - 0.5)
        h['combBB'] = ROOT.TH1D(f"hist_combBB_{tag}", "BxB comb", 200, -0.5, comb_max - 0.5)
        h['combAB'] = ROOT.TH1D(f"hist_combAB_{tag}", "AxB comb", 200, -0.5, comb_max - 0.5)
        h['combTotal'] = ROOT.TH1D(f"hist_combTotal_{tag}", "Total comb", 200, -0.5, 4 * comb_max - 0.5)
        return h

    # ---------- average pt determination ----------
    def average_pts_ungroomed(self, df, active_cuts):
        """<pt_jet> per slice, taken directly from the dataframe (as before).
        Identical for every cut configuration."""
        avgs = {}
        for (lo, hi) in self.pt_slices:
            df_slice = df[(df['jet_pt'] >= lo) & (df['jet_pt'] < hi)]
            if len(df_slice) == 0:
                continue
            avg = df_slice.groupby(['event_id','jet_id'])['jet_pt'].first().mean() #df_slice['jet_pt'].mean()
            for cut in active_cuts:
                avgs[((lo, hi), cut)] = avg
        return avgs

    def average_pts_groomed(self, grouped, active_cuts, report_every=20000, njet_cutoff=-1):
        """Pre-pass: cluster + Lund only (no EEC) to get <pt_groomed> per (slice, cut)."""
        sums = defaultdict(float)
        counts = defaultdict(int)
        lund_gen = fjcontrib.LundGenerator(self.jet_def_ca)
        njets = 0
        print("[pre-pass] scanning jets to determine <groomed pt> per slice ...")
        for _, jet_constituents in grouped:
            cs, jet = self.cluster_jet(jet_constituents)
            if jet is None:
                continue
            njets += 1
            if njets % report_every == 0:
                print(f"[pre-pass] {njets} jets")
            if njets >= njet_cutoff and njet_cutoff >= 0:
                break
            lund_plane_elements = lund_gen.result(jet)
            for cut_mode, cut_value in active_cuts:
                _, parent_radiator, subjets = self.get_split(lund_plane_elements, cut_mode, cut_value)
                if parent_radiator is None:
                    continue
                rad_pt = parent_radiator.perp()
                for sl in self.slices_for_pt(rad_pt):
                    sums[(sl, (cut_mode, cut_value))] += rad_pt
                    counts[(sl, (cut_mode, cut_value))] += 1
        print(f"[pre-pass] done, {njets} jets scanned")
        return {k: sums[k] / counts[k] for k in counts}

    # ---------- main ----------
    def run(self, infile, outfile, zcuts=[0.1, 0.2], use_maxkt=True, unweighted=False,
            binning="ungroomed", slice_prefix=None, njet_cutoff=-1):

        self.binning = binning
        self.unweighted = unweighted
        if slice_prefix is not None:
            self.slice_prefix = slice_prefix
        else:
            self.slice_prefix = "gjetpt" if binning == "groomed" else "jetpt"

        df = pd.read_parquet(infile)

        # Determine which cut configurations to use
        active_cuts = [("sd", z) for z in zcuts]
        if use_maxkt:
            active_cuts.append(("maxkt", None))

        # Pre-filter jets that can never end up in any slice.
        lo_min = min(lo for lo, _ in self.pt_slices)
        if binning == "ungroomed":
            mask = np.zeros(len(df), dtype=bool)
            for (lo, hi) in self.pt_slices:
                mask |= (df['jet_pt'] >= lo) & (df['jet_pt'] < hi)
            df = df[mask]
        else:
            # groomed pt <= ungroomed pt, so jets below the lowest slice edge can be dropped
            df = df[df['jet_pt'] >= lo_min]

        if len(df) == 0:
            print("No jets left after pt pre-selection, nothing to do.")
            return

        grouped_jets = df.groupby(['event_id', 'jet_id'], sort=False)

        # ---- averages used for the ptRL scaling ----
        if binning == "groomed":
            avg_pts = self.average_pts_groomed(grouped_jets, active_cuts, njet_cutoff=njet_cutoff)
        else:
            avg_pts = self.average_pts_ungroomed(df, active_cuts)

        for (lo, hi) in self.pt_slices:
            for cut in active_cuts:
                key = ((lo, hi), cut)
                if key in avg_pts:
                    print(f"[{self.slice_tag(lo, hi)}_{self.format_cut_tag(*cut)}] "
                          f"<pt> = {avg_pts[key]:.2f}  ({binning})")

        # Make raw distributions for unfolding
        raw1Dhists = {}
        raw3Dhists = {}
        if binning == "groomed":
            for cut in active_cuts:
                raw1Dhist = ROOT.TH1D(f"groomed_{self.format_cut_tag(*cut)}_jet_pt_raw1D", "groomed jet p_{T}; p_{T,gr. jet}", len(self.pt_bins) - 1, self.pt_bins)
                raw1Dhists[cut] = raw1Dhist
                for obj in self.objects:
                    raw3Dhist = ROOT.TH3D(f"{obj}_{self.format_cut_tag(*cut)}_raw", 
                                          f"raw {obj} EEC;p_{T,gr. jet};R_{L};weight", 
                                          len(self.JETPT_UNF_BINS) - 1, self.JETPT_UNF_BINS,
                                          self.RL_NBINS, self.RL_BINS,
                                          self.W_NBINS, self.W_BINS)
                    raw3Dhists[(cut, obj)] = raw3Dhist
        else:
            for cut in active_cuts:
                raw1Dhist = ROOT.TH1D(f"ungroomed_{self.format_cut_tag(*cut)}_jet_pt_raw1D", "ungroomed jet p_{T}; p_{T,jet}", len(self.pt_bins) - 1, self.pt_bins)
                raw1Dhists[cut] = raw1Dhist
                for obj in self.objects:
                    raw3Dhist = ROOT.TH3D(f"{obj}_{self.format_cut_tag(*cut)}_raw", 
                                          f"raw {obj} EEC;p_{T, jet};R_{L};weight", 
                                          len(self.JETPT_UNF_BINS) - 1, self.JETPT_UNF_BINS,
                                          self.RL_NBINS, self.RL_BINS,
                                          self.W_NBINS, self.W_BINS)
                    raw3Dhists[(cut, obj)] = raw3Dhist
        

        # ---- book everything up front ----
        root_outfile = ROOT.TFile(outfile, "RECREATE")
        hists = {}
        for (lo, hi) in self.pt_slices:
            for cut in active_cuts:
                tag = f"{self.slice_tag(lo, hi)}_{self.format_cut_tag(*cut)}"
                hists[((lo, hi), cut)] = self.book_hists(tag, cut[0])

        counters = defaultdict(lambda: {'num_jets': 0., 'num_jets_passed_cut': 0.,
                                        'sum_jetpt_passed_cut': 0., 'sum_radpt_passed_cut': 0.})

        # ---- main jet loop ----
        lund_gen = fjcontrib.LundGenerator(self.jet_def_ca)
        njets = 0
        for (event_idx, jet_id), jet_constituents in grouped_jets:
            njets += 1
            if njets % 20000 == 0:
                print(f"[main] {njets} jets (event {event_idx})")
            
            if njets >= njet_cutoff and njet_cutoff >= 0:
                break

            jet_pt_stored = float(jet_constituents['jet_pt'].iloc[0])

            cs, jet = self.cluster_jet(jet_constituents)
            if jet is None:
                continue

            # In ungroomed mode the slice is known before any grooming, so all jets
            # (including those failing the cut) can be counted.
            ungroomed_slices = self.slices_for_pt(jet_pt_stored)
            if binning == "ungroomed":
                if not ungroomed_slices:
                    continue
                for sl in ungroomed_slices:
                    for cut in active_cuts:
                        counters[(sl, cut)]['num_jets'] += 1
                        hists[(sl, cut)]['jetpt_all'].Fill(jet.perp())
                        raw1Dhists[cut].Fill(jet.perp()) # for unfolding

            lund_plane_elements = lund_gen.result(jet)

            for cut in active_cuts:
                cut_mode, cut_value = cut
                selected_d, parent_radiator, subjets = self.get_split(lund_plane_elements,
                                                                      cut_mode, cut_value)
                if selected_d is None:
                    continue
                subjet_a, subjet_b = subjets

                # ---- which slices does this jet belong to? ----
                if binning == "groomed":
                    target_slices = self.slices_for_pt(parent_radiator.perp())
                else:
                    target_slices = ungroomed_slices
                if not target_slices:
                    continue

                # ---- correlators: computed once, filled into every matching slice ----
                weight_specs = [("wwjetpt", jet.perp()), ("wwradpt", parent_radiator.perp())]
                if self.unweighted:
                    weight_specs.append(("wwnullpt", -1))

                obj_specs = [("full", jet, None), ("rad", parent_radiator, None),
                             ("AA", subjet_a, None), ("BB", subjet_b, None),
                             ("AB", subjet_a, subjet_b)]

                computed = {}
                for obj, sj_A, sj_B in obj_specs:
                    for wlabel, wval in weight_specs:
                        if obj == "full" and wlabel == "wwradpt":
                            continue
                        computed[(obj, wlabel)] = self.compute_correlator(sj_A, wval, sj_B)

                n_A = self.count_constituents(subjet_a)
                n_B = self.count_constituents(subjet_b)
                n_ungroomed = self.count_constituents(jet)
                n_groomed = self.count_constituents(parent_radiator)
                kt = selected_d.kt()

                for sl in target_slices:
                    h = hists[(sl, cut)]
                    c = counters[(sl, cut)]
                    avg_pt = avg_pts.get((sl, cut), 1.0)

                    if binning == "groomed":
                        # no meaningful "all jets" population in groomed mode
                        c['num_jets'] += 1
                        h['jetpt_all'].Fill(jet.perp())
                        raw1Dhists[cut].Fill(parent_radiator.perp()) # for unfolding

                    c['num_jets_passed_cut'] += 1
                    c['sum_jetpt_passed_cut'] += jet.perp()
                    c['sum_radpt_passed_cut'] += parent_radiator.perp()

                    h['jetpt_cut'].Fill(jet.perp())
                    h['radiatorpt'].Fill(parent_radiator.perp())
                    if kt > 0:
                        h['radiatorkt'].Fill(np.log(kt))
                    if cut_mode == "sd":
                        h['rg'].Fill(selected_d.Delta())

                    h['nA_nB'].Fill(n_A, n_B)
                    h['nTotalUnGroomed'].Fill(n_ungroomed)
                    h['nTotalGroomed'].Fill(n_groomed)
                    h['combAA'].Fill(n_A * n_A)
                    h['combBB'].Fill(n_B * n_B)
                    h['combAB'].Fill(n_A * n_B * 2)
                    h['combTotal'].Fill((n_A + n_B) ** 2)

                    for (obj, wlabel), pairs in computed.items():
                        self.fill_correlator(pairs,
                                             h[f"{obj}_{wlabel}"],
                                             h[f"{obj}_ptRL_{wlabel}"],
                                             avg_pt,
                                             weighted=(wlabel != "wwnullpt"))
                        if wlabel == "wwradpt":
                            for rl_value, weight_value in pairs:
                                if binning == "groomed":
                                    raw3Dhists[(cut, obj)].Fill(parent_radiator.perp(), rl_value, weight_value)
                                else:
                                    raw3Dhists[(cut, obj)].Fill(jet.perp(), rl_value, weight_value)

        print(f"[main] done, {njets} jets processed")

        # ---- normalize / format / write ----
        for cut in active_cuts:
            raw1Dhists[cut].Write()
            for obj in self.objects:
                raw3Dhists[(cut, obj)].Write()

        for (lo, hi) in self.pt_slices:
            for cut in active_cuts:
                sl = (lo, hi)
                h = hists[(sl, cut)]
                c = counters[(sl, cut)]
                tag = f"{self.slice_tag(lo, hi)}_{self.format_cut_tag(*cut)}"
                avg_pt = avg_pts.get((sl, cut), 1.0)

                # self.FormatHist(h['full_wwjetpt'], c['num_jets_passed_cut'])
                # self.FormatHist(h['full_ptRL_wwjetpt'], c['num_jets_passed_cut'],
                #                 pt_rl=True, avg_pt=avg_pt)
                # ... (same for the rest, kept commented out as in the original)

                cab = []
                for wlabel in ["wwjetpt", "wwradpt"]:
                    cab.append(self.GetCABHist(h[f"AA_{wlabel}"], h[f"BB_{wlabel}"],
                                               h[f"AB_{wlabel}"], f"CAB_{tag}_{wlabel}"))
                    cab.append(self.GetCABHist(h[f"AA_ptRL_{wlabel}"], h[f"BB_ptRL_{wlabel}"],
                                               h[f"AB_ptRL_{wlabel}"], f"CAB_ptRL_{tag}_{wlabel}"))

                for hist in h.values():
                    hist.Write()
                for hist in cab:  # can probably remove CAB, needs proper normalization first
                    hist.Write()

                h_counters = ROOT.TH1D(f"counters_{tag}", "counters", 4, 0, 4)
                h_counters.GetXaxis().SetBinLabel(1, "num_jets")
                h_counters.GetXaxis().SetBinLabel(2, "num_jets_passed_cut")
                h_counters.GetXaxis().SetBinLabel(3, "sum_jetpt_passed_cut")
                h_counters.GetXaxis().SetBinLabel(4, "sum_radpt_passed_cut")
                h_counters.SetBinContent(1, c['num_jets'])
                h_counters.SetBinContent(2, c['num_jets_passed_cut'])
                h_counters.SetBinContent(3, c['sum_jetpt_passed_cut'])
                h_counters.SetBinContent(4, c['sum_radpt_passed_cut'])
                h_counters.Sumw2(False)  # exact counts, avoids error propagation weirdness
                h_counters.Write()

                print(f"[{tag}] jets={c['num_jets']} passed_cut={c['num_jets_passed_cut']}")

        # store the binning mode in the file so downstream code can check it
        ROOT.TNamed("binning_mode", self.binning).Write()
        root_outfile.Close()

        # write per-slice average pt
        avg_txt_path = outfile.replace(".root", "_avg_jet_pts.txt")
        with open(avg_txt_path, "w") as f:
            if binning == "groomed":
                # groomed pt depends on the cut, so the cut tag is part of the key
                f.write("# lo hi cut_tag avg_groomed_pt\n")
                for (lo, hi) in self.pt_slices:
                    for cut in active_cuts:
                        if ((lo, hi), cut) in avg_pts:
                            f.write(f"{lo} {hi} {self.format_cut_tag(*cut)} "
                                    f"{avg_pts[((lo, hi), cut)]:.6f}\n")
            else:
                written = set()
                for (lo, hi) in self.pt_slices:
                    for cut in active_cuts:
                        if ((lo, hi), cut) in avg_pts and (lo, hi) not in written:
                            written.add((lo, hi))
                            f.write(f"{lo} {hi} {avg_pts[((lo, hi), cut)]:.6f}\n")

        print(f"Done: {infile} -> {outfile}  (binning = {binning})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process jets and generate EEC histograms.")
    parser.add_argument("infile", help="Input Parquet file")
    parser.add_argument("outfile", help="Output ROOT file")
    parser.add_argument("--zcuts", type=float, nargs='+', default=[0.1, 0.2],
                        help="Specify z-cut values for SD selection (e.g., --zcuts 0.1 0.2)")
    parser.add_argument("--no-maxkt", action="store_true", help="Disable maxkt selection")
    parser.add_argument("--add-noweight", action="store_true",
                        help="Also look at no weight EECs (saved _wwnullpt)")
    parser.add_argument("--binning", choices=["ungroomed", "groomed"], default="ungroomed",
                        help="Bin the histograms in the ungroomed jet pt (default) or in the "
                             "groomed/radiator pt of the selected splitting")
    parser.add_argument("--slice-prefix", default=None,
                        help="Override the slice tag prefix (default: 'jetpt' for ungroomed "
                             "binning, 'gjetpt' for groomed binning)")
    parser.add_argument("--njetcutoff", type=int, default=-1, help="Number of jets to run")

    args = parser.parse_args()
    DataAnalysis().run(args.infile, args.outfile,
                       zcuts=args.zcuts,
                       use_maxkt=not args.no_maxkt,
                       unweighted=args.add_noweight,
                       binning=args.binning,
                       slice_prefix=args.slice_prefix,
                       njet_cutoff=args.njetcutoff)