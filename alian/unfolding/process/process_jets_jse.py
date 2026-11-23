
import pandas as pd
import fastjet as fj
import fjcontrib
import fjext
import ecorrel

import ROOT

import sys
import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np
from array import array
import math
import json

from pyjetty.mputils import *
from pyjetty.mputils.mputils import pinfo, pwarning

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

from pyjetty.alice_analysis.process.base import process_base

from enum import Enum
import fjtools

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

class MyAnalysis:
    def __init__(self, def_gen, narrow=False):
        # Jet pts
        self.target_jet_pts = [ 50, 100, 200, 500 ]
        self.gen = def_gen
        self.narrow = narrow
        
        self.partontypes = [ "inclusive", "quark", "gluon"]

        # num_files_to_parse = 1 #100

        # Jet Definitions
        self.jet_R = 0.4
        self.jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0) #self.jet_R) # jetR * 2

        # Cut definitions
        # Use soft drop ('sd') with a z cut, or use max kt ('maxkt') selection.
        self.cut_configs = [ ("sd", 0.1), ("maxkt", None) ]
        # Example: self.cut_configs = [("sd", 0.1), ("maxkt", None)]

        # EEC definitions
        self.trk_thrd = 1
        self.dphi_cut = -9999
        self.deta_cut = -9999

    def narrow_hi(self, target_jetpt):
        """High edge for the narrow bin (jetpt * 1.1)."""
        return int(round(target_jetpt * 1.1))

    def jetpt_tag(self, target_jetpt):
        """Tag used in filenames: 'jetpt50_55' for narrow, 'jetpt50' otherwise."""
        if self.narrow:
            return f"jetpt{target_jetpt}_{self.narrow_hi(target_jetpt)}"
        return f"jetpt{target_jetpt}"
    
    def output_dir(self):
        base = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles"
        if self.narrow:
            d = os.path.join(base, "narrowerbins1.1")
            os.makedirs(d, exist_ok=True)
            return d
        return base
        
    def get_parton_type(self, pid: int) -> str:
        if abs(pid) in {1, 2, 3, 4, 5, 6}:
            return "quark"
        elif pid == 21:
            return "gluon"
        elif pid == 22:
            return "photon"
        else:
            return "unknown"

    def FormatHist(self, hist, norm_factor, color, markerstyle=0, coloralpha=1, pt_rl=False, avg_pt=None):
        if pt_rl:
            if avg_pt is None:
                raise ValueError("avg_pt is required for ptRL histogram scaling")
            hist.Scale(math.log(avg_pt) / avg_pt)
        if norm_factor not in (-1, 0):
            hist.Scale(1/norm_factor, "width")
        hist.SetLineColor(color)
        if coloralpha != 1:
            hist.SetLineColorAlpha(color, coloralpha)
        hist.SetLineWidth(2)
        if markerstyle > 0:
            hist.SetMarkerStyle(markerstyle)
            hist.SetMarkerColor(color)

    def book_histograms(self, target_jetpt, cut_mode, cut_value, partontype):
        nbins = 25
        xmin, xmax = 0.001, 1.0
        ptrl_xmin, ptrl_xmax = 0.1, 50
        log_bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
        ptrl_log_bins = np.logspace(np.log10(ptrl_xmin), np.log10(ptrl_xmax), nbins + 1)
        pt_bins = np.arange(int(target_jetpt / 2), target_jetpt * 1.2 + 2, dtype=float)
        radpt_bins = np.linspace(0, int(target_jetpt * 1.2 + 2), int(target_jetpt * 1.2 + 2) + 1)
        radkt_bins = np.linspace(-5, 5, nbins + 1)

        cut_tag = self.format_cut_tag(cut_mode, cut_value)
        cut_label = self.get_cut_label(cut_mode, cut_value)
        hist_jetpt_label = f"SD z_{{cut}} = {cut_value}" if cut_mode == "sd" else "Max k_{T} jet p_{T}"

        n_max = 60
        comb_max = n_max * n_max
        suffix = f"{partontype}_jetpt{target_jetpt}_{cut_tag}"

        h = {}
        h["jetpt_all"] = ROOT.TH1D(f"hist_jetpt_all_{suffix}", f"jet p_{{T}} ({cut_label}); p_{{T, jet}}", len(pt_bins) - 1, pt_bins)
        h["jetpt_cut"] = ROOT.TH1D(f"hist_jetpt_{suffix}", f"{hist_jetpt_label}; p_{{T, jet}}", len(pt_bins) - 1, pt_bins)
        h["full"] = ROOT.TH1D(f"hist_full_{suffix}", f"EEC ({cut_label}); R_{{L}}", nbins, log_bins)
        h["full_ptRL"] = ROOT.TH1D(f"hist_full_ptRL_{suffix}", f"EEC ({cut_label}); <p_{{T}}>R_{{L}} [GeV/c]", nbins, ptrl_log_bins)

        h["rad_wwjetpt"] = ROOT.TH1D(f"hist_rad_{suffix}_wwjetpt", "EEC; R_{L}", nbins, log_bins)
        h["rad_wwradpt"] = ROOT.TH1D(f"hist_rad_{suffix}_wwradpt", "EEC; R_{L}", nbins, log_bins)
        h["rad_ptRL_wwjetpt"] = ROOT.TH1D(f"hist_rad_ptRL_{suffix}_wwjetpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
        h["rad_ptRL_wwradpt"] = ROOT.TH1D(f"hist_rad_ptRL_{suffix}_wwradpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

        for xx in ("AA", "BB", "AB"):
            h[f"{xx}_wwjetpt"] = ROOT.TH1D(f"hist_{xx}_{suffix}_wwjetpt", f"{xx[0]}x{xx[1]}; R_{{L}}", nbins, log_bins)
            h[f"{xx}_wwradpt"] = ROOT.TH1D(f"hist_{xx}_{suffix}_wwradpt", f"{xx[0]}x{xx[1]}; R_{{L}}", nbins, log_bins)
            h[f"{xx}_ptRL_wwjetpt"] = ROOT.TH1D(f"hist_{xx}_ptRL_{suffix}_wwjetpt", f"{xx[0]}x{xx[1]}; <p_{{T}}>R_{{L}} [GeV/c]", nbins, ptrl_log_bins)
            h[f"{xx}_ptRL_wwradpt"] = ROOT.TH1D(f"hist_{xx}_ptRL_{suffix}_wwradpt", f"{xx[0]}x{xx[1]}; <p_{{T}}>R_{{L}} [GeV/c]", nbins, ptrl_log_bins)

        h["radiatorpt"] = ROOT.TH1D(f"radiator_pt_{suffix}", "radiator p_{T}; p_{T,radiator}", len(radpt_bins) - 1, radpt_bins)
        h["radiatorkt"] = ROOT.TH1D(f"radiator_lnkt_{suffix}", "radiator ln k_{T}; ln(k_{T,radiator})", nbins, radkt_bins)
        h["rg"] = ROOT.TH1D(f"hist_rg_{suffix}", "R_{g} = #DeltaR_{AB}; R_{g}; (1/N_{jets}) dN/dR_{g}", nbins, log_bins)

        h["nA_nB"] = ROOT.TH2D(f"hist_nA_nB_{suffix}", "N particles in subjet A vs B; N_{A}; N_{B}", n_max, -0.5, n_max - 0.5, n_max, -0.5, n_max - 0.5)
        h["nTotalUnGroomed"] = ROOT.TH1D(f"hist_nTotalUnGroomed_{suffix}", "N particles in ungroomed jet; N_{total ungroomed}; counts", n_max, -0.5, n_max - 0.5)
        h["nTotalGroomed"] = ROOT.TH1D(f"hist_nTotalGroomed_{suffix}", "N particles in groomed jet; N_{total groomed}; counts", n_max, -0.5, n_max - 0.5)
        h["combAA"] = ROOT.TH1D(f"hist_combAA_{suffix}", "AxA combinations; N_{A}^{2}; counts", 200, -0.5, comb_max - 0.5)
        h["combBB"] = ROOT.TH1D(f"hist_combBB_{suffix}", "BxB combinations; N_{B}^{2}; counts", 200, -0.5, comb_max - 0.5)
        h["combAB"] = ROOT.TH1D(f"hist_combAB_{suffix}", "AxB combinations; N_{A} N_{B}; counts", 200, -0.5, comb_max - 0.5)
        h["combTotal"] = ROOT.TH1D(f"hist_combTotal_{suffix}", "Total combinations; (N_{A}+N_{B})^{2}; counts", 200, -0.5, 4 * comb_max - 0.5)

        return h


    def GetCABHist(self,hist_AA, hist_BB, hist_AB, name="hist_correlation"):
        """
        Calculates AB / sqrt(AA * BB) bin-by-bin.
        """
        # Clone one of the inputs to get the same binning/axes
        print(name)
        hist_CAB = hist_AB.Clone(name)
        hist_CAB.SetTitle("C_{AB};R_{L};AxB / #sqrt{AxA #times BxB}")
        hist_CAB.Reset() # Clear the counts

        for i in range(1, hist_CAB.GetNbinsX() + 1):
            aa = hist_AA.GetBinContent(i)
            bb = hist_BB.GetBinContent(i)
            ab = hist_AB.GetBinContent(i)
            
            sig_aa = hist_AA.GetBinError(i) # error
            sig_bb = hist_BB.GetBinError(i)
            sig_ab = hist_AB.GetBinError(i)

            # Basic check to avoid division by zero or sqrt of negative
            if aa > 0 and bb > 0:
                denom = math.sqrt(aa * bb)
                result = ab / denom
                # print("bin", i, "// aa:", aa, "bb:", bb, "ab:", ab, "==> C_AB:", result)
                
                # Error propagation (Simplified: assumes AA and BB errors are small 
                # or you can use standard Taylor expansion for full propagation)
                # For simplicity, here we just copy the relative error of AB
                # if val_ab != 0:
                #     rel_err = hist_AB.GetBinError(i) / val_ab
                #     hist_CAB.SetBinError(i, result * rel_err)

                # Here is the proper error propagation:
                # A note: Statistical Correlation: Because $AA$, $BB$, and $AB$ are calculated from the same set of jets, they are technically correlated. 
                # If your analysis requires extreme precision (e.g., for a publication), you would typically calculate $R$ for each jet individually, 
                # then find the mean and the error on the mean of $R$ across all jets.
                term_ab = sig_ab / denom
                term_aa = (result * sig_aa) / (2 * aa)
                term_bb = (result * sig_bb) / (2 * bb)
                
                sig_R = np.sqrt(term_ab**2 + term_aa**2 + term_bb**2) # Combine in quadrature

                hist_CAB.SetBinContent(i, result)
                hist_CAB.SetBinError(i, sig_R)
            else:
                hist_CAB.SetBinContent(i, 0)
                hist_CAB.SetBinError(i, 0)
                # print("bin", i, "// aa:", aa, "bb:", bb, "ab:", ab, "==> C_AB: n/a")

        return hist_CAB

    def format_cut_tag(self, cut_mode, cut_value):
        if cut_mode == "sd":
            return f"sd{str(cut_value)}" #f"sd{str(cut_value).replace('.', 'p')}"
        if cut_mode == "maxkt":
            return "maxkt"
        return f"{cut_mode}{str(cut_value)}" #f"{cut_mode}{str(cut_value).replace('.', 'p')}"

    def get_cut_label(self, cut_mode, cut_value):
        if cut_mode == "sd":
            return f"Soft Drop z > {cut_value}"
        if cut_mode == "maxkt":
            return "Max k_{T}"
        return f"{cut_mode} {cut_value}"

    def select_split(self, lund_plane_elements, cut_mode, cut_value):
        if cut_mode == "sd":
            for d in lund_plane_elements:
                if d.z() > cut_value:
                    return d
            return None
        if cut_mode == "maxkt":
            # Only select max kt for kt > 1
            filtered = [d for d in lund_plane_elements if d.kt() > 1.0]
            return max(filtered, key=lambda d: d.kt(), default=None) if filtered else None
        return None

    def FillHists(self, label, sj, hist, weight, hist_ptRL=None, avg_pt=None, sj_B=None):
        # Get constituents for this specific prong
        sj_constituents = fj.sorted_by_pt(sj.constituents())
        
        # Apply your pt threshold (trk_thrd = 1)
        c_select = fj.vectorPJ()
        for c in sj_constituents:
            if c.pt() < self.trk_thrd:
                break
            c_select.append(c)
                
        # Calculate the EEC 
        if label == "AxB":
            sj_constituents_B = fj.sorted_by_pt(sj_B.constituents())
            c_select_B = fj.vectorPJ()
            for c in sj_constituents_B:
                if c.pt() < self.trk_thrd:
                    break
                c_select_B.append(c)

            eec_result = ecorrel.CorrelatorBuilder( c_select, c_select_B, weight, 2, 1, self.dphi_cut, self.deta_cut )
        else:
            eec_result = ecorrel.CorrelatorBuilder( c_select, weight, 2, 1, self.dphi_cut, self.deta_cut )

        # Fill the histograms
        for index in range(eec_result.correlator(2).rs().size()):
            rl_value = eec_result.correlator(2).rs()[index]
            weight_value = eec_result.correlator(2).weights()[index]
            hist.Fill(rl_value, weight_value)
            if hist_ptRL is not None and avg_pt is not None:
                hist_ptRL.Fill(rl_value * avg_pt, weight_value)
            # hist_wwjetpt.Fill(eec_result_wwjetpt.correlator(2).rs()[index], eec_result_wwjetpt.correlator(2).weights()[index])
            # if label != "full":
            #     hist_wwradpt.Fill(eec_result_wwradpt.correlator(2).rs()[index], eec_result_wwradpt.correlator(2).weights()[index])

    def count_constituents(self, sj):
        """Count constituents in a subjet passing the pt threshold (trk_thrd)."""
        sj_constituents = fj.sorted_by_pt(sj.constituents())
        n = 0
        for c in sj_constituents:
            if c.pt() < self.trk_thrd:
                break
            n += 1
        return n

    def load_avg_pt_rad(self):
        path = (f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/"
                f"avg_pt_rad_{self.gen}.json")
        if not os.path.exists(path):
            pinfo(f"ERROR: avg_pt_rad file not found: {path}")
            pinfo("Run precompute_avg_pt_rad.py first. Exiting.")
            sys.exit(1)
        with open(path) as f:
            return json.load(f)
    
    def run(self):
        avg_jet_pts = {}

        # Load precomputed avg_pt_rad, keyed by source_label -> config_tag -> {"avg_pt_rad", "n_jets"}
        avg_rad_data = self.load_avg_pt_rad()

        for i, target_jetpt in enumerate(self.target_jet_pts):
            # if i == 0 and self.gen == "pythia":
            #     continue  # skip 50 GeV for pythia, as it is already done
            out_dir = self.output_dir()
            jp_tag = self.jetpt_tag(target_jetpt)

            root_outfile = ROOT.TFile(
                os.path.join(out_dir,
                    f"jse_preliminary_curves_{self.gen}_{jp_tag}.root"), "RECREATE")
            
            # Choose input parquet filename depending on narrow vs wide
            if self.narrow:
                hi = self.narrow_hi(target_jetpt)
                combined_fname = f"FilteredJetsForAnalysisCombined_{target_jetpt}_{hi}.parquet"
            else:
                combined_fname = "FilteredJetsForAnalysisCombined.parquet"

            if self.gen == "pythia":
                path = (f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
                        f"AnalysisResults/blianggi/jse/pythia_otf/55555648/{target_jetpt}gev/"
                        f"{combined_fname}")
            elif self.gen == "herwig":
                path = (f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
                        f"generation/blianggi/herwiggen/tree_gen/55293842/{target_jetpt}gev/"
                        f"{combined_fname}")

            # if self.gen == "pythia":
            #     path = (f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
            #             f"AnalysisResults/blianggi/jse/pythia_otf/55555648/{target_jetpt}gev/"
            #             f"FilteredJetsForAnalysisCombined.parquet")
            # elif self.gen == "herwig":
            #     path = (f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
            #             f"generation/blianggi/herwiggen/tree_gen/55293842/{target_jetpt}gev/"
            #             f"FilteredJetsForAnalysisCombined.parquet")
            df = pd.read_parquet(path)

            avg_jet_pt = df['jet_pt'].mean()
            avg_jet_pts[target_jetpt] = avg_jet_pt
            print(f"target_jetpt={target_jetpt} average jet pT = {avg_jet_pt:.2f}")

            grouped_jets = df.groupby(['event_id', 'jet_id'])

            # Source label to look up avg_pt_rad for this bin
            if self.narrow:
                source_label = f"{target_jetpt}-{self.narrow_hi(target_jetpt)}"
            else:
                main_hi = {50: 60, 100: 120, 200: 240, 500: 600}[target_jetpt]
                source_label = f"{target_jetpt}-{main_hi}"

            # Book all histograms and per-config bookkeeping up front
            hists = {}          # (cut_mode, cut_value, partontype) -> dict of hists
            count_records = {}  # (cut_mode, cut_value, partontype) -> list
            num_jets = {}       # (cut_mode, cut_value, partontype) -> float
            num_jets_passed = {}
            avg_pt_rad_lookup = {}  # (cut_mode, cut_value, partontype) -> float or None

            for cut_mode, cut_value in self.cut_configs:
                cut_tag = self.format_cut_tag(cut_mode, cut_value)
                for partontype in self.partontypes:
                    key = (cut_mode, cut_value, partontype)
                    hists[key] = self.book_histograms(target_jetpt, cut_mode, cut_value, partontype)
                    count_records[key] = []
                    num_jets[key] = 0.
                    num_jets_passed[key] = 0.
                    # Look up avg_pt_rad for wwrad ptRL weighting
                    config_tag = f"{partontype}_{cut_tag}"
                    apr = None
                    try:
                        apr = avg_rad_data["sources"][source_label][config_tag]["avg_pt_rad"]
                    except (KeyError, TypeError):
                        pwarning(f"avg_pt_rad not found for {source_label} / {config_tag}; wwrad ptRL will fall back to avg_jet_pt")
                    avg_pt_rad_lookup[key] = apr

            # ---- Single pass over all jets ----
            for (event_idx, jet_id), jet_constituents in grouped_jets:
                if event_idx % 1000 == 0:
                    print("event", event_idx)

                jet_parton_pid = jet_constituents['parton_pid'].values[0]
                jet_true_type = self.get_parton_type(jet_parton_pid)

                pj_particles = [fj.PseudoJet(row.c_px, row.c_py, row.c_pz, row.c_e)
                                for row in jet_constituents.itertuples()]
                cs = fj.ClusterSequence(pj_particles, self.jet_def_ca)
                jets = fj.sorted_by_pt(cs.inclusive_jets())
                if not jets:
                    continue
                jet = jets[0]

                lund_gen = fjcontrib.LundGenerator(self.jet_def_ca)
                lund_plane_elements = lund_gen.result(jet)

                # Which partontypes does this jet contribute to?
                active_partontypes = ["inclusive"]
                if jet_true_type in self.partontypes:
                    active_partontypes.append(jet_true_type)

                for cut_mode, cut_value in self.cut_configs:
                    selected_d = self.select_split(lund_plane_elements, cut_mode, cut_value)

                    # num_jets counts all jets seen for this config/partontype (pre-cut)
                    for partontype in active_partontypes:
                        num_jets[(cut_mode, cut_value, partontype)] += 1

                    if selected_d is None:
                        continue

                    parent_radiator = selected_d.pair()
                    subjets = sorted(parent_radiator.pieces(), key=lambda x: x.pt(), reverse=True)
                    if len(subjets) != 2:
                        continue
                    subjet_a, subjet_b = subjets

                    rad_pt = parent_radiator.perp()
                    r_g = selected_d.Delta()
                    ln_kt = np.log(selected_d.kt())

                    n_A = self.count_constituents(subjet_a)
                    n_B = self.count_constituents(subjet_b)
                    n_total_ungroomed = self.count_constituents(jet)
                    n_total_groomed = self.count_constituents(parent_radiator)
                    comb_AA = n_A * n_A
                    comb_BB = n_B * n_B
                    comb_AB = n_A * n_B * 2
                    comb_total = (n_A + n_B) * (n_A + n_B)

                    for partontype in active_partontypes:
                        key = (cut_mode, cut_value, partontype)
                        h = hists[key]
                        num_jets_passed[key] += 1

                        h["jetpt_all"].Fill(jet.perp())
                        h["jetpt_cut"].Fill(jet.perp())
                        h["radiatorpt"].Fill(rad_pt)
                        h["radiatorkt"].Fill(ln_kt)
                        h["rg"].Fill(r_g)

                        h["nA_nB"].Fill(n_A, n_B)
                        h["nTotalUnGroomed"].Fill(n_total_ungroomed)
                        h["nTotalGroomed"].Fill(n_total_groomed)
                        h["combAA"].Fill(comb_AA)
                        h["combBB"].Fill(comb_BB)
                        h["combAB"].Fill(comb_AB)
                        h["combTotal"].Fill(comb_total)

                        if len(count_records[key]) < 100:
                            count_records[key].append({
                                "event_id": int(event_idx), "jet_id": int(jet_id),
                                "n_A": int(n_A), "n_B": int(n_B),
                                "n_total_ungroomed": int(n_total_ungroomed),
                                "n_total_groomed": int(n_total_groomed),
                                "comb_AA": int(comb_AA), "comb_BB": int(comb_BB),
                                "comb_AB": int(comb_AB), "comb_total": int(comb_total),
                            })

                        # avg_pt to use for wwrad ptRL: precomputed avg_pt_rad if available
                        avg_pt_rad = avg_pt_rad_lookup[key]
                        avg_pt_rad_eff = avg_pt_rad if avg_pt_rad is not None else avg_jet_pt

                        # full EEC (weighted by jet pt only)
                        self.FillHists("full", jet, h["full"], jet.perp(),
                                       hist_ptRL=h["full_ptRL"], avg_pt=avg_jet_pt)

                        for label, sj in (("rad", parent_radiator), ("A", subjet_a), ("B", subjet_b)):
                            xx = {"rad": "rad", "A": "AA", "B": "BB"}[label]
                            # wwjetpt: weight by jet pt, ptRL scaled by avg_jet_pt
                            self.FillHists(label, sj, h[f"{xx}_wwjetpt"], jet.perp(),
                                           hist_ptRL=h[f"{xx}_ptRL_wwjetpt"], avg_pt=avg_jet_pt)
                            # wwradpt: weight by radiator pt, ptRL scaled by avg_pt_rad
                            self.FillHists(label, sj, h[f"{xx}_wwradpt"], rad_pt,
                                           hist_ptRL=h[f"{xx}_ptRL_wwradpt"], avg_pt=avg_pt_rad_eff)

                        # AxB
                        self.FillHists("AxB", subjet_a, h["AB_wwjetpt"], jet.perp(),
                                       hist_ptRL=h["AB_ptRL_wwjetpt"], avg_pt=avg_jet_pt, sj_B=subjet_b)
                        self.FillHists("AxB", subjet_a, h["AB_wwradpt"], rad_pt,
                                       hist_ptRL=h["AB_ptRL_wwradpt"], avg_pt=avg_pt_rad_eff, sj_B=subjet_b)

            # ---- Post-processing: normalize, C_AB, write ----
            for cut_mode, cut_value in self.cut_configs:
                cut_tag = self.format_cut_tag(cut_mode, cut_value)
                for partontype in self.partontypes:
                    key = (cut_mode, cut_value, partontype)
                    h = hists[key]
                    npass = num_jets_passed[key]
                    apr = avg_pt_rad_lookup[key]
                    apr_eff = apr if apr is not None else avg_jet_pt

                    self.FormatHist(h["full"], npass, ROOT.kGray)
                    self.FormatHist(h["full_ptRL"], npass, ROOT.kGray, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(h["rad_wwjetpt"], npass, ROOT.kBlack)
                    self.FormatHist(h["rad_ptRL_wwjetpt"], npass, ROOT.kBlack, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(h["rad_wwradpt"], npass, ROOT.kBlack)
                    self.FormatHist(h["rad_ptRL_wwradpt"], npass, ROOT.kBlack, pt_rl=True, avg_pt=apr_eff)

                    self.FormatHist(h["AA_wwjetpt"], npass, ROOT.kBlue)
                    self.FormatHist(h["AA_ptRL_wwjetpt"], npass, ROOT.kBlue, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(h["BB_wwjetpt"], npass, ROOT.kOrange + 7)
                    self.FormatHist(h["BB_ptRL_wwjetpt"], npass, ROOT.kOrange + 7, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(h["AB_wwjetpt"], npass, ROOT.kGreen + 2)
                    self.FormatHist(h["AB_ptRL_wwjetpt"], npass, ROOT.kGreen + 2, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(h["AA_wwradpt"], npass, ROOT.kBlue)
                    self.FormatHist(h["AA_ptRL_wwradpt"], npass, ROOT.kBlue, pt_rl=True, avg_pt=apr_eff)
                    self.FormatHist(h["BB_wwradpt"], npass, ROOT.kOrange + 7)
                    self.FormatHist(h["BB_ptRL_wwradpt"], npass, ROOT.kOrange + 7, pt_rl=True, avg_pt=apr_eff)
                    self.FormatHist(h["AB_wwradpt"], npass, ROOT.kGreen + 2)
                    self.FormatHist(h["AB_ptRL_wwradpt"], npass, ROOT.kGreen + 2, pt_rl=True, avg_pt=apr_eff)

                    self.FormatHist(h["rg"], npass, ROOT.kRed + 1)

                    h["CAB_wwjetpt"] = self.GetCABHist(h["AA_wwjetpt"], h["BB_wwjetpt"], h["AB_wwjetpt"],
                                                       f"CAB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt")
                    h["CAB_wwradpt"] = self.GetCABHist(h["AA_wwradpt"], h["BB_wwradpt"], h["AB_wwradpt"],
                                                       f"CAB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt")
                    h["CAB_ptRL_wwjetpt"] = self.GetCABHist(h["AA_ptRL_wwjetpt"], h["BB_ptRL_wwjetpt"], h["AB_ptRL_wwjetpt"],
                                                            f"CAB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt")
                    h["CAB_ptRL_wwradpt"] = self.GetCABHist(h["AA_ptRL_wwradpt"], h["BB_ptRL_wwradpt"], h["AB_ptRL_wwradpt"],
                                                            f"CAB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt")

                    self.FormatHist(h["CAB_wwjetpt"], -1, ROOT.kPink + 10)
                    self.FormatHist(h["CAB_wwradpt"], -1, ROOT.kViolet + 7)
                    self.FormatHist(h["CAB_ptRL_wwjetpt"], -1, ROOT.kPink + 10)
                    self.FormatHist(h["CAB_ptRL_wwradpt"], -1, ROOT.kViolet + 7)

                    # Write everything
                    write_order = [
                        "jetpt_all", "jetpt_cut", "radiatorpt", "radiatorkt",
                        "full", "full_ptRL",
                        "rad_wwjetpt", "rad_ptRL_wwjetpt", "rad_wwradpt", "rad_ptRL_wwradpt",
                        "AA_wwjetpt", "AA_ptRL_wwjetpt", "BB_wwjetpt", "BB_ptRL_wwjetpt",
                        "AB_wwjetpt", "AB_ptRL_wwjetpt",
                        "AA_wwradpt", "AA_ptRL_wwradpt", "BB_wwradpt", "BB_ptRL_wwradpt",
                        "AB_wwradpt", "AB_ptRL_wwradpt",
                        "CAB_wwjetpt", "CAB_wwradpt", "CAB_ptRL_wwjetpt", "CAB_ptRL_wwradpt",
                        "nA_nB", "nTotalUnGroomed", "nTotalGroomed",
                        "combAA", "combBB", "combAB", "combTotal",
                    ]
                    for hname in write_order:
                        h[hname].Write()
                    if cut_mode == "sd":
                        h["rg"].Write()

                    # Per-jet counts JSON
                    # Per-jet counts JSON
                    json_dir = out_dir
                    json_path = os.path.join(
                        json_dir,
                        f"jse_counts_{self.gen}_{jp_tag}_{partontype}_{cut_tag}.json")
                    recs = count_records[key]
                    summary = {
                        "gen": self.gen, "target_jetpt": target_jetpt,
                        "partontype": partontype, "cut_mode": cut_mode, "cut_value": cut_value,
                        "num_jets": num_jets[key], "num_jets_passed_cut": npass,
                        "avg_pt_rad": apr,
                        "totals": {
                            "sum_n_A": int(sum(r["n_A"] for r in recs)),
                            "sum_n_B": int(sum(r["n_B"] for r in recs)),
                            "sum_n_total_ungroomed": int(sum(r["n_total_ungroomed"] for r in recs)),
                            "sum_n_total_groomed": int(sum(r["n_total_groomed"] for r in recs)),
                            "sum_comb_AA": int(sum(r["comb_AA"] for r in recs)),
                            "sum_comb_BB": int(sum(r["comb_BB"] for r in recs)),
                            "sum_comb_AB": int(sum(r["comb_AB"] for r in recs)),
                            "sum_comb_total": int(sum(r["comb_total"] for r in recs)),
                        },
                        "per_jet": recs,
                    }
                    with open(json_path, "w") as f:
                        json.dump(summary, f, indent=2)
                    print(f"Wrote counts to {json_path}  ({len(recs)} jets)")

            root_outfile.Close()

        avg_txt_path = os.path.join(self.output_dir(), f"jse_avg_jet_pts_{self.gen}.txt")
        with open(avg_txt_path, "w") as avg_file:
            for jetpt in self.target_jet_pts:
                if jetpt in avg_jet_pts:
                    avg_file.write(f"{jetpt} {avg_jet_pts[jetpt]:.6f}\n")

        return avg_jet_pts

def main():
    parser = argparse.ArgumentParser(description="Process jets for JSE analysis")
    parser.add_argument("gen", choices=["pythia", "herwig"], help="Generator to process")
    parser.add_argument("--narrow", action="store_true",
                        help="Use narrower bin samples (jetpt to jetpt*1.1), "
                             "output to rootfiles/narrowerbins1.1/")
    args = parser.parse_args()
    gen = args.gen

    analysis = MyAnalysis(gen, narrow=args.narrow)
    avg = analysis.run()
    print(f"\nAverage jet pts for {gen}:")
    for jetpt in analysis.target_jet_pts:
        print(f"  {analysis.jetpt_tag(jetpt)}: {avg[jetpt]:.6f}")

    # if gen == "pythia":
    #     analysis_pythia = MyAnalysis("pythia")
    #     pythia_avg = analysis_pythia.run()
    #     print("\nAverage jet pts for pythia:")
    #     for jetpt in analysis_pythia.target_jet_pts:
    #         print(f"  jetpt{jetpt}: {pythia_avg[jetpt]:.6f}")
    # elif gen == "herwig":
    #     analysis_herwig = MyAnalysis("herwig")
    #     herwig_avg = analysis_herwig.run()
    #     print("\nAverage jet pts for herwig:")
    #     for jetpt in analysis_herwig.target_jet_pts:
    #         print(f"  jetpt{jetpt}: {herwig_avg[jetpt]:.6f}")

''' RUN AS FOLLOWS
 - Wide bins:
   - Pythia: python process_jets_jse.py pythia
   - Herwig: python process_jets_jse.py herwig
 - Narrow bins (jetpt -> jetpt*1.1, output to rootfiles/narrowerbins1.1/):
   - Pythia: python process_jets_jse.py pythia --narrow
   - Herwig: python process_jets_jse.py herwig --narrow
'''

if __name__ == "__main__":
    main()

