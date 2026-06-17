
import pandas as pd
import fastjet as fj
import fjcontrib
import fjext
import ecorrel

import ROOT

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
    def __init__(self, def_gen):
        # Jet pts
        self.target_jet_pts = [ 50, 100, 200, 500 ]
        self.gen = def_gen
        
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

    def run(self):
        avg_jet_pts = {}
        for i, target_jetpt in enumerate(self.target_jet_pts):
            root_outfile = ROOT.TFile(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/jse_preliminary_curves_{self.gen}_jetpt{target_jetpt}.root", "RECREATE")

            # Load the data
            # path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/{self.target_jet_pts[i]}gev/JetsForAnalysis.parquet"
            # path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/{self.target_jet_pts[i]}gev/{n+1}/JetsForAnalysis.parquet"
            if ( self.gen == "pythia" ):
                path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/53423546/{self.target_jet_pts[i]}gev/FilteredJetsForAnalysisCombined.parquet"
            elif ( self.gen == "herwig" ):
                # path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/storage/herwig/1006458/{self.target_jet_pts[i]}gev/FilteredJetsForAnalysisCombined.parquet"
                path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/herwiggen/tree_gen/54380351/{self.target_jet_pts[i]}gev/FilteredJetsForAnalysisCombined.parquet"
            df = pd.read_parquet(path)

            avg_jet_pt = df['jet_pt'].mean()
            avg_jet_pts[target_jetpt] = avg_jet_pt
            print(f"target_jetpt={target_jetpt} average jet pT = {avg_jet_pt:.2f}")

            # Group by jet_id to process one jet at a time
            grouped_jets = df.groupby(['event_id', 'jet_id']) # grouped_jets = df.groupby('jet_id')
            # print("grouped_jets", print(grouped_jets.head(2)))


            for cut_mode, cut_value in self.cut_configs:

                for partontype in self.partontypes:

                    nbins = 25 #15 #50
                    xmin, xmax = 0.001, 1.0
                    ptrl_xmin, ptrl_xmax = 0.1, 50
                    log_bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
                    ptrl_log_bins = np.logspace(np.log10(ptrl_xmin), np.log10(ptrl_xmax), nbins + 1)
                    pt_bins = np.arange(int(target_jetpt / 2), target_jetpt * 1.2 + 2, dtype=float)
                    radpt_bins = np.linspace(0, int(target_jetpt*1.2+2), int(target_jetpt*1.2+2)+1)
                    radkt_bins = np.linspace(-5, 5, nbins+1)
                    # rg_bins = np.logspace(np.log10(0.001), np.log10(1), nbins + 1)  # theta_g = Delta R / R, range [0, 1]

                    cut_tag = self.format_cut_tag(cut_mode, cut_value)
                    cut_label = self.get_cut_label(cut_mode, cut_value)
                    hist_jetpt_label = f"SD z_{{cut}} = {cut_value}" if cut_mode == "sd" else "Max k_{T} jet p_{T}"

                    hist_jetpt_all = ROOT.TH1D( f"hist_jetpt_all_{partontype}_jetpt{target_jetpt}_{cut_tag}", f"jet p_{{T}} ({cut_label}); p_{{T, jet}}", len(pt_bins) - 1, pt_bins)
                    hist_jetpt_cut = ROOT.TH1D( f"hist_jetpt_{partontype}_jetpt{target_jetpt}_{cut_tag}", f"{hist_jetpt_label}; p_{{T, jet}}", len(pt_bins) - 1, pt_bins)

                    hist_full = ROOT.TH1D( f"hist_full_{partontype}_jetpt{target_jetpt}_{cut_tag}", f"EEC ({cut_label}); R_{{L}}", nbins, log_bins)
                    hist_full_ptRL = ROOT.TH1D( f"hist_full_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}", f"EEC ({cut_label}); <p_{{T}}>R_{{L}} [GeV/c]", nbins, ptrl_log_bins)

                    hist_rad_wwjetpt = ROOT.TH1D( f"hist_rad_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "EEC; R_{L}", nbins, log_bins)
                    hist_rad_wwradpt = ROOT.TH1D( f"hist_rad_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "EEC; R_{L}", nbins, log_bins)
                    hist_rad_ptRL_wwjetpt = ROOT.TH1D( f"hist_rad_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                    hist_rad_ptRL_wwradpt = ROOT.TH1D( f"hist_rad_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                    hist_AA_wwjetpt = ROOT.TH1D( f"hist_AA_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "AxA; R_{L}", nbins, log_bins)
                    hist_BB_wwjetpt = ROOT.TH1D( f"hist_BB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "BxB", nbins, log_bins)
                    hist_AB_wwjetpt = ROOT.TH1D( f"hist_AB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "AxB", nbins, log_bins)
                    hist_AA_ptRL_wwjetpt = ROOT.TH1D( f"hist_AA_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "AxA; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                    hist_BB_ptRL_wwjetpt = ROOT.TH1D( f"hist_BB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "BxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                    hist_AB_ptRL_wwjetpt = ROOT.TH1D( f"hist_AB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt", "AxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                    hist_AA_wwradpt = ROOT.TH1D( f"hist_AA_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "AxA; R_{L}", nbins, log_bins)
                    hist_BB_wwradpt = ROOT.TH1D( f"hist_BB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "BxB", nbins, log_bins)
                    hist_AB_wwradpt = ROOT.TH1D( f"hist_AB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "AxB", nbins, log_bins)
                    hist_AA_ptRL_wwradpt = ROOT.TH1D( f"hist_AA_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "AxA; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                    hist_BB_ptRL_wwradpt = ROOT.TH1D( f"hist_BB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "BxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                    hist_AB_ptRL_wwradpt = ROOT.TH1D( f"hist_AB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt", "AxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                    hist_radiatorpt = ROOT.TH1D( f"radiator_pt_{partontype}_jetpt{target_jetpt}_{cut_tag}", "radiator p_{T}; p_{T,radiator}", len(radpt_bins) - 1, radpt_bins)
                    hist_radiatorkt = ROOT.TH1D( f"radiator_lnkt_{partontype}_jetpt{target_jetpt}_{cut_tag}", "radiator ln k_{T}; ln(k_{T,radiator})", nbins, radkt_bins)

                    hist_rg = ROOT.TH1D(f"hist_rg_{partontype}_jetpt{target_jetpt}_{cut_tag}", "R_{g} = #DeltaR_{AB}; R_{g}; (1/N_{jets}) dN/dR_{g}", nbins, log_bins)

                    # --- Particle / combination counting setup ---
                    n_max = 60  # adjust if subjets can have more particles
                    hist_nA_nB = ROOT.TH2D(f"hist_nA_nB_{partontype}_jetpt{target_jetpt}_{cut_tag}", "N particles in subjet A vs B; N_{A}; N_{B}", n_max, -0.5, n_max - 0.5, n_max, -0.5, n_max - 0.5)
                    hist_nTotalUnGroomed = ROOT.TH1D(f"hist_nTotalUnGroomed_{partontype}_jetpt{target_jetpt}_{cut_tag}", "N particles in ungroomed jet; N_{total ungroomed}; counts", n_max, -0.5, n_max - 0.5)
                    hist_nTotalGroomed = ROOT.TH1D(f"hist_nTotalGroomed_{partontype}_jetpt{target_jetpt}_{cut_tag}", "N particles in groomed jet; N_{total groomed}; counts", n_max, -0.5, n_max - 0.5)

                    comb_max = n_max * n_max
                    hist_combAA = ROOT.TH1D(f"hist_combAA_{partontype}_jetpt{target_jetpt}_{cut_tag}", "AxA combinations; N_{A}^{2}; counts", 200, -0.5, comb_max - 0.5)
                    hist_combBB = ROOT.TH1D(f"hist_combBB_{partontype}_jetpt{target_jetpt}_{cut_tag}", "BxB combinations; N_{B}^{2}; counts", 200, -0.5, comb_max - 0.5)
                    hist_combAB = ROOT.TH1D(f"hist_combAB_{partontype}_jetpt{target_jetpt}_{cut_tag}", "AxB combinations; N_{A} N_{B}; counts", 200, -0.5, comb_max - 0.5)
                    hist_combTotal = ROOT.TH1D(f"hist_combTotal_{partontype}_jetpt{target_jetpt}_{cut_tag}", "Total combinations; (N_{A}+N_{B})^{2}; counts", 200, -0.5, 4 * comb_max - 0.5)

                    # Per-jet records for this configuration
                    count_records = []

                    num_jets = 0.
                    num_jets_passed_cut = 0.


                    # Loop over jets
                    for (event_idx, jet_id), jet_constituents in grouped_jets:

                        if event_idx % 1000 == 0:
                            print("event", event_idx)

                        jet_pt = jet_constituents['jet_pt'].iloc[0]
                        # print("jet #", jet_id, "with jet pt =", jet_pt)

                        parton_pid_vals = jet_constituents['parton_pid'].values
                        px = jet_constituents['c_px'].values
                        py = jet_constituents['c_py'].values
                        pz = jet_constituents['c_pz'].values
                        energy = jet_constituents['c_e'].values
                        n_jet_constituents = len(jet_constituents)

                        jet_parton_pid = parton_pid_vals[0]
                        jet_parton_type = self.get_parton_type(jet_parton_pid)
                        if partontype == "inclusive":
                            jet_parton_type = partontype
                        if jet_parton_type != partontype: # skip the jets that are not from the correct parton
                            continue


                        # Convert constituents to FastJet PseudoVectors
                        pj_particles = [ fj.PseudoJet(row.c_px, row.c_py, row.c_pz, row.c_e) for row in jet_constituents.itertuples() ]
                        
                        # Re-cluster with C/A algorithm (essential for Lund Plane)
                        cs = fj.ClusterSequence(pj_particles, self.jet_def_ca)
                        
                        # Get the C/A jet (usually the one with the highest pt)
                        jets = fj.sorted_by_pt(cs.inclusive_jets())
                        if not jets: continue
                        # if len(jets) > 1:
                        #     print(f"WARNING: {len(jets)} C/A jets from one anti-kT jet, stored pt={jet_constituents['jet_pt'].iloc[0]:.2f}, leading CA pt={jets[0].perp():.2f}")
                        jet = jets[0]
                        num_jets += 1
                        hist_jetpt_all.Fill(jet.perp()) # this is the C/A jet pt, for anti-kt, use jet_pt

                        # Generate the Lund Plane
                        # This builds the tree of all declusterings in the C/A history
                        lund_gen = fjcontrib.LundGenerator(self.jet_def_ca) #lp.LundGenerator(jet_def)
                        lund_plane_elements = lund_gen.result(jet) # lund_gen(jet)


                        # Find the selected splitting according to the current cut mode
                        selected_d = self.select_split(lund_plane_elements, cut_mode, cut_value)
                        if selected_d is None:
                            continue

                        parent_radiator = selected_d.pair()
                        subjets = sorted(parent_radiator.pieces(), key=lambda x: x.pt(), reverse=True)
                        if len(subjets) == 2:
                            subjet_a, subjet_b = subjets
                        else:
                            print("This PseudoJet has no parents (it's a single particle).")
                            continue

                        num_jets_passed_cut += 1
                        hist_jetpt_cut.Fill(jet.perp()) # this is the C/A jet pt, for anti-kt, use jet_pt
                        hist_radiatorpt.Fill(parent_radiator.perp()) # equal to selected_d.pair().perp()
                        hist_radiatorkt.Fill(np.log(selected_d.kt()))

                        # Calculate and fill r_g = Delta R_{AB}  # selected_d.Delta() gives the angle between the two prongs in the Lund plane
                        r_g = selected_d.Delta() #/ self.jet_R
                        hist_rg.Fill(r_g)

                        # --- Count particles and combinations for this configuration ---
                        n_A = self.count_constituents(subjet_a)
                        n_B = self.count_constituents(subjet_b)
                        n_total_ungroomed = self.count_constituents(jet)
                        n_total_groomed = self.count_constituents(parent_radiator)

                        # Combinations (N^2 convention, matching ordered self-pairs from CorrelatorBuilder)
                        comb_AA = n_A * n_A
                        comb_BB = n_B * n_B
                        comb_AB = n_A * n_B * 2
                        comb_total = (n_A + n_B) * (n_A + n_B)

                        # Fill aggregate histograms
                        hist_nA_nB.Fill(n_A, n_B)
                        hist_nTotalUnGroomed.Fill(n_total_ungroomed)
                        hist_nTotalGroomed.Fill(n_total_groomed)
                        hist_combAA.Fill(comb_AA)
                        hist_combBB.Fill(comb_BB)
                        hist_combAB.Fill(comb_AB)
                        hist_combTotal.Fill(comb_total)

                        # Store per-jet record
                        count_records.append({
                            "event_id": int(event_idx),
                            "jet_id": int(jet_id),
                            "n_A": int(n_A),
                            "n_B": int(n_B),
                            "n_total_ungroomed": int(n_total_ungroomed),
                            "n_total_groomed": int(n_total_groomed),
                            "comb_AA": int(comb_AA),
                            "comb_BB": int(comb_BB),
                            "comb_AB": int(comb_AB),
                            "comb_total": int(comb_total),
                        })

                        # Get EEC of full, AxA and BxB
                        for label, sj, hist_wwjetpt, hist_wwradpt, hist_ptRL_wwjetpt, hist_ptRL_wwradpt in [
                            ("full", jet, hist_full, None, hist_full_ptRL, None),
                            ("rad", parent_radiator, hist_rad_wwjetpt, hist_rad_wwradpt, hist_rad_ptRL_wwjetpt, hist_rad_ptRL_wwradpt),
                            ("A", subjet_a, hist_AA_wwjetpt, hist_AA_wwradpt, hist_AA_ptRL_wwjetpt, hist_AA_ptRL_wwradpt),
                            ("B", subjet_b, hist_BB_wwjetpt, hist_BB_wwradpt, hist_BB_ptRL_wwjetpt, hist_BB_ptRL_wwradpt)
                        ]:
                            self.FillHists(label, sj, hist_wwjetpt, jet.perp(), hist_ptRL=hist_ptRL_wwjetpt, avg_pt=avg_jet_pt)
                            if label != "full":
                                self.FillHists(label, sj, hist_wwradpt, parent_radiator.perp(), hist_ptRL=hist_ptRL_wwradpt, avg_pt=avg_jet_pt)

                        # Now do AxB
                        self.FillHists("AxB", subjet_a, hist_AB_wwjetpt, jet.perp(), hist_ptRL=hist_AB_ptRL_wwjetpt, avg_pt=avg_jet_pt, sj_B=subjet_b)
                        self.FillHists("AxB", subjet_a, hist_AB_wwradpt, parent_radiator.perp(), hist_ptRL=hist_AB_ptRL_wwradpt, avg_pt=avg_jet_pt, sj_B=subjet_b)
                        # if event_idx > 200: #jet_id > 10:
                        #     break #TODO: get rid of after testing
                        
                    # Normalize and format all curves       
                    self.FormatHist(hist_full, num_jets_passed_cut, ROOT.kGray)
                    self.FormatHist(hist_full_ptRL, num_jets_passed_cut, ROOT.kGray, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(hist_rad_wwjetpt, num_jets_passed_cut, ROOT.kBlack)
                    self.FormatHist(hist_rad_ptRL_wwjetpt, num_jets_passed_cut, ROOT.kBlack, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(hist_rad_wwradpt, num_jets_passed_cut, ROOT.kBlack)
                    self.FormatHist(hist_rad_ptRL_wwradpt, num_jets_passed_cut, ROOT.kBlack, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(hist_AA_wwjetpt, num_jets_passed_cut, ROOT.kBlue)
                    self.FormatHist(hist_AA_ptRL_wwjetpt, num_jets_passed_cut, ROOT.kBlue, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(hist_BB_wwjetpt, num_jets_passed_cut, ROOT.kOrange+7)
                    self.FormatHist(hist_BB_ptRL_wwjetpt, num_jets_passed_cut, ROOT.kOrange+7, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(hist_AB_wwjetpt, num_jets_passed_cut, ROOT.kGreen+2)
                    self.FormatHist(hist_AB_ptRL_wwjetpt, num_jets_passed_cut, ROOT.kGreen+2, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(hist_AA_wwradpt, num_jets_passed_cut, ROOT.kBlue)
                    self.FormatHist(hist_AA_ptRL_wwradpt, num_jets_passed_cut, ROOT.kBlue, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(hist_BB_wwradpt, num_jets_passed_cut, ROOT.kOrange+7)
                    self.FormatHist(hist_BB_ptRL_wwradpt, num_jets_passed_cut, ROOT.kOrange+7, pt_rl=True, avg_pt=avg_jet_pt)
                    self.FormatHist(hist_AB_wwradpt, num_jets_passed_cut, ROOT.kGreen+2)
                    self.FormatHist(hist_AB_ptRL_wwradpt, num_jets_passed_cut, ROOT.kGreen+2, pt_rl=True, avg_pt=avg_jet_pt)

                    self.FormatHist(hist_rg, num_jets_passed_cut, ROOT.kRed+1) #TODO: do this?
                    
                    # Calculate C_AB for both RL and <pT>RL
                    hist_CAB_wwjetpt = self.GetCABHist(hist_AA_wwjetpt, hist_BB_wwjetpt, hist_AB_wwjetpt, f"CAB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt")
                    hist_CAB_wwradpt = self.GetCABHist(hist_AA_wwradpt, hist_BB_wwradpt, hist_AB_wwradpt, f"CAB_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt")
                    hist_CAB_ptRL_wwjetpt = self.GetCABHist(hist_AA_ptRL_wwjetpt, hist_BB_ptRL_wwjetpt, hist_AB_ptRL_wwjetpt, f"CAB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwjetpt")
                    hist_CAB_ptRL_wwradpt = self.GetCABHist(hist_AA_ptRL_wwradpt, hist_BB_ptRL_wwradpt, hist_AB_ptRL_wwradpt, f"CAB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_tag}_wwradpt")

                    self.FormatHist(hist_CAB_wwjetpt, -1, ROOT.kPink+10)
                    self.FormatHist(hist_CAB_wwradpt, -1, ROOT.kViolet+7)
                    self.FormatHist(hist_CAB_ptRL_wwjetpt, -1, ROOT.kPink+10)
                    self.FormatHist(hist_CAB_ptRL_wwradpt, -1, ROOT.kViolet+7)

                    for i in range(1, hist_CAB_wwjetpt.GetNbinsX() + 1):
                        con = hist_CAB_wwjetpt.GetBinContent(i)
                        # print("check bin", i, "==> C_AB:", con)
                
                    # Write to root file
                    hist_jetpt_all.Write()
                    hist_jetpt_cut.Write()
                    hist_radiatorpt.Write()
                    hist_radiatorkt.Write()
                    hist_full.Write()
                    hist_full_ptRL.Write()
                    hist_rad_wwjetpt.Write()
                    hist_rad_ptRL_wwjetpt.Write()
                    hist_rad_wwradpt.Write()
                    hist_rad_ptRL_wwradpt.Write()
                    hist_AA_wwjetpt.Write()
                    hist_AA_ptRL_wwjetpt.Write()
                    hist_BB_wwjetpt.Write()
                    hist_BB_ptRL_wwjetpt.Write()
                    hist_AB_wwjetpt.Write()
                    hist_AB_ptRL_wwjetpt.Write()
                    hist_AA_wwradpt.Write()
                    hist_AA_ptRL_wwradpt.Write()
                    hist_BB_wwradpt.Write()
                    hist_BB_ptRL_wwradpt.Write()
                    hist_AB_wwradpt.Write()
                    hist_AB_ptRL_wwradpt.Write()
                    hist_CAB_wwjetpt.Write()
                    hist_CAB_wwradpt.Write()
                    hist_CAB_ptRL_wwjetpt.Write()
                    hist_CAB_ptRL_wwradpt.Write()
                    if cut_mode == "sd":
                        hist_rg.Write()
                    hist_nA_nB.Write()
                    hist_nTotalUnGroomed.Write()
                    hist_nTotalGroomed.Write()
                    hist_combAA.Write()
                    hist_combBB.Write()
                    hist_combAB.Write()
                    hist_combTotal.Write()

                    # --- Save per-jet counts to JSON ---
                    json_dir = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles"
                    json_path = os.path.join(
                        json_dir,
                        f"jse_counts_{self.gen}_jetpt{target_jetpt}_{partontype}_{cut_tag}.json"
                    )
                    summary = {
                        "gen": self.gen,
                        "target_jetpt": target_jetpt,
                        "partontype": partontype,
                        "cut_mode": cut_mode,
                        "cut_value": cut_value,
                        "num_jets": num_jets,
                        "num_jets_passed_cut": num_jets_passed_cut,
                        "totals": {
                            "sum_n_A": int(sum(r["n_A"] for r in count_records)),
                            "sum_n_B": int(sum(r["n_B"] for r in count_records)),
                            "sum_n_total_ungroomed": int(sum(r["n_total_ungroomed"] for r in count_records)),
                            "sum_n_total_groomed": int(sum(r["n_total_groomed"] for r in count_records)),
                            "sum_comb_AA": int(sum(r["comb_AA"] for r in count_records)),
                            "sum_comb_BB": int(sum(r["comb_BB"] for r in count_records)),
                            "sum_comb_AB": int(sum(r["comb_AB"] for r in count_records)),
                            "sum_comb_total": int(sum(r["comb_total"] for r in count_records)),
                        },
                        "per_jet": count_records,
                    }
                    with open(json_path, "w") as f:
                        json.dump(summary, f, indent=2)
                    print(f"Wrote counts to {json_path}  ({len(count_records)} jets)")

            root_outfile.Close()

        avg_txt_path = f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/jse_avg_jet_pts_{self.gen}.txt"
        with open(avg_txt_path, "w") as avg_file:
            for jetpt in self.target_jet_pts:
                avg_file.write(f"{jetpt} {avg_jet_pts[jetpt]:.6f}\n")

        return avg_jet_pts

def main():
    # analysis_pythia = MyAnalysis("pythia")
    # pythia_avg = analysis_pythia.run()

    analysis_herwig = MyAnalysis("herwig")
    herwig_avg = analysis_herwig.run()

    # print("\nAverage jet pts for pythia:") #comment this out too if not running pythia
    # for jetpt in analysis_pythia.target_jet_pts:
    #     print(f"  jetpt{jetpt}: {pythia_avg[jetpt]:.6f}")

    print("\nAverage jet pts for herwig:")
    for jetpt in analysis_herwig.target_jet_pts:
        print(f"  jetpt{jetpt}: {herwig_avg[jetpt]:.6f}")
    # analysis.run("pythia")
    # analysis.run("herwig")



if __name__ == "__main__":
    main()

