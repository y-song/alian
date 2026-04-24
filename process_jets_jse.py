
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
    def __init__(self):
        # Jet pts
        self.target_jet_pts = [ 50, 100, 200, 500]
        self.partontypes = [ "inclusive", "quark", "gluon"]
        # num_files_to_parse = 1 #100

        # Jet Definitions
        self.jet_R = 0.4
        self.jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, self.jet_R)

        # SD definitions
        self.z_cuts = [ 0.1 ]

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

    def FormatHist(self, hist, norm_factor, color):
        hist.Scale(1/norm_factor, "width")
        hist.SetLineColor(color)
        hist.SetLineWidth(2)

    def GetCABHist(self,hist_AA, hist_BB, hist_AB, name="hist_correlation"):
        """
        Calculates AB / sqrt(AA * BB) bin-by-bin.
        """
        # Clone one of the inputs to get the same binning/axes
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

        return hist_CAB

    def FillHists(self, label, sj, hist, weight, sj_B=None):
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
            # eec_result_wwjetpt = ecorrel.CorrelatorBuilder( c_select, c_select_B, jet.perp(), 2, 1, self.dphi_cut, self.deta_cut )
            # eec_result_wwradpt = ecorrel.CorrelatorBuilder( c_select, c_select_B, parent_radiator.perp(), 2, 1, self.dphi_cut, self.deta_cut ) #sj.perp()
        else:
            eec_result = ecorrel.CorrelatorBuilder( c_select, weight, 2, 1, self.dphi_cut, self.deta_cut )
            # eec_result_wwjetpt = ecorrel.CorrelatorBuilder( c_select, jet.perp(), 2, 1, self.dphi_cut, self.deta_cut )
            # if label != "full": 
            #     eec_result_wwradpt = ecorrel.CorrelatorBuilder( c_select, parent_radiator.perp(), 2, 1, self.dphi_cut, self.deta_cut ) #sj.perp()

        # Fill the histograms
        for index in range(eec_result.correlator(2).rs().size()):
            hist.Fill(eec_result.correlator(2).rs()[index], eec_result.correlator(2).weights()[index])
            # hist_wwjetpt.Fill(eec_result_wwjetpt.correlator(2).rs()[index], eec_result_wwjetpt.correlator(2).weights()[index])
            # if label != "full":
            #     hist_wwradpt.Fill(eec_result_wwradpt.correlator(2).rs()[index], eec_result_wwradpt.correlator(2).weights()[index])

    def run(self):
        root_outfile = ROOT.TFile("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/jse_preliminary_curves.root", "RECREATE")

        for i, target_jetpt in enumerate(self.target_jet_pts):

            # Load the data
            # path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/{self.target_jet_pts[i]}gev/JetsForAnalysis.parquet"
            path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/{self.target_jet_pts[i]}gev/FilteredJetsForAnalysisCombined.parquet"
            # path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/{self.target_jet_pts[i]}gev/{n+1}/JetsForAnalysis.parquet"
            df = pd.read_parquet(path)

            # Group by jet_id to process one jet at a time
            grouped_jets = df.groupby(['event_id', 'jet_id']) # grouped_jets = df.groupby('jet_id')
            # print("grouped_jets", print(grouped_jets.head(2)))


            for z_cut in self.z_cuts:

                for partontype in self.partontypes:

                    # Make a canvas
                    canvas_wwjetpt = ROOT.TCanvas("canvas_wwjetpt", f"Subjet EECs: jet p_T = {target_jetpt}, z_cut = {z_cut}, weight=p_T,1p_T,1 / p_T,jet^2", 800, 600)
                    canvas_wwjetpt.SetLogx()

                    canvas_wwradpt = ROOT.TCanvas("canvas_wwradpt", f"Subjet EECs: jet p_T = {target_jetpt}, z_cut = {z_cut}, weight=p_T,1p_T,1 / p_T,radiator^2", 800, 600)
                    canvas_wwradpt.SetLogx()

                    can_CAB = ROOT.TCanvas("can_CAB", "C_{AB}", 800, 600)
                    can_CAB.SetLogx()

                    can_radpt = ROOT.TCanvas("can_radpt", "Radiator p_{T}", 800, 600)
                    can_radkt = ROOT.TCanvas("can_radkt", "Radiator k_{T}", 800, 600)
                    # can_radkt.SetLogx()

                    nbins = 50
                    xmin, xmax = 0.01, 1.0
                    log_bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

                    # Initialize ROOT histograms // ww = weight with // radpt = radiator pt
                    hist_full = ROOT.TH1D(f"hist_full_{partontype}_jetpt{target_jetpt}_zcut{z_cut}", "EEC; R_{L}", nbins, log_bins)

                    hist_AA_wwjetpt = ROOT.TH1D(f"hist_AA_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwjetpt", "AxA; R_{L}", nbins, log_bins)
                    hist_BB_wwjetpt = ROOT.TH1D(f"hist_BB_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwjetpt", "BxB", nbins, log_bins)
                    hist_AB_wwjetpt = ROOT.TH1D(f"hist_AB_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwjetpt", "AxB", nbins, log_bins)

                    hist_AA_wwradpt = ROOT.TH1D(f"hist_AA_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwradpt", "AxA; R_{L}", nbins, log_bins)
                    hist_BB_wwradpt = ROOT.TH1D(f"hist_BB_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwradpt", "BxB", nbins, log_bins)
                    hist_AB_wwradpt = ROOT.TH1D(f"hist_AB_{partontype}_jetpt{target_jetpt}_zcut{z_cut}_wwradpt", "AxB", nbins, log_bins)

                    radpt_bins = np.linspace(0, target_jetpt, target_jetpt+1)
                    radkt_bins = np.linspace(-5, 5, nbins+1) #np.logspace(np.log10(0.1), np.log10(target_jetpt), nbins+1) #TODO: fille this in!!
                    hist_radiatorpt = ROOT.TH1D(f"radiator_pt_{partontype}_jetpt{target_jetpt}_zcut{z_cut}", "radiator p_{T}; p_{T,radiator}", target_jetpt, radpt_bins)
                    hist_radiatorkt = ROOT.TH1D(f"ln_radiator_kt_{partontype}_jetpt{target_jetpt}_zcut{z_cut}", "ln radiator k_{T}; ln(k_{T,radiator})", nbins, radkt_bins)

                    num_jets = 0.
                    num_jets_passed_SD = 0.


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
                        jet = jets[0]
                        num_jets += 1

                        # # Select jet pt --> this is done in Filtered file
                        # if not (jet.perp() >= target_jetpt-0.5 and jet.perp() < target_jetpt+0.5):
                        #     continue

                        # Generate the Lund Plane
                        # This builds the tree of all declusterings in the C/A history
                        lund_gen = fjcontrib.LundGenerator(self.jet_def_ca) #lp.LundGenerator(jet_def)
                        lund_plane_elements = lund_gen.result(jet) # lund_gen(jet)


                        # Find the Soft Drop prong (z > 0.1)
                        # We walk down the primary declustering sequence (widest angle first)
                        for d in lund_plane_elements:
                            # The LundPlane objects provide .z(), .Delta(), .kt(), etc.
                            if d.z() > z_cut:
                                # sd_info["z"] = d.z()
                                # sd_info["theta"] = d.Delta() # Angular separation
                                # sd_info["kt"] = d.kt()       # Relative transverse momentum
                                parent_radiator = d.pair() # Gives the parent of the prongs that passed Soft Drop
                                subjets = sorted(parent_radiator.pieces(), key=lambda x: x.pt(), reverse=True) #to ensure subjet_a is the harder subjet
                                # TODO: compare parent_radiator.pieces() with d.pieces(). are these the same??? If not, which is correct below?
                                if len(subjets) == 2:
                                    subjet_a, subjet_b = subjets
                                    # print(f"Subjet 1 pT: {subjet_a.pt()}, Subjet 2 pT: {subjet_b.pt()}")
                                else:
                                    print("This PseudoJet has no parents (it's a single particle).")
                                
                                num_jets_passed_SD += 1

                                # Add to jet histograms
                                hist_radiatorpt.Fill(parent_radiator.perp())
                                hist_radiatorkt.Fill( np.log(d.kt()) ) # TODO: is this right? should it be parent_radiator.perp()? Or one of the subjets?? prob not that.


                                # Get EEC of AxA and BxB
                                for label, sj, hist_wwjetpt, hist_wwradpt in [("full", jet, hist_full, None), ("A", subjet_a, hist_AA_wwjetpt, hist_AA_wwradpt), ("B", subjet_b, hist_BB_wwjetpt, hist_BB_wwradpt)]:
                                    # print("Filling for combo", label)
                                    # # Get constituents for this specific prong
                                    # sj_constituents = fj.sorted_by_pt(sj.constituents())
                                    
                                    # # Apply your pt threshold (trk_thrd = 1)
                                    # c_select = fj.vectorPJ()
                                    # for c in sj_constituents:
                                    #     if c.pt() < trk_thrd:
                                    #         break
                                    #     c_select.append(c)
                                            
                                    # # Calculate the EEC for this subjet
                                    # eec_result_wwjetpt = ecorrel.CorrelatorBuilder( c_select, jet.perp(), 2, 1, dphi_cut, deta_cut )
                                    # if label != "full": 
                                    #     eec_result_wwradpt = ecorrel.CorrelatorBuilder( c_select, parent_radiator.perp(), 2, 1, dphi_cut, deta_cut ) #sj.perp()

                                    # for index in range(eec_result_wwjetpt.correlator(2).rs().size()):
                                    #     hist_wwjetpt.Fill(eec_result_wwjetpt.correlator(2).rs()[index], eec_result_wwjetpt.correlator(2).weights()[index])
                                    #     if label != "full": 
                                    #         hist_wwradpt.Fill(eec_result_wwradpt.correlator(2).rs()[index], eec_result_wwradpt.correlator(2).weights()[index])

                                    self.FillHists(label, sj, hist_wwjetpt, jet.perp())
                                    if label != "full": 
                                        self.FillHists(label, sj, hist_wwradpt, parent_radiator.perp())

                                # Now do AxB
                                self.FillHists("AxB", subjet_a, hist_AB_wwjetpt, jet.perp(), sj_B=subjet_b)
                                self.FillHists("AxB", subjet_a, hist_AB_wwradpt, parent_radiator.perp(), sj_B=subjet_b)
                                # sj_A_const = fj.sorted_by_pt(subjet_a.constituents())
                                # sj_B_const = fj.sorted_by_pt(subjet_b.constituents())

                                # c_select_A = fj.vectorPJ()
                                # c_select_B = fj.vectorPJ()
                                # for c in sj_A_const:
                                #     if c.pt() < trk_thrd:
                                #         break
                                #     c_select_A.append(c)
                                # for c in sj_B_const:
                                #     if c.pt() < trk_thrd:
                                #         break
                                #     c_select_B.append(c)

                                # eec_result_wwjetpt = ecorrel.CorrelatorBuilder( c_select_A, c_select_B, jet.perp(), 2, 1, dphi_cut, deta_cut )
                                # eec_result_wwradpt = ecorrel.CorrelatorBuilder( c_select_A, c_select_B, parent_radiator.perp(), 2, 1, dphi_cut, deta_cut ) #sj.perp()

                                # for index in range(eec_result_wwjetpt.correlator(2).rs().size()):
                                #     hist_AB_wwjetpt.Fill(eec_result_wwjetpt.correlator(2).rs()[index], eec_result_wwjetpt.correlator(2).weights()[index])
                                #     hist_AB_wwradpt.Fill(eec_result_wwradpt.correlator(2).rs()[index], eec_result_wwradpt.correlator(2).weights()[index])
                                

                                break # Soft Drop stops at the first splitting that passes
                        # if event_idx > 200: #jet_id > 10:
                        #     break #TODO: get rid of after testing
                        
                    # Normalize and format all curves       
                    self.FormatHist(hist_full, num_jets_passed_SD, ROOT.kBlack)

                    self.FormatHist(hist_AA_wwjetpt, num_jets_passed_SD, ROOT.kBlue)
                    self.FormatHist(hist_BB_wwjetpt, num_jets_passed_SD, ROOT.kOrange+7)
                    self.FormatHist(hist_AB_wwjetpt, num_jets_passed_SD, ROOT.kGreen+2)

                    self.FormatHist(hist_AA_wwradpt, num_jets_passed_SD, ROOT.kBlue)
                    self.FormatHist(hist_BB_wwradpt, num_jets_passed_SD, ROOT.kOrange+7)
                    self.FormatHist(hist_AB_wwradpt, num_jets_passed_SD, ROOT.kGreen+2)


                    # Add a Legend
                    legend = ROOT.TLegend(0.7, 0.5, 0.9, 0.65)
                    legend.AddEntry(hist_full, f"all {partontype} jets that passed SD", "l")
                    legend.AddEntry(hist_AA_wwjetpt, "AxA", "l")
                    legend.AddEntry(hist_BB_wwjetpt, "BxB", "l")
                    legend.AddEntry(hist_AB_wwjetpt, "AxB", "l")


                    canvas_wwjetpt.cd()
                    hist_full.Draw("HIST")
                    hist_AA_wwjetpt.Draw("HIST SAME")
                    hist_BB_wwjetpt.Draw("HIST SAME")
                    hist_AB_wwjetpt.Draw("HIST SAME")
                    legend.Draw()
                    canvas_wwjetpt.SaveAs(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/subjet_eec_comparison_{partontype}_jetpt{target_jetpt}_R0.4_sd{z_cut}_wwjetpt.pdf")

                    canvas_wwradpt.cd()
                    hist_full.Draw("HIST")
                    hist_AA_wwradpt.Draw("HIST SAME")
                    hist_BB_wwradpt.Draw("HIST SAME")
                    hist_AB_wwradpt.Draw("HIST SAME")
                    legend.Draw()
                    canvas_wwradpt.SaveAs(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/subjet_eec_comparison_{partontype}_jetpt{target_jetpt}_R0.4_sd{z_cut}_wwradpt.pdf")

                    
                    # Calulate C_AB
                    can_CAB.cd()
                    # Usage in your main script:
                    hist_CAB_wwjetpt = self.GetCABHist(hist_AA_wwjetpt, hist_BB_wwjetpt, hist_AB_wwjetpt, "CAB_wwjetpt")
                    hist_CAB_wwradpt = self.GetCABHist(hist_AA_wwradpt, hist_BB_wwradpt, hist_AB_wwradpt, "CAB_wwradpt")

                    self.FormatHist(hist_CAB_wwjetpt, 1, ROOT.kPink+10)
                    self.FormatHist(hist_CAB_wwradpt, 1, ROOT.kViolet+7)
                    legend_CAB = ROOT.TLegend(0.2, 0.5, 0.4, 0.65)
                    legend_CAB.AddEntry(hist_CAB_wwjetpt, "weight uses jet pt", "l")
                    legend_CAB.AddEntry(hist_CAB_wwradpt, "weight uses rad pt", "l")

                    hist_CAB_wwjetpt.Draw("HIST")
                    hist_CAB_wwradpt.Draw("HIST SAME")
                    legend_CAB.Draw()
                    can_CAB.SaveAs(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/CAB_{partontype}_jetpt{target_jetpt}_R0.4_sd{z_cut}.pdf")

                    # Plot jet level info
                    can_radpt.cd()
                    hist_radiatorpt.Draw()
                    can_radpt.SaveAs(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/radiator_pt_{partontype}_jetpt{target_jetpt}_R0.4_sd{z_cut}.pdf")
                    
                    can_radkt.cd()
                    hist_radiatorkt.Draw()
                    can_radkt.SaveAs(f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/radiator_kt_{partontype}_jetpt{target_jetpt}_R0.4_sd{z_cut}.pdf")
                    
                    # Write to root file
                    hist_radiatorpt.Write()
                    hist_radiatorkt.Write()
                    hist_full.Write()
                    hist_AA_wwjetpt.Write()
                    hist_BB_wwjetpt.Write()
                    hist_AB_wwjetpt.Write()
                    hist_AA_wwradpt.Write()
                    hist_BB_wwradpt.Write()
                    hist_AB_wwradpt.Write()
                    hist_CAB_wwjetpt.Write()
                    hist_CAB_wwradpt.Write()


def main():
    analysis = MyAnalysis()
    analysis.run()



if __name__ == "__main__":
    main()

