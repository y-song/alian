#!/usr/bin/env python
import sys
import math
import numpy as np
import pandas as pd
import fastjet as fj
import fjcontrib
import ecorrel
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()


class DataAnalysis:
    def __init__(self):
        self.jet_R = 0.4
        self.jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
        self.cut_configs = [("sd", 0.1), ("maxkt", None)]
        self.trk_thrd = 1
        self.dphi_cut = -9999
        self.deta_cut = -9999
        # Jet-pt slices (lower edges); each pair = [low, high) in GeV.
        self.pt_slices = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]

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
        return f"jetpt{lo}_{hi}"

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

    def count_constituents(self, sj):
        n = 0
        for c in fj.sorted_by_pt(sj.constituents()):
            if c.pt() < self.trk_thrd:
                break
            n += 1
        return n

    def FillHists(self, label, sj, hist, weight, hist_ptRL=None, avg_pt=None, sj_B=None):
        c_select = fj.vectorPJ()
        for c in fj.sorted_by_pt(sj.constituents()):
            if c.pt() < self.trk_thrd:
                break
            c_select.append(c)

        if label == "AxB":
            c_select_B = fj.vectorPJ()
            for c in fj.sorted_by_pt(sj_B.constituents()):
                if c.pt() < self.trk_thrd:
                    break
                c_select_B.append(c)
            eec_result = ecorrel.CorrelatorBuilder(c_select, c_select_B, weight, 2, 1, self.dphi_cut, self.deta_cut)
        else:
            eec_result = ecorrel.CorrelatorBuilder(c_select, weight, 2, 1, self.dphi_cut, self.deta_cut)

        for index in range(eec_result.correlator(2).rs().size()):
            rl_value = eec_result.correlator(2).rs()[index]
            weight_value = eec_result.correlator(2).weights()[index]
            hist.Fill(rl_value, weight_value)
            if hist_ptRL is not None and avg_pt is not None:
                hist_ptRL.Fill(rl_value * avg_pt, weight_value)

    # ---------- main ----------
    def run(self, infile, outfile):
        df = pd.read_parquet(infile)
        root_outfile = ROOT.TFile(outfile, "RECREATE")

        nbins = 25
        xmin, xmax = 0.001, 1.0
        ptrl_xmin, ptrl_xmax = 0.1, 100
        log_bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
        ptrl_log_bins = np.logspace(np.log10(ptrl_xmin), np.log10(ptrl_xmax), nbins + 1)
        pt_bins = np.linspace(0, 100, 101)       # flat 0-100 GeV jet pt
        radpt_bins = np.linspace(0, 100, 101)    # flat 0-100 GeV radiator pt
        radkt_bins = np.linspace(-5, 5, nbins + 1)

        avg_jet_pts = {}

        for (lo, hi) in self.pt_slices:
            sl_tag = self.slice_tag(lo, hi)
            df_slice = df[(df['jet_pt'] >= lo) & (df['jet_pt'] < hi)]
            if len(df_slice) == 0:
                print(f"[{sl_tag}] no jets, skipping")
                continue

            avg_jet_pt = df_slice['jet_pt'].mean()
            avg_jet_pts[(lo, hi)] = avg_jet_pt
            print(f"[{sl_tag}] average jet pT = {avg_jet_pt:.2f}")

            grouped_jets = df_slice.groupby(['event_id', 'jet_id'])

            for cut_mode, cut_value in self.cut_configs:
                cut_tag = self.format_cut_tag(cut_mode, cut_value)
                tag = f"{sl_tag}_{cut_tag}"

                hist_jetpt_all = ROOT.TH1D(f"hist_jetpt_all_{tag}", "jet p_{T}; p_{T,jet}", len(pt_bins) - 1, pt_bins)
                hist_jetpt_cut = ROOT.TH1D(f"hist_jetpt_{tag}", "jet p_{T} (cut); p_{T,jet}", len(pt_bins) - 1, pt_bins)

                hist_full = ROOT.TH1D(f"hist_full_{tag}", "EEC; R_{L}", nbins, log_bins)
                hist_full_ptRL = ROOT.TH1D(f"hist_full_ptRL_{tag}", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                hist_rad_wwjetpt = ROOT.TH1D(f"hist_rad_{tag}_wwjetpt", "EEC; R_{L}", nbins, log_bins)
                hist_rad_wwradpt = ROOT.TH1D(f"hist_rad_{tag}_wwradpt", "EEC; R_{L}", nbins, log_bins)
                hist_rad_ptRL_wwjetpt = ROOT.TH1D(f"hist_rad_ptRL_{tag}_wwjetpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                hist_rad_ptRL_wwradpt = ROOT.TH1D(f"hist_rad_ptRL_{tag}_wwradpt", "EEC; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                hist_AA_wwjetpt = ROOT.TH1D(f"hist_AA_{tag}_wwjetpt", "AxA; R_{L}", nbins, log_bins)
                hist_BB_wwjetpt = ROOT.TH1D(f"hist_BB_{tag}_wwjetpt", "BxB", nbins, log_bins)
                hist_AB_wwjetpt = ROOT.TH1D(f"hist_AB_{tag}_wwjetpt", "AxB", nbins, log_bins)
                hist_AA_ptRL_wwjetpt = ROOT.TH1D(f"hist_AA_ptRL_{tag}_wwjetpt", "AxA; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                hist_BB_ptRL_wwjetpt = ROOT.TH1D(f"hist_BB_ptRL_{tag}_wwjetpt", "BxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                hist_AB_ptRL_wwjetpt = ROOT.TH1D(f"hist_AB_ptRL_{tag}_wwjetpt", "AxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                hist_AA_wwradpt = ROOT.TH1D(f"hist_AA_{tag}_wwradpt", "AxA; R_{L}", nbins, log_bins)
                hist_BB_wwradpt = ROOT.TH1D(f"hist_BB_{tag}_wwradpt", "BxB", nbins, log_bins)
                hist_AB_wwradpt = ROOT.TH1D(f"hist_AB_{tag}_wwradpt", "AxB", nbins, log_bins)
                hist_AA_ptRL_wwradpt = ROOT.TH1D(f"hist_AA_ptRL_{tag}_wwradpt", "AxA; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                hist_BB_ptRL_wwradpt = ROOT.TH1D(f"hist_BB_ptRL_{tag}_wwradpt", "BxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)
                hist_AB_ptRL_wwradpt = ROOT.TH1D(f"hist_AB_ptRL_{tag}_wwradpt", "AxB; <p_{T}>R_{L} [GeV/c]", nbins, ptrl_log_bins)

                hist_radiatorpt = ROOT.TH1D(f"radiator_pt_{tag}", "radiator p_{T}; p_{T,radiator}", len(radpt_bins) - 1, radpt_bins)
                hist_radiatorkt = ROOT.TH1D(f"radiator_lnkt_{tag}", "radiator ln k_{T}; ln(k_{T,radiator})", nbins, radkt_bins)
                hist_rg = ROOT.TH1D(f"hist_rg_{tag}", "R_{g} = #DeltaR_{AB}; R_{g}", nbins, log_bins)

                n_max = 60
                hist_nA_nB = ROOT.TH2D(f"hist_nA_nB_{tag}", "N in A vs B; N_{A}; N_{B}", n_max, -0.5, n_max - 0.5, n_max, -0.5, n_max - 0.5)
                hist_nTotalUnGroomed = ROOT.TH1D(f"hist_nTotalUnGroomed_{tag}", "N ungroomed; N", n_max, -0.5, n_max - 0.5)
                hist_nTotalGroomed = ROOT.TH1D(f"hist_nTotalGroomed_{tag}", "N groomed; N", n_max, -0.5, n_max - 0.5)
                comb_max = n_max * n_max
                hist_combAA = ROOT.TH1D(f"hist_combAA_{tag}", "AxA comb", 200, -0.5, comb_max - 0.5)
                hist_combBB = ROOT.TH1D(f"hist_combBB_{tag}", "BxB comb", 200, -0.5, comb_max - 0.5)
                hist_combAB = ROOT.TH1D(f"hist_combAB_{tag}", "AxB comb", 200, -0.5, comb_max - 0.5)
                hist_combTotal = ROOT.TH1D(f"hist_combTotal_{tag}", "Total comb", 200, -0.5, 4 * comb_max - 0.5)

                num_jets = 0.
                num_jets_passed_cut = 0.

                for (event_idx, jet_id), jet_constituents in grouped_jets:
                    if event_idx % 10000 == 0:
                        print(f"[{tag}] event {event_idx}")

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
                        continue
                    jet = jets[0]
                    num_jets += 1
                    hist_jetpt_all.Fill(jet.perp())

                    lund_gen = fjcontrib.LundGenerator(self.jet_def_ca)
                    lund_plane_elements = lund_gen.result(jet)

                    selected_d = self.select_split(lund_plane_elements, cut_mode, cut_value)
                    if selected_d is None:
                        continue

                    parent_radiator = selected_d.pair()
                    subjets = sorted(parent_radiator.pieces(), key=lambda x: x.pt(), reverse=True)
                    if len(subjets) != 2:
                        continue
                    subjet_a, subjet_b = subjets

                    num_jets_passed_cut += 1
                    hist_jetpt_cut.Fill(jet.perp())
                    hist_radiatorpt.Fill(parent_radiator.perp())
                    hist_radiatorkt.Fill(np.log(selected_d.kt()))
                    hist_rg.Fill(selected_d.Delta())

                    n_A = self.count_constituents(subjet_a)
                    n_B = self.count_constituents(subjet_b)
                    hist_nA_nB.Fill(n_A, n_B)
                    hist_nTotalUnGroomed.Fill(self.count_constituents(jet))
                    hist_nTotalGroomed.Fill(self.count_constituents(parent_radiator))
                    hist_combAA.Fill(n_A * n_A)
                    hist_combBB.Fill(n_B * n_B)
                    hist_combAB.Fill(n_A * n_B * 2)
                    hist_combTotal.Fill((n_A + n_B) ** 2)

                    for label, sj, h_jw, h_rw, h_jw_ptRL, h_rw_ptRL in [
                        ("full", jet, hist_full, None, hist_full_ptRL, None),
                        ("rad", parent_radiator, hist_rad_wwjetpt, hist_rad_wwradpt, hist_rad_ptRL_wwjetpt, hist_rad_ptRL_wwradpt),
                        ("A", subjet_a, hist_AA_wwjetpt, hist_AA_wwradpt, hist_AA_ptRL_wwjetpt, hist_AA_ptRL_wwradpt),
                        ("B", subjet_b, hist_BB_wwjetpt, hist_BB_wwradpt, hist_BB_ptRL_wwjetpt, hist_BB_ptRL_wwradpt),
                    ]:
                        self.FillHists(label, sj, h_jw, jet.perp(), hist_ptRL=h_jw_ptRL, avg_pt=avg_jet_pt)
                        if label != "full":
                            self.FillHists(label, sj, h_rw, parent_radiator.perp(), hist_ptRL=h_rw_ptRL, avg_pt=avg_jet_pt)

                    self.FillHists("AxB", subjet_a, hist_AB_wwjetpt, jet.perp(), hist_ptRL=hist_AB_ptRL_wwjetpt, avg_pt=avg_jet_pt, sj_B=subjet_b)
                    self.FillHists("AxB", subjet_a, hist_AB_wwradpt, parent_radiator.perp(), hist_ptRL=hist_AB_ptRL_wwradpt, avg_pt=avg_jet_pt, sj_B=subjet_b)

                # normalize / format
                self.FormatHist(hist_full, num_jets_passed_cut)
                self.FormatHist(hist_full_ptRL, num_jets_passed_cut, pt_rl=True, avg_pt=avg_jet_pt)
                for h in [hist_rad_wwjetpt, hist_rad_wwradpt,
                          hist_AA_wwjetpt, hist_BB_wwjetpt, hist_AB_wwjetpt,
                          hist_AA_wwradpt, hist_BB_wwradpt, hist_AB_wwradpt, hist_rg]:
                    self.FormatHist(h, num_jets_passed_cut)
                for h in [hist_rad_ptRL_wwjetpt, hist_rad_ptRL_wwradpt,
                          hist_AA_ptRL_wwjetpt, hist_BB_ptRL_wwjetpt, hist_AB_ptRL_wwjetpt,
                          hist_AA_ptRL_wwradpt, hist_BB_ptRL_wwradpt, hist_AB_ptRL_wwradpt]:
                    self.FormatHist(h, num_jets_passed_cut, pt_rl=True, avg_pt=avg_jet_pt)

                hist_CAB_wwjetpt = self.GetCABHist(hist_AA_wwjetpt, hist_BB_wwjetpt, hist_AB_wwjetpt, f"CAB_{tag}_wwjetpt")
                hist_CAB_wwradpt = self.GetCABHist(hist_AA_wwradpt, hist_BB_wwradpt, hist_AB_wwradpt, f"CAB_{tag}_wwradpt")
                hist_CAB_ptRL_wwjetpt = self.GetCABHist(hist_AA_ptRL_wwjetpt, hist_BB_ptRL_wwjetpt, hist_AB_ptRL_wwjetpt, f"CAB_ptRL_{tag}_wwjetpt")
                hist_CAB_ptRL_wwradpt = self.GetCABHist(hist_AA_ptRL_wwradpt, hist_BB_ptRL_wwradpt, hist_AB_ptRL_wwradpt, f"CAB_ptRL_{tag}_wwradpt")

                for h in [hist_jetpt_all, hist_jetpt_cut, hist_radiatorpt, hist_radiatorkt,
                          hist_full, hist_full_ptRL,
                          hist_rad_wwjetpt, hist_rad_ptRL_wwjetpt, hist_rad_wwradpt, hist_rad_ptRL_wwradpt,
                          hist_AA_wwjetpt, hist_AA_ptRL_wwjetpt, hist_BB_wwjetpt, hist_BB_ptRL_wwjetpt,
                          hist_AB_wwjetpt, hist_AB_ptRL_wwjetpt,
                          hist_AA_wwradpt, hist_AA_ptRL_wwradpt, hist_BB_wwradpt, hist_BB_ptRL_wwradpt,
                          hist_AB_wwradpt, hist_AB_ptRL_wwradpt,
                          hist_CAB_wwjetpt, hist_CAB_wwradpt, hist_CAB_ptRL_wwjetpt, hist_CAB_ptRL_wwradpt,
                          hist_nA_nB, hist_nTotalUnGroomed, hist_nTotalGroomed,
                          hist_combAA, hist_combBB, hist_combAB, hist_combTotal]:
                    h.Write()
                if cut_mode == "sd":
                    hist_rg.Write()

                print(f"[{tag}] jets={num_jets} passed_cut={num_jets_passed_cut}")

        root_outfile.Close()

        # write per-slice average jet pt
        avg_txt_path = outfile.replace(".root", "_avg_jet_pts.txt")
        with open(avg_txt_path, "w") as f:
            for (lo, hi), avg in avg_jet_pts.items():
                f.write(f"{lo} {hi} {avg:.6f}\n")
        print(f"Done: {infile} -> {outfile}")
        for (lo, hi), avg in avg_jet_pts.items():
            print(f"  jetpt{lo}_{hi}: <pt> = {avg:.2f}")


if __name__ == "__main__":
    infile, outfile = sys.argv[1], sys.argv[2]
    DataAnalysis().run(infile, outfile)