# Plot MC curves for JSE analysis
# 1. Plot all curves
# 2. Plot PYTHIA q vs q
# 3. Plot PYTHIA vs Herwig

import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from enum import IntEnum


class Color(IntEnum):
    BLUE   = ROOT.TColor.GetColor("#1f77b4")  # matplotlib C0
    GREEN  = ROOT.TColor.GetColor("#2ca02c")  # matplotlib C2
    RED    = ROOT.TColor.GetColor("#d62728")  # matplotlib C3
    ORANGE = ROOT.TColor.GetColor("#ff7f0e")  # matplotlib C1


class PlotCurves:
    def __init__(self):
        ROOT.gROOT.SetBatch(True)
        

        ROOT.gROOT.SetBatch(True)          # FIX 1: must be first — headless nodes
        ROOT.gStyle.SetLegendBorderSize(0)
        ROOT.gStyle.SetLegendFillColor(0)
        ROOT.gStyle.SetPadGridX(1)
        ROOT.gStyle.SetPadGridY(1)

        self.crosscheck = True
        self.target_jet_pts = [ 50, 100, 200, 500 ]
        self.partontypes    = ["inclusive", "quark", "gluon"]
        self.cut_modes      = [("sd", 0.1), ("maxkt", None)]
        # self.z_cuts = [0.1]

        self.rootfile_template = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/jse_preliminary_curves_{gen}_jetpt{jetpt}.root"
        self.current_jetpt = None
        self.pythia_rootfile = None
        self.herwig_rootfile = None

        # self.cut_modes = [("sd", 0.1), ("maxkt", None)]
        self.cut_mode = "" # "sd" #is this right?
        self.base_plot_dir = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots"

        self.FULL_HISTNAME_TEMPLATE = "hist_full_{}_jetpt{}_{}"
        self.RAD_HISTNAME_TEMPLATE  = "hist_rad_{}_jetpt{}_{}_ww{}pt"
        self.AA_HISTNAME_TEMPLATE   = "hist_AA_{}_jetpt{}_{}_ww{}pt"
        self.BB_HISTNAME_TEMPLATE   = "hist_BB_{}_jetpt{}_{}_ww{}pt"
        self.AB_HISTNAME_TEMPLATE   = "hist_AB_{}_jetpt{}_{}_ww{}pt"
        self.CAB_HISTNAME_TEMPLATE  = "CAB_{}_jetpt{}_{}_ww{}pt"

        self._persistent_canvases = []
        self._canvas_counter      = 0

    SECONDARY_COLORS = [ROOT.kGray, ROOT.kBlack, ROOT.kBlue, ROOT.kOrange+7, ROOT.kGreen+2]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get_file(self, gen):
        return self.pythia_rootfile if gen == "pythia" else self.herwig_rootfile

    def set_current_rootfiles(self, jetpt):
        if self.current_jetpt == jetpt:
            return
        for attr in ("pythia_rootfile", "herwig_rootfile"):
            f = getattr(self, attr, None)
            if f:
                try:
                    f.Close()
                except Exception:
                    pass
        self.pythia_rootfile = ROOT.TFile.Open(
            self.rootfile_template.format(gen="pythia", jetpt=jetpt))
        self.herwig_rootfile = ROOT.TFile.Open(
            self.rootfile_template.format(gen="herwig", jetpt=jetpt))
        self.current_jetpt = jetpt

    def get_cut_suffix(self, z_cut):
        return f"sd{z_cut}" if self.cut_mode == "sd" else "maxkt"

    def get_cut_label(self, z_cut):
        return f"SD z_{{cut}} = {z_cut}" if self.cut_mode == "sd" else "maxkt selection"

    def get_passed_label(self):
        return "SD" if self.cut_mode == "sd" else "max k_{T}"

    def get_output_dir(self, subdir, z_cut, plot_type=None):
        parts = [self.base_plot_dir, subdir, self.get_cut_suffix(z_cut)]
        if plot_type:
            parts.extend(plot_type.split('/'))
        path = os.path.join(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def make_canvas(self, base_name, gen="", partontype="", target_jetpt="",
                    z_cut="", den_weight="", title=None, w=800, h=600):
        self._canvas_counter += 1
        unique_name = (
            f"{base_name}_{gen}_{partontype}_{target_jetpt}"
            f"_{self.get_cut_suffix(z_cut)}_{den_weight}_{self._canvas_counter}"
        )
        canvas = ROOT.TCanvas(unique_name, title or base_name, w, h)
        self._persistent_canvases.append(canvas)
        return canvas

    def FormatHist(self, hist, color, linestyle, markerstyle=0, coloralpha=1):
        hist.SetLineColor(color)
        if coloralpha != 1:
            hist.SetLineColorAlpha(color, coloralpha)
        hist.SetLineStyle(linestyle)
        hist.SetLineWidth(2)
        if markerstyle > 0:
            hist.SetLineStyle(1)
            hist.SetMarkerStyle(markerstyle)
            hist.SetMarkerColor(color)

    def draw_hori_line(self, x1, x2, y1, color, linestyle, linewidth=1):
        line = ROOT.TLine(x1, y1, x2, y1)
        line.SetLineWidth(linewidth)
        line.SetLineColor(color)
        line.SetLineStyle(linestyle)
        line.Draw("SAME")
        return line

    def MakeEventLeg(self, gen, parton_type, jet_pt, z_cut, den_weight="",
                     x1=0.15, y1=0.7, x2=0.37, y2=0.88):
        event_leg = ROOT.TLegend(x1, y1, x2, y2)
        event_leg.SetBorderSize(0)
        event_leg.SetFillColor(0)
        event_leg.SetMargin(0)
        event_leg.AddEntry(ROOT.nullptr, f"{gen} {parton_type} jets", "")
        event_leg.AddEntry(ROOT.nullptr, f"jet p_{{T}} = {jet_pt} GeV/c", "")
        event_leg.AddEntry(ROOT.nullptr, self.get_cut_label(z_cut), "")
        if den_weight:
            event_leg.AddEntry(ROOT.nullptr,
                               f"using weight p_{{T,1}}p_{{T,2}} / p_{{T,{den_weight}}}^{{2}}", "")
        return event_leg

    # -------------------------------------------------------------------------
    # Histogram retrieval
    # -------------------------------------------------------------------------

    def GetSubjetEECHists(self, file, partontype, target_jetpt, z_cut, den_weight, ptRL=False):
        cut_suffix = self.get_cut_suffix(z_cut)
        if ptRL:
            hist_full = (None if den_weight == "rad" else
                         file.Get(f"hist_full_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}"))
            hist_rad = file.Get(f"hist_rad_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}pt")
            hist_AA  = file.Get(f"hist_AA_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}pt")
            hist_BB  = file.Get(f"hist_BB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}pt")
            hist_AB  = file.Get(f"hist_AB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}pt")
        else:
            hist_full = (None if den_weight == "rad" else
                         file.Get(self.FULL_HISTNAME_TEMPLATE.format(partontype, target_jetpt, cut_suffix)))
            hist_rad = file.Get(self.RAD_HISTNAME_TEMPLATE.format(partontype, target_jetpt, cut_suffix, den_weight))
            hist_AA  = file.Get(self.AA_HISTNAME_TEMPLATE.format( partontype, target_jetpt, cut_suffix, den_weight))
            hist_BB  = file.Get(self.BB_HISTNAME_TEMPLATE.format( partontype, target_jetpt, cut_suffix, den_weight))
            hist_AB  = file.Get(self.AB_HISTNAME_TEMPLATE.format( partontype, target_jetpt, cut_suffix, den_weight))

        # FIX 2: detach every histogram from the file immediately so they survive
        # when set_current_rootfiles() closes this file on the next jet pT iteration
        for h in [hist_full, hist_rad, hist_AA, hist_BB, hist_AB]:
            if h:
                h.SetDirectory(0)

        return hist_full, hist_rad, hist_AA, hist_BB, hist_AB

    def GetCABHist(self, file, partontype, target_jetpt, z_cut, den_weight, ptRL=False):
        cut_suffix = self.get_cut_suffix(z_cut)
        if ptRL:
            h = file.Get(f"CAB_ptRL_{partontype}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}pt")
        else:
            h = file.Get(self.CAB_HISTNAME_TEMPLATE.format(partontype, target_jetpt, cut_suffix, den_weight))
        if h:
            h.SetDirectory(0)   # FIX 2 (same issue)
        return h

    def GetRGHist(self, file, partontype, target_jetpt, z_cut):
        cut_suffix = self.get_cut_suffix(z_cut)
        h = file.Get(f"hist_rg_{partontype}_jetpt{target_jetpt}_{cut_suffix}")
        if h:
            h.SetDirectory(0)   # FIX 2 (same issue)
        return h

    # -------------------------------------------------------------------------
    # Drawing utilities
    # -------------------------------------------------------------------------

    def _draw_eec_set(self, hists, first=True):
        draw_opt = "HIST" if first else "HIST SAME"
        for h in hists:
            h.Draw(draw_opt)
            draw_opt = "HIST SAME"

    def _make_eec_legend2(self, ref_hists):
        legend2 = ROOT.TLegend(0.68, 0.5, 0.88, 0.65)
        if ref_hists[0] is not None:
            legend2.AddEntry(ref_hists[0], f"all jets that passed {self.get_passed_label()}", "l")
        legend2.AddEntry(ref_hists[1], "radiator", "l")
        legend2.AddEntry(ref_hists[2], "AxA", "l")
        legend2.AddEntry(ref_hists[3], "BxB", "l")
        legend2.AddEntry(ref_hists[4], "AxB", "l")
        return legend2

    def _format_secondary_hists(self, hists, colors):
        for hist, color in zip(hists, colors):
            if hist is None:
                continue
            self.FormatHist(hist, color, ROOT.kDashed) #, ROOT.kOpenSquare)

    def _save_canvas(self, canvas, subdir, z_cut, filename, plot_type=None):
        canvas.SaveAs(os.path.join(self.get_output_dir(subdir, z_cut, plot_type), filename))

    def _draw_eec_canvas(self, canvas, hists_primary, hists_secondary=None,
                         ev_leg=None, legend1=None, legend2=None,
                         crosscheck_hist=None):
        canvas.cd()
        all_hists = list(hists_primary)
        if hists_secondary:
            all_hists += list(hists_secondary)
        if crosscheck_hist:
            all_hists.append(crosscheck_hist)

        global_max = max(h.GetMaximum() for h in all_hists if h)
        hists_primary[0].GetYaxis().SetRangeUser(0, global_max * 1.4)

        self._draw_eec_set(hists_primary, first=True)
        if hists_secondary:
            self._draw_eec_set(hists_secondary, first=False)
        if crosscheck_hist:
            crosscheck_hist.Draw("HIST SAME")
        for leg in [ev_leg, legend1, legend2]:
            if leg:
                leg.Draw()

    # -------------------------------------------------------------------------
    # Plot: jet pT distribution
    # -------------------------------------------------------------------------

    def plot_jetpt(self, partontype, target_jetpt, z_cut):
        can_jetpt = self.make_canvas("can_jetpt", partontype=partontype,
                                     target_jetpt=target_jetpt, z_cut=z_cut,
                                     title="Jet pT distribution")
        can_jetpt.SetLogy()
        print("here!", z_cut)
        cut_suffix = self.get_cut_suffix(z_cut)
        suffix_str = f"_sd{z_cut}" if self.cut_mode == "sd" else "_maxkt"

        def _get_jetpt_hists(rootfile):
            h_all = rootfile.Get(f"hist_jetpt_all_{partontype}_jetpt{target_jetpt}{suffix_str}")
            h_cut = rootfile.Get(f"hist_jetpt_{partontype}_jetpt{target_jetpt}_{cut_suffix}")
            for h in [h_all, h_cut]:
                if h:
                    h.SetDirectory(0)
            return h_all, h_cut

        self.set_current_rootfiles(target_jetpt)
        h_all_pythia, h_cut_pythia = _get_jetpt_hists(self.pythia_rootfile)
        self.FormatHist(h_all_pythia, ROOT.kBlue,   ROOT.kSolid,  coloralpha=0.75)
        self.FormatHist(h_cut_pythia, ROOT.kBlue-7, ROOT.kDashed, coloralpha=0.75)

        ev_leg = self.MakeEventLeg("PYTHIA and HERWIG", partontype, target_jetpt, z_cut)
        legend = ROOT.TLegend(0.4, 0.7, 0.77, 0.88)
        legend.AddEntry(h_all_pythia, f"PYTHIA all jets (mean: {h_all_pythia.GetMean():.2f})", "l")
        legend.AddEntry(h_cut_pythia, f"PYTHIA {self.cut_mode.upper()} jets (mean: {h_cut_pythia.GetMean():.2f})", "l")

        h_all_pythia.SetMaximum(h_all_pythia.GetMaximum() * 10)
        can_jetpt.cd()
        h_all_pythia.Draw("HIST")
        h_cut_pythia.Draw("HIST SAME")

        if partontype == "inclusive":
            h_all_herwig, h_cut_herwig = _get_jetpt_hists(self.herwig_rootfile)
            self.FormatHist(h_all_herwig, Color.RED,   ROOT.kSolid,  coloralpha=0.75)
            self.FormatHist(h_cut_herwig, ROOT.kRed-7, ROOT.kDashed, coloralpha=0.75)
            legend.AddEntry(h_all_herwig, f"HERWIG all jets (mean: {h_all_herwig.GetMean():.2f})", "l")
            legend.AddEntry(h_cut_herwig, f"HERWIG {self.cut_mode.upper()} jets (mean: {h_cut_herwig.GetMean():.2f})", "l")
            h_all_herwig.Draw("HIST SAME")
            h_cut_herwig.Draw("HIST SAME")

        ev_leg.Draw()
        legend.Draw()
        self._save_canvas(can_jetpt, "pythia_vs_herwig", z_cut,
                          f"jetpt_pythia_vs_herwig_{partontype}_{target_jetpt}_R0.4_{cut_suffix}.pdf")

    # -------------------------------------------------------------------------
    # Plot: subjet EEC basic
    # -------------------------------------------------------------------------

    def plot_basic(self, gen, partontype, target_jetpt, z_cut, den_weight):
        self.set_current_rootfiles(target_jetpt)   # FIX 4: removed duplicate call
        file = self.get_file(gen)
        cut_suffix = self.get_cut_suffix(z_cut)

        for ptRL, subdir_suffix, canvas_name in [
            (False, "subjet_eec",      "canvas_basic"),
            (True,  "subjet_eec/ptRL", "canvas_ptRL"),
        ]:
            hists = self.GetSubjetEECHists(file, partontype, target_jetpt, z_cut, den_weight, ptRL=ptRL)
            hist_full, hist_rad, hist_AA, hist_BB, hist_AB = hists

            if not all([hist_rad, hist_AA, hist_BB, hist_AB]):
                continue

            canvas = self.make_canvas(canvas_name, gen=gen, partontype=partontype,
                                      target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight)
            canvas.SetLogx()

            crosscheck = None
            if not ptRL and self.crosscheck:
                crosscheck = hist_AA.Clone(
                    f"hist_crosscheck_{gen}_{partontype}_{target_jetpt}_{cut_suffix}_{den_weight}")
                crosscheck.Add(hist_BB)
                crosscheck.Add(hist_AB)
                self.FormatHist(crosscheck, ROOT.kGray+2, ROOT.kDashed)

            ev_leg = self.MakeEventLeg(gen, partontype, target_jetpt, z_cut, den_weight)
            legend = ROOT.TLegend(0.68, 0.5, 0.88, 0.65)
            if hist_full is not None:
                legend.AddEntry(hist_full, f"all {partontype} jets that passed {self.get_passed_label()}", "l")
            legend.AddEntry(hist_rad, "radiator", "l")
            if crosscheck:
                legend.AddEntry(crosscheck, "cross-check #Sigma_{subjets} = rad.", "l")
            legend.AddEntry(hist_AA, "AxA", "l")
            legend.AddEntry(hist_BB, "BxB", "l")
            legend.AddEntry(hist_AB, "AxB", "l")

            hists_to_draw = [h for h in hists if h is not None]
            self._draw_eec_canvas(canvas, hists_to_draw, ev_leg=ev_leg, legend2=legend,
                                  crosscheck_hist=crosscheck)

            prefix = "subjet_eec_ptRL" if ptRL else "subjet_eec"
            output_name = (
                f"{prefix}_comparison_{gen}_{partontype}_jetpt{target_jetpt}"
                f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
            )
            self._save_canvas(canvas, gen, z_cut, output_name, plot_type=subdir_suffix)

    # -------------------------------------------------------------------------
    # Plot: R_g overlaid with AA, BB, AB  (SD only)
    # -------------------------------------------------------------------------

    def plot_rg(self, gen, partontype, target_jetpt, z_cut, den_weight):
        if self.cut_mode != "sd":
            return

        self.set_current_rootfiles(target_jetpt)
        file = self.get_file(gen)
        cut_suffix = self.get_cut_suffix(z_cut)

        hist_rg = self.GetRGHist(file, partontype, target_jetpt, z_cut)
        if not hist_rg:
            print(f"WARNING: R_g histogram not found for {gen} {partontype} jetpt{target_jetpt} {cut_suffix}")
            return

        _, _, hist_AA, hist_BB, hist_AB = self.GetSubjetEECHists(
            file, partontype, target_jetpt, z_cut, den_weight)
        if not all([hist_AA, hist_BB, hist_AB]):
            print(f"WARNING: AA/BB/AB histograms not found for {gen} {partontype} jetpt{target_jetpt} {cut_suffix}")
            return

        # ----- Canvas 1: R_g vs all EEC components -----
        canvas = self.make_canvas(
            "canvas_rg", gen=gen, partontype=partontype,
            target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight,
            title="R_{g} vs EEC components"
        )
        canvas.SetLogx()
        canvas.cd()

        hist_AA.SetLineColor(Color.BLUE)
        hist_BB.SetLineColor(Color.ORANGE)
        hist_AB.SetLineColor(Color.GREEN)
        hist_rg.SetLineColor(ROOT.kRed+1)
        hist_rg.SetLineStyle(ROOT.kDashed)

        y_max = max(hist_AA.GetMaximum(), hist_BB.GetMaximum(),
                    hist_AB.GetMaximum(), hist_rg.GetMaximum()) * 1.3

        hist_AA.SetMaximum(y_max)
        hist_AA.SetMinimum(0)
        hist_AA.GetXaxis().SetTitle("R_{L} or R_{g}")
        hist_AA.GetYaxis().SetTitle("(1/N_{jets}) dN/d(R_{L} or R_{g})")

        hist_AA.Draw("HIST")
        hist_BB.Draw("HIST SAME")
        hist_AB.Draw("HIST SAME")
        hist_rg.Draw("HIST SAME")

        ev_leg = self.MakeEventLeg(gen, partontype, target_jetpt, z_cut, den_weight)
        legend = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
        legend.AddEntry(hist_AA, "AxA (EEC)", "l")
        legend.AddEntry(hist_BB, "BxB (EEC)", "l")
        legend.AddEntry(hist_AB, "AxB (EEC)", "l")
        legend.AddEntry(hist_rg, "R_{g} = #DeltaR_{AB}/R", "l")

        ev_leg.Draw()
        legend.Draw()

        output_name = (
            f"rg_vs_eec_{gen}_{partontype}_jetpt{target_jetpt}"
            f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
        )
        self._save_canvas(canvas, gen, z_cut, output_name, plot_type="rg")

        # ----- Canvas 2: R_g vs AxB only (self-normalized, two-panel with ratio) -----
        # Self-normalize both curves (clone so we don't disturb canvas 1's hists)
        hist_rg_norm = hist_rg.Clone("hist_rg_norm")
        hist_AB_norm = hist_AB.Clone("hist_AB_norm")
        hist_rg_norm.SetDirectory(0)
        hist_AB_norm.SetDirectory(0)

        rg_integral = hist_rg_norm.Integral()
        ab_integral = hist_AB_norm.Integral()
        if rg_integral > 0:
            hist_rg_norm.Scale(1.0 / rg_integral)
        if ab_integral > 0:
            hist_AB_norm.Scale(1.0 / ab_integral)

        canvas2 = self.make_canvas(
            "canvas_rg_vs_AxB", gen=gen, partontype=partontype,
            target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight,
            title="R_{g} vs AxB (self-normalized)"
        )

        # Top pad (main plot)
        pad1 = ROOT.TPad("pad1_rgAxB", "pad1_rgAxB", 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.SetLogx()
        pad1.Draw()

        # Bottom pad (ratio)
        canvas2.cd()
        pad2 = ROOT.TPad("pad2_rgAxB", "pad2_rgAxB", 0, 0.0, 1, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.35)
        pad2.SetLogx()
        pad2.SetGridy()
        pad2.Draw()

        # ----- Top panel -----
        pad1.cd()
        hist_rg_norm.SetLineColor(ROOT.kRed+1)
        hist_rg_norm.SetLineStyle(ROOT.kDashed)
        hist_AB_norm.SetLineColor(Color.GREEN)
        hist_AB_norm.SetLineStyle(ROOT.kSolid)

        y_max2 = max(hist_rg_norm.GetMaximum(), hist_AB_norm.GetMaximum()) * 1.3
        hist_AB_norm.SetMaximum(y_max2)
        hist_AB_norm.SetMinimum(0)
        hist_AB_norm.GetYaxis().SetTitle("Self-normalized")
        hist_AB_norm.GetYaxis().SetTitleSize(0.05)
        hist_AB_norm.GetYaxis().SetTitleOffset(0.9)
        hist_AB_norm.GetXaxis().SetLabelSize(0)  # hide x labels on top pad

        hist_AB_norm.Draw("HIST")
        hist_rg_norm.Draw("HIST SAME")

        ev_leg2 = self.MakeEventLeg(gen, partontype, target_jetpt, z_cut, den_weight)
        legend2 = ROOT.TLegend(0.65, 0.74, 0.88, 0.88)
        legend2.AddEntry(hist_AB_norm, "AxB (EEC)", "l")
        legend2.AddEntry(hist_rg_norm, "R_{g} = #DeltaR_{AB}/R", "l")
        ev_leg2.Draw()
        legend2.Draw()

        # ----- Bottom panel (ratio R_g / AxB) -----
        pad2.cd()
        hist_ratio = hist_AB_norm.Clone("hist_ratio_AxB_rg")
        hist_ratio.SetDirectory(0)
        hist_ratio.Divide(hist_rg_norm)
        hist_ratio.SetLineColor(ROOT.kBlack)
        hist_ratio.SetMarkerStyle(20)
        hist_ratio.SetMarkerSize(0.7)

        hist_ratio.GetYaxis().SetTitle("AxB / R_{g}")
        hist_ratio.GetYaxis().SetNdivisions(505)
        hist_ratio.GetYaxis().SetTitleSize(0.11)
        hist_ratio.GetYaxis().SetTitleOffset(0.4)
        hist_ratio.GetYaxis().SetLabelSize(0.09)
        hist_ratio.GetXaxis().SetTitle("R_{L} or R_{g}")
        hist_ratio.GetXaxis().SetTitleSize(0.12)
        hist_ratio.GetXaxis().SetTitleOffset(1.0)
        hist_ratio.GetXaxis().SetLabelSize(0.09)
        hist_ratio.SetMinimum(0.0)
        hist_ratio.SetMaximum(2.0)

        hist_ratio.Draw("EP")

        line = ROOT.TLine(hist_ratio.GetXaxis().GetXmin(), 1.0,
                          hist_ratio.GetXaxis().GetXmax(), 1.0)
        line.SetLineColor(ROOT.kGray+2)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw("SAME")

        canvas2.cd()

        output_name2 = (
            f"rg_vs_AxB_{gen}_{partontype}_jetpt{target_jetpt}"
            f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
        )
        self._save_canvas(canvas2, gen, z_cut, output_name2, plot_type="rg")

    # -------------------------------------------------------------------------
    # Plot: C_AB  (regular and ptRL unified)
    # -------------------------------------------------------------------------

    def _plot_CAB_impl(self, gen, partontype, target_jetpt, z_cut, ptRL=False):
        self.set_current_rootfiles(target_jetpt)
        file = self.get_file(gen)
        cut_suffix = self.get_cut_suffix(z_cut)

        canvas_name = "can_CAB_ptRL" if ptRL else "can_CAB"
        title       = "C_{AB}_ptRL"  if ptRL else "C_{AB}"
        subdir      = "CAB/ptRL"     if ptRL else "CAB"
        file_prefix = "CAB_ptRL"     if ptRL else "CAB"

        can_CAB = self.make_canvas(canvas_name, gen=gen, partontype=partontype,
                                   target_jetpt=target_jetpt, z_cut=z_cut, title=title)
        can_CAB.SetLogx()
        can_CAB.SetLogy()

        hist_wwjetpt = self.GetCABHist(file, partontype, target_jetpt, z_cut, "jet", ptRL=ptRL)
        hist_wwradpt = self.GetCABHist(file, partontype, target_jetpt, z_cut, "rad", ptRL=ptRL)

        ev_leg = self.MakeEventLeg(gen, partontype, target_jetpt, z_cut)
        legend = ROOT.TLegend(0.2, 0.5, 0.4, 0.65)
        legend.AddEntry(hist_wwjetpt, "weight uses jet pt", "l")
        legend.AddEntry(hist_wwradpt, "weight uses rad pt", "l")

        can_CAB.cd()
        hist_wwjetpt.Draw("HIST")
        hist_wwradpt.Draw("HIST SAME")
        ev_leg.Draw()
        legend.Draw()
        self.draw_hori_line(1e-3, 1, 1, ROOT.kGray+3, 9)

        output_name = f"{file_prefix}_{gen}_{partontype}_jetpt{target_jetpt}_R0.4_{cut_suffix}.pdf"
        self._save_canvas(can_CAB, gen, z_cut, output_name, plot_type=subdir)

    def plot_CAB(self, gen, partontype, target_jetpt, z_cut):
        self._plot_CAB_impl(gen, partontype, target_jetpt, z_cut, ptRL=False)

    def plot_CAB_ptRL(self, gen, partontype, target_jetpt, z_cut):
        self._plot_CAB_impl(gen, partontype, target_jetpt, z_cut, ptRL=True)

    # -------------------------------------------------------------------------
    # Plot: radiator property
    # -------------------------------------------------------------------------

    def plot_rad_prop(self, gen, radprop, partontype, target_jetpt, z_cut):
        self.set_current_rootfiles(target_jetpt)
        file = self.get_file(gen)
        cut_suffix = self.get_cut_suffix(z_cut)

        can_radprop = self.make_canvas("can_radprop", gen=gen, partontype=partontype,
                                       target_jetpt=target_jetpt, z_cut=z_cut, den_weight=radprop)
        hist = file.Get(f"radiator_{radprop}_{partontype}_jetpt{target_jetpt}_{cut_suffix}")
        if hist:
            hist.SetDirectory(0)

        can_radprop.cd()
        hist.Draw()
        output_name = f"radiator_{radprop}_{gen}_{partontype}_jetpt{target_jetpt}_R0.4_{cut_suffix}.pdf"
        self._save_canvas(can_radprop, gen, z_cut, output_name, "radiator_prop")

    # -------------------------------------------------------------------------
    # Plot: quark vs gluon  /  herwig vs pythia  (unified two-generator plot)
    # -------------------------------------------------------------------------

    def _plot_two_gen_eec(self, file_a, file_b, partontype_a, partontype_b,
                          label_a, label_b, gen_label,
                          target_jetpt, z_cut, den_weight,
                          subdir, file_tag, ratio_ytitle,
                          ratio_subdir, ratio_file_tag,
                          canvas_name, ratio_canvas_name,
                          ptRL_canvas_name, ptRL_subdir, ptRL_file_tag):
        cut_suffix = self.get_cut_suffix(z_cut)

        canvas = self.make_canvas(canvas_name, partontype=partontype_a,
                                  target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight)
        canvas_ratio = self.make_canvas(ratio_canvas_name, partontype=partontype_a,
                                        target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight)
        canvas.SetLogx()
        canvas_ratio.SetLogx()

        hists_a = self.GetSubjetEECHists(file_a, partontype_a, target_jetpt, z_cut, den_weight)
        hists_b = self.GetSubjetEECHists(file_b, partontype_b, target_jetpt, z_cut, den_weight)
        hist_full_a, hist_rad_a, hist_AA_a, hist_BB_a, hist_AB_a = hists_a
        hist_full_b, hist_rad_b, hist_AA_b, hist_BB_b, hist_AB_b = hists_b

        if not all([hist_rad_a, hist_AA_a, hist_BB_a, hist_AB_a,
                    hist_rad_b, hist_AA_b, hist_BB_b, hist_AB_b]):
            return

        self._format_secondary_hists(hists_b, self.SECONDARY_COLORS)

        ev_leg = self.MakeEventLeg(
            gen_label,
            "q and g" if partontype_a == "quark" else partontype_a,
            target_jetpt, z_cut, den_weight)

        legend1 = ROOT.TLegend(0.68, 0.68, 0.88, 0.73)
        legend1.AddEntry(hist_rad_a if hist_full_a is None else hist_full_a, label_a, "l")
        legend1.AddEntry(hist_rad_b if hist_full_b is None else hist_full_b, label_b, "l")
        legend2 = self._make_eec_legend2(hists_a)

        hists_a_to_draw = [h for h in hists_a if h is not None]
        hists_b_to_draw = [h for h in hists_b if h is not None]
        self._draw_eec_canvas(canvas, hists_a_to_draw, hists_secondary=hists_b_to_draw,
                              ev_leg=ev_leg, legend1=legend1, legend2=legend2)
        output_name = (
            f"subjet_eec_comparison_{file_tag}_jetpt{target_jetpt}"
            f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
        )
        self._save_canvas(canvas, subdir, z_cut, output_name, plot_type="subjet_eec")

        # ptRL variant
        hists_ptRL_a = self.GetSubjetEECHists(
            file_a, partontype_a, target_jetpt, z_cut, den_weight, ptRL=True)
        hists_ptRL_b = self.GetSubjetEECHists(
            file_b, partontype_b, target_jetpt, z_cut, den_weight, ptRL=True)
        _, hist_rad_ptRL_a, hist_AA_ptRL_a, hist_BB_ptRL_a, hist_AB_ptRL_a = hists_ptRL_a
        _, hist_rad_ptRL_b, hist_AA_ptRL_b, hist_BB_ptRL_b, hist_AB_ptRL_b = hists_ptRL_b
        self._format_secondary_hists(hists_ptRL_b, self.SECONDARY_COLORS)

        if all([hist_rad_ptRL_a, hist_AA_ptRL_a, hist_BB_ptRL_a, hist_AB_ptRL_a,
                hist_rad_ptRL_b, hist_AA_ptRL_b, hist_BB_ptRL_b, hist_AB_ptRL_b]):
            canvas_ptRL = self.make_canvas(ptRL_canvas_name, partontype=partontype_a,
                                           target_jetpt=target_jetpt, z_cut=z_cut,
                                           den_weight=den_weight)
            canvas_ptRL.SetLogx()
            hists_ptRL_a_to_draw = [h for h in hists_ptRL_a if h is not None]
            hists_ptRL_b_to_draw = [h for h in hists_ptRL_b if h is not None]
            self._draw_eec_canvas(canvas_ptRL, hists_ptRL_a_to_draw,
                                  hists_secondary=hists_ptRL_b_to_draw,
                                  ev_leg=ev_leg, legend1=legend1, legend2=legend2)
            output_name_ptRL = (
                f"subjet_eec_ptRL_comparison_{ptRL_file_tag}_jetpt{target_jetpt}"
                f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
            )
            self._save_canvas(canvas_ptRL, subdir, z_cut, output_name_ptRL,
                              plot_type=ptRL_subdir)

        # CAB ratio
        hist_CAB_a = self.GetCABHist(file_a, partontype_a, target_jetpt, z_cut, den_weight)
        hist_CAB_b = self.GetCABHist(file_b, partontype_b, target_jetpt, z_cut, den_weight)
        ratio_CAB = hist_CAB_a.Clone(
            f"ratio_CAB_{ratio_file_tag}_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}")
        ratio_CAB.Divide(hist_CAB_b)
        ratio_CAB.GetYaxis().SetTitle(ratio_ytitle)

        canvas_ratio.cd()
        ratio_CAB.Draw("HIST")
        self.draw_hori_line(1e-3, 1, 1, ROOT.kGray+3, 9)
        output_name_ratio = (
            f"CAB_ratio_{ratio_file_tag}_jetpt{target_jetpt}"
            f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
        )
        self._save_canvas(canvas_ratio, subdir, z_cut, output_name_ratio,
                          plot_type=ratio_subdir)

    def plot_q_vs_g(self, gen, target_jetpt, z_cut, den_weight):
        self.set_current_rootfiles(target_jetpt)
        file = self.get_file(gen)
        self._plot_two_gen_eec(
            file_a=file, file_b=file,
            partontype_a="quark", partontype_b="gluon",
            label_a="quark", label_b="gluon",
            gen_label=gen,
            target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight,
            subdir=gen,
            file_tag=f"{gen}_QVSG",
            ratio_ytitle="C_{AB}^{quark} / C_{AB}^{gluon}",
            ratio_subdir="CAB/ratio", #None,
            ratio_file_tag=f"QVSG_{gen}",
            canvas_name="canvas_q_vs_g",
            ratio_canvas_name="canvas_ratio_qvsg",
            ptRL_canvas_name="canvas_ptRL_qvsg",
            ptRL_subdir= "subjet_eec/QVSG/ptRL", #"QVSG/ptRL",
            ptRL_file_tag=f"{gen}_QVSG",
        )

    def plot_herwig_vs_pythia(self, partontype, target_jetpt, z_cut, den_weight):
        self.set_current_rootfiles(target_jetpt)
        self._plot_two_gen_eec(
            file_a=self.pythia_rootfile, file_b=self.herwig_rootfile,
            partontype_a="inclusive", partontype_b="inclusive",
            label_a="pythia", label_b="herwig",
            gen_label="pythia and herwig",
            target_jetpt=target_jetpt, z_cut=z_cut, den_weight=den_weight,
            subdir="pythia_vs_herwig",
            file_tag="PYTHIA_VS_HERWIG",
            ratio_ytitle="C_{AB}^{pythia} / C_{AB}^{herwig}",
            ratio_subdir="CAB_ratio",
            ratio_file_tag="PYTHIA_VS_HERWIG",
            canvas_name="canvas_herwig_vs_pythia",
            ratio_canvas_name="canvas_ratio_hvp",
            ptRL_canvas_name="canvas_ptRL_hvp",
            ptRL_subdir="subjet_eec/ptRL",
            ptRL_file_tag="PYTHIA_VS_HERWIG",
        )
        self._plot_CAB_ptRL_ratio_hvp(partontype, target_jetpt, z_cut)

    def _plot_CAB_ptRL_ratio_hvp(self, partontype, target_jetpt, z_cut):
        """CAB ptRL ratio for PYTHIA vs HERWIG (no q vs g equivalent)."""
        cut_suffix = self.get_cut_suffix(z_cut)
        self.set_current_rootfiles(target_jetpt)
        hist_a = self.GetCABHist(
            self.pythia_rootfile, "inclusive", target_jetpt, z_cut, "jet", ptRL=True)
        hist_b = self.GetCABHist(
            self.herwig_rootfile, "inclusive", target_jetpt, z_cut, "jet", ptRL=True)
        if not (hist_a and hist_b):
            return
        ratio = hist_a.Clone(
            f"ratio_CAB_ptRL_pythia_over_herwig_jetpt{target_jetpt}_{cut_suffix}_wwjetpt")
        ratio.Divide(hist_b)
        ratio.GetYaxis().SetTitle(
            "C_{AB}^{pythia}_{ptRL} / C_{AB}^{herwig}_{ptRL}")

        canvas = self.make_canvas("canvas_ratio_ptRL_hvp", partontype=partontype,
                                  target_jetpt=target_jetpt, z_cut=z_cut)
        canvas.SetLogx()
        canvas.cd()
        ratio.Draw("HIST")
        self.draw_hori_line(1e-3, 1, 1, ROOT.kGray+3, 9)
        output_name = (
            f"CAB_ptRL_ratio_PYTHIA_VS_HERWIG_jetpt{target_jetpt}_R0.4_{cut_suffix}.pdf"
        )
        self._save_canvas(canvas, "pythia_vs_herwig", z_cut, output_name,
                          plot_type="CAB_ratio/ptRL")

    # -------------------------------------------------------------------------
    # Plot: PYTHIA vs HERWIG ratio across all jet pTs
    # -------------------------------------------------------------------------

    def plot_herwig_vs_pythia_ratio_acrosspt(self, z_cut, den_weight):
        cut_suffix = self.get_cut_suffix(z_cut)
        markers = [ROOT.kFullCircle, ROOT.kFullSquare, ROOT.kFullDiamond, ROOT.kFullStar]
        colors  = [ROOT.kBlack, ROOT.kBlue, ROOT.kOrange+7, ROOT.kGreen+2]

        panel_labels = ["radiator", "AxA", "BxB", "AxB"]
        ratio_names  = ["rad", "AA", "BB", "AB"]

        canvas_ratio = self.make_canvas("canvas_ratio_acrosspt", partontype="inclusive",
                                        z_cut=z_cut, den_weight=den_weight, w=1200, h=1000)
        canvas_ratio.Divide(2, 2, 0.005, 0.005)

        ev_leg    = self.MakeEventLeg("pythia and herwig", "inclusive", "50-500",
                                      z_cut, den_weight)
        legend_pt = ROOT.TLegend(0.4, 0.7, 0.77, 0.88)
        legend_pt.SetBorderSize(0)

        dummy_graphs = []
        for target_jetpt, marker, color in zip(self.target_jet_pts, markers, colors):
            dummy = ROOT.TGraph(1)
            dummy.SetMarkerStyle(marker)
            dummy.SetMarkerColor(color)
            dummy.SetLineColor(color)
            legend_pt.AddEntry(dummy, f"jet p_{{T}} = {target_jetpt} GeV/c", "p")
            dummy_graphs.append(dummy)

        persistent_hists = []

        for ijetpt, target_jetpt in enumerate(self.target_jet_pts):
            self.set_current_rootfiles(target_jetpt)

            hists_pythia = self.GetSubjetEECHists(
                self.pythia_rootfile, "inclusive", target_jetpt, z_cut, den_weight)
            hists_herwig = self.GetSubjetEECHists(
                self.herwig_rootfile, "inclusive", target_jetpt, z_cut, den_weight)

            _, hist_rad_p, hist_AA_p, hist_BB_p, hist_AB_p = hists_pythia
            _, hist_rad_h, hist_AA_h, hist_BB_h, hist_AB_h = hists_herwig

            if not all([hist_rad_p, hist_AA_p, hist_BB_p, hist_AB_p,
                        hist_rad_h, hist_AA_h, hist_BB_h, hist_AB_h]):
                persistent_hists.append(None)
                continue

            pythia_hists = [hist_rad_p, hist_AA_p, hist_BB_p, hist_AB_p]
            herwig_hists = [hist_rad_h, hist_AA_h, hist_BB_h, hist_AB_h]

            ratios = []
            for hist_p, hist_h, name in zip(pythia_hists, herwig_hists, ratio_names):
                r = hist_p.Clone(
                    f"ratio_pythia_over_herwig_{name}_jetpt{target_jetpt}"
                    f"_{cut_suffix}_ww{den_weight}")
                r.SetDirectory(0)
                r.Divide(hist_h)
                r.GetYaxis().SetTitle("PYTHIA / HERWIG")
                r.GetYaxis().SetTitleSize(0.06)
                r.GetYaxis().SetTitleOffset(0.9)
                r.GetYaxis().SetRangeUser(0, 2.5)
                r.GetXaxis().SetTitleSize(0.06)
                self.FormatHist(r, colors[ijetpt], ijetpt + 1, markers[ijetpt])
                ratios.append(r)

            persistent_hists.append(ratios)

        latex_labels = []

        for ipanel, (name, label) in enumerate(zip(ratio_names, panel_labels)):
            pad = canvas_ratio.cd(ipanel + 1)
            pad.SetLogx()
            pad.SetLeftMargin(0.14)
            pad.SetBottomMargin(0.14)

            first = True
            for ijetpt in range(len(self.target_jet_pts)):
                if persistent_hists[ijetpt] is None:
                    continue
                ratio = persistent_hists[ijetpt][ipanel]
                if not ratio:
                    continue
                ratio.Draw("PE" if first else "PE SAME")
                first = False

            if not first:
                ref = next(
                    persistent_hists[i][ipanel]
                    for i in range(len(self.target_jet_pts))
                    if persistent_hists[i] is not None and persistent_hists[i][ipanel]
                )
                self.draw_hori_line(
                    ref.GetXaxis().GetXmin(),
                    ref.GetXaxis().GetXmax(),
                    1, ROOT.kGray+3, 9)

            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.07)
            latex.DrawLatex(0.18, 0.84, label)
            latex_labels.append(latex)

            if ipanel == 0:
                ev_leg.Draw()
                legend_pt.Draw()

        output_name = (
            f"subjet_eec_ratio_PYTHIA_VS_HERWIG_inclusive_alljetpt"
            f"_R0.4_{cut_suffix}_ww{den_weight}.pdf"
        )
        self._save_canvas(canvas_ratio, "pythia_vs_herwig", z_cut, output_name, "subjet_eec")

        # CAB ratio across pT
        can_CAB = self.make_canvas("can_CAB_acrosspt", partontype="inclusive",
                                   z_cut=z_cut, den_weight=den_weight, title="C_{AB}")
        can_CAB.SetLogx()
        can_CAB.SetLogy()
        can_CAB.cd()

        legend_CAB = ROOT.TLegend(0.38, 0.12, 0.58, 0.27)
        legend_CAB.SetBorderSize(0)
        persistent_CAB_ratios = []

        first_CAB = True
        for ijetpt, target_jetpt in enumerate(self.target_jet_pts):
            self.set_current_rootfiles(target_jetpt)

            hist_CAB_pythia = self.GetCABHist(
                self.pythia_rootfile, "inclusive", target_jetpt, z_cut, den_weight)
            hist_CAB_herwig = self.GetCABHist(
                self.herwig_rootfile, "inclusive", target_jetpt, z_cut, den_weight)

            if not hist_CAB_pythia or not hist_CAB_herwig:
                continue

            ratio_CAB = hist_CAB_pythia.Clone(
                f"ratio_CAB_pythia_over_herwig_jetpt{target_jetpt}_{cut_suffix}_ww{den_weight}")
            ratio_CAB.SetDirectory(0)
            ratio_CAB.Divide(hist_CAB_herwig)
            ratio_CAB.GetYaxis().SetTitle("C_{AB}^{pythia} / C_{AB}^{herwig}")
            self.FormatHist(ratio_CAB, colors[ijetpt], ijetpt + 1, markers[ijetpt])
            legend_CAB.AddEntry(ratio_CAB, f"jet p_{{T}} = {target_jetpt} GeV/c", "pe")
            persistent_CAB_ratios.append(ratio_CAB)

            ratio_CAB.Draw("PE" if first_CAB else "PE SAME")
            first_CAB = False

        if persistent_CAB_ratios:
            ev_leg.Draw()
            legend_CAB.Draw()
            self.draw_hori_line(1e-3, 1, 1, ROOT.kGray+3, 9)

        output_name = (
            f"CAB_ratio_PYTHIA_VS_HERWIG_inclusive_alljetpt"
            f"_R0.4_{cut_suffix}.pdf"
        )
        self._save_canvas(can_CAB, "pythia_vs_herwig", z_cut, output_name,
                          plot_type="CAB_ratio")

    # -------------------------------------------------------------------------
    # MPV (Most Probable Value) Summary Plots
    # -------------------------------------------------------------------------

    @staticmethod
    def gaus_log(x, mu, C, sg):
        """Log-normal distribution function."""
        return C * np.exp(-(np.log(x / mu)) ** 2 / (2 * sg * sg))

    @staticmethod
    def _select_fit_window(xs, ys, yerrs, n_side=3, n_min=2):
        """Restrict (xs, ys, yerrs) to a window around the maximum of ys.

        Tries to keep `n_side` points on each side of the peak; if there aren't
        enough points on one side, it falls back toward `n_min` but always keeps
        at least `n_min` points per side when the data allows.

        Returns (xs_w, ys_w, yerrs_w, peak_idx_global).
        """
        n = len(ys)
        if n == 0:
            return xs, ys, yerrs, None

        peak = int(np.argmax(ys))

        # How many points are actually available on each side of the peak
        left_avail = peak
        right_avail = n - 1 - peak

        # Start by asking for n_side on each side, clamp to what's available
        left = min(n_side, left_avail)
        right = min(n_side, right_avail)

        # If a side is short, try to compensate on the other side so the total
        # window still has a reasonable number of points for a 3-param fit.
        target_total = 2 * n_side + 1
        deficit = target_total - (left + right + 1)
        if deficit > 0:
            # give extra to whichever side still has room
            extra_left = min(deficit, left_avail - left)
            left += extra_left
            deficit -= extra_left
            extra_right = min(deficit, right_avail - right)
            right += extra_right

        lo = peak - left
        hi = peak + right + 1  # slice end is exclusive

        return xs[lo:hi], ys[lo:hi], yerrs[lo:hi], peak
    
    def _fit_mpv(self, hist, fit_range=None, save_path=None, plot_title=None):
        """Extract the peak position (mu) from a ROOT histogram by fitting gaus_log.

        Returns (mu, mu_err) or (None, None) if the fit fails.

        If save_path is provided, also saves a diagnostic plot of data + fit.
        """
        if hist is None:
            return None, None

        # Pull bin centers and contents into numpy arrays
        nb = hist.GetNbinsX()
        xs, ys, yerrs = [], [], []
        for ib in range(1, nb + 1):
            x = hist.GetBinCenter(ib)
            y = hist.GetBinContent(ib)
            e = hist.GetBinError(ib)
            if x <= 0 or y <= 0:
                continue
            if fit_range is not None and (x < fit_range[0] or x > fit_range[1]):
                continue
            xs.append(x); ys.append(y); yerrs.append(e if e > 0 else 1.0)

        if len(xs) < 5:
            return None, None

        xs = np.array(xs); ys = np.array(ys); yerrs = np.array(yerrs)

        # --- restrict to a window around the peak ---
        xs_full, ys_full, yerrs_full = xs, ys, yerrs  # keep originals for plotting
        xs, ys, yerrs, peak_idx = self._select_fit_window(
            xs, ys, yerrs, n_side=3, n_min=2
        )

        # Guard: need at least as many points as free parameters (3) to fit
        if len(xs) < 3:
            print(f"  Not enough points around peak to fit ({len(xs)} found)")
            if save_path is not None:
                try:
                    self._save_fit_diagnostic(xs_full, ys_full, yerrs_full,
                                              None, save_path, plot_title,
                                              None, None, p0=None)
                except Exception:
                    pass
            return None, None


        # Seed mu at the bin with the largest content
        mu0 = xs[np.argmax(ys)]
        C0  = ys.max()
        # Estimate sigma from points above half-max
        half_max = ys.max() / 2.0
        above = xs[ys >= half_max]
        if len(above) >= 2:
            # FWHM in log-space ≈ 2.355 * sigma (Gaussian relation)
            sg0 = (np.log(above.max()) - np.log(above.min())) / 2.355
            sg0 = max(sg0, 0.05)   # floor to avoid zero
        else:
            sg0 = 0.5

        p0 = [mu0, C0, sg0]
        mu, mu_err, popt = None, None, None
        try:
            popt, pcov = curve_fit(
                self.gaus_log, xs, ys,
                p0=p0,
                sigma=yerrs, absolute_sigma=False,
                maxfev=5000,
            )
            mu_fit, C_fit, sg_fit = popt
            if not np.isfinite(mu_fit) or mu_fit <= 0:
                popt = None
            else:
                mu = float(mu_fit)
                mu_err = float(np.sqrt(pcov[0, 0])) if pcov is not None else 0.0
        except Exception as ex:
            print(f"  MPV fit failed: {ex}")
            popt = None

        # Optionally save a diagnostic plot
        if save_path is not None:
            try:
                self._save_fit_diagnostic(
                    xs_full, ys_full, yerrs_full,   # full data for context
                    popt, save_path, plot_title, mu, mu_err,
                    p0=p0, fit_xs=xs,               # windowed x's actually fit
                )
            except Exception as ex:
                print(f"  Failed to save fit diagnostic: {ex}")

        return mu, mu_err

    def _save_fit_diagnostic(self, xs, ys, yerrs, popt, save_path,
                            title, mu, mu_err, p0=None, fit_xs=None):
        """Save a matplotlib plot showing histogram data points, the initial-guess
        curve (from p0), and the fitted curve."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(xs, ys, yerr=yerrs, fmt="o", markersize=4,
                    color="black", capsize=2, label="data", zorder=2)

        # highlight points used in the fit
        if fit_xs is not None and len(fit_xs):
            mask = np.isin(xs, fit_xs)
            ax.errorbar(xs[mask], ys[mask], yerr=yerrs[mask], fmt="o",
                        markersize=6, mfc="none", mec="C2", mew=1.5,
                        capsize=2, label="fit window", zorder=2.2)
            grid_lo, grid_hi = fit_xs.min(), fit_xs.max()
        else:
            grid_lo, grid_hi = xs.min(), xs.max()

        # build the curve grid only over the fit window
        x_grid = np.logspace(np.log10(grid_lo), np.log10(grid_hi), 400)

        # ---- Initial-guess (prior) curve ----
        if p0 is not None:
            y_init = self.gaus_log(x_grid, *p0)
            ax.plot(x_grid, y_init, "--", color="C0", linewidth=1.5, alpha=0.8,
                    label=(f"initial guess\n"
                           f"$\\mu_0$ = {p0[0]:.4g}, $C_0$ = {p0[1]:.4g}, "
                           f"$\\sigma_0$ = {p0[2]:.4g}"),
                    zorder=2.5)

        # ---- Fitted curve ----
        if popt is not None:
            y_fit = self.gaus_log(x_grid, *popt)
            ax.plot(x_grid, y_fit, "-", color="C3", linewidth=2,
                    label=(f"gaus_log fit\n"
                           f"$\\mu$ = {popt[0]:.4g}, $C$ = {popt[1]:.4g}, "
                           f"$\\sigma$ = {popt[2]:.4g}"),
                    zorder=3)
            ax.axvline(popt[0], color="C3", linestyle="--", alpha=0.6,
                       label=f"peak $\\mu$ = {popt[0]:.4g} $\\pm$ {mu_err:.2g}")
        else:
            ax.text(0.5, 0.5, "FIT FAILED", transform=ax.transAxes,
                    ha="center", va="center", fontsize=20, color="red", alpha=0.5)

        # Optional vertical line at the seed mu (p0[0]) for visual comparison
        if p0 is not None:
            ax.axvline(p0[0], color="C0", linestyle=":", alpha=0.5,
                       label=f"seed $\\mu_0$ = {p0[0]:.4g}")

        ax.set_xscale("log")
        ax.set_xlabel("x")
        ax.set_ylabel("entries")
        if title:
            ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _init_mpv_storage(self):
        """Initialize nested dicts to store MPV vs jet pT for each case.

        Structure: self.mpv_data[case_key][component][jetpt] = (mu, mu_err)
        case_key examples:
        'basic_{gen}_{partontype}_{z_cut_suffix}_ww{den_weight}'
        'basic_ptRL_{gen}_{partontype}_{z_cut_suffix}_ww{den_weight}'
        'hvp_{partontype}_{z_cut_suffix}_ww{den_weight}'        (components: rad_pythia, rad_herwig, ...)
        'hvp_ptRL_{partontype}_{z_cut_suffix}_ww{den_weight}'
        'qvg_{gen}_{z_cut_suffix}_ww{den_weight}'                (components: rad_quark, rad_gluon, ...)
        'qvg_ptRL_{gen}_{z_cut_suffix}_ww{den_weight}'
        """
        self.mpv_data = {}
        self._mpv_cache = {}   # phys_key -> (mu, mu_err)
    
    def _phys_key(self, gen, partontype, target_jetpt, z_cut, den_weight, ptRL, comp_name):
        cut_suffix = self.get_cut_suffix(z_cut)
        return (gen, partontype, target_jetpt, cut_suffix, den_weight, ptRL, comp_name)

    def _fit_or_get_cached(self, gen, partontype, target_jetpt, z_cut,
                        den_weight, ptRL, comp_name, h, tag,
                        must_exist=False):
        pk = self._phys_key(gen, partontype, target_jetpt, z_cut,
                            den_weight, ptRL, comp_name)
        if pk in self._mpv_cache:
            return self._mpv_cache[pk]          # already fit AND already plotted

        if must_exist:
            print(f"  WARNING: expected cached fit not found, re-fitting {pk}")
        if h is None:
            self._mpv_cache[pk] = (None, None)  # cache the null so we don't retry
            return None, None

        cut_suffix = self.get_cut_suffix(z_cut)
        save_path = self._fit_diag_path(
            f"phys_{gen}_{partontype}_{cut_suffix}_ww{den_weight}"
            + ("_ptRL" if ptRL else ""),
            comp_name, target_jetpt)
        title = (f"{tag} | {gen} {partontype} | jet p_T={target_jetpt} | "
                f"{cut_suffix} | ww{den_weight} | {comp_name}")
        mu, mu_err = self._fit_mpv(h, save_path=save_path, plot_title=title)
        self._mpv_cache[pk] = (mu, mu_err)
        return mu, mu_err

    def _store_mpv(self, case_key, component, jetpt, mu, mu_err):
        if mu is None:
            return
        self.mpv_data.setdefault(case_key, {}).setdefault(component, {})[jetpt] = (mu, mu_err)

    def _fit_diag_path(self, case_key, component, jetpt):
        """Build the path for a fit diagnostic PDF."""
        return os.path.join(
            self.base_plot_dir, "mpv_summary", "fits", case_key,
            f"fit_{case_key}_{component}_jetpt{jetpt}.pdf"
        )

    def _collect_mpv_basic(self, gen, partontype, target_jetpt, z_cut, den_weight):
        """Collect MPVs for the 'basic' (and ptRL) case. This is where fits happen."""
        file = self.get_file(gen)
        cut_suffix = self.get_cut_suffix(z_cut)

        for ptRL, tag in [(False, "basic"), (True, "basic_ptRL")]:
            hists = self.GetSubjetEECHists(file, partontype, target_jetpt, z_cut,
                                        den_weight, ptRL=ptRL)
            case_key = f"{tag}_{gen}_{partontype}_{cut_suffix}_ww{den_weight}"
            for comp_name, h in zip(["full", "rad", "AA", "BB", "AB"], hists):
                mu, mu_err = self._fit_or_get_cached(
                    gen, partontype, target_jetpt, z_cut, den_weight,
                    ptRL, comp_name, h, tag)
                self._store_mpv(case_key, comp_name, target_jetpt, mu, mu_err)

    def _collect_mpv_hvp(self, partontype, target_jetpt, z_cut, den_weight):
        """Collect MPVs for pythia vs herwig (reuses fits done in basic)."""
        cut_suffix = self.get_cut_suffix(z_cut)
        for ptRL, tag in [(False, "hvp"), (True, "hvp_ptRL")]:
            case_key = f"{tag}_{partontype}_{cut_suffix}_ww{den_weight}"
            comp_labels = ["full", "rad", "AA", "BB", "AB"]
            # NOTE: pass gen names that MATCH what basic used ("pythia"/"herwig"),
            # and pass h=None since we expect a cache hit (no fitting needed).
            for label in comp_labels:
                for gen_name in ("pythia", "herwig"):
                    mu, mu_err = self._fit_or_get_cached(
                        gen_name, partontype, target_jetpt, z_cut, den_weight,
                        ptRL, label, h=None, tag=tag, must_exist=True)
                    self._store_mpv(case_key, f"{label}_{gen_name}",
                                    target_jetpt, mu, mu_err)

    def _collect_mpv_qvg(self, gen, target_jetpt, z_cut, den_weight):
        """Collect MPVs for quark vs gluon (reuses fits done in basic)."""
        cut_suffix = self.get_cut_suffix(z_cut)
        for ptRL, tag in [(False, "qvg"), (True, "qvg_ptRL")]:
            case_key = f"{tag}_{gen}_{cut_suffix}_ww{den_weight}"
            comp_labels = ["full", "rad", "AA", "BB", "AB"]
            for label in comp_labels:
                for parton_name in ("quark", "gluon"):
                    mu, mu_err = self._fit_or_get_cached(
                        gen, parton_name, target_jetpt, z_cut, den_weight,
                        ptRL, label, h=None, tag=tag, must_exist=True)
                    self._store_mpv(case_key, f"{label}_{parton_name}",
                                    target_jetpt, mu, mu_err)

    def _plot_mpv_summary(self, case_key, title, xlabel, ylabel, outdir, filename,
                        component_styles=None):
        """Generic MPV summary plot: y = peak position vs x = jet pT.

        component_styles: dict mapping component name -> dict of matplotlib kwargs
                        (color, marker, linestyle, label). If None, auto-assign.
        """
        if case_key not in self.mpv_data:
            return
        data = self.mpv_data[case_key]
        if not data:
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        default_colors = ["k", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"] # black, C0 blue, C1 orange, C2 green, C3 red, C4 purple, C5 brown, C6 pink, C7 gray, C8 olive/yellow-green
        default_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]

        for i, (comp, jetpt_dict) in enumerate(sorted(data.items())):
            if not jetpt_dict:
                continue
            pts  = sorted(jetpt_dict.keys())
            mus  = [jetpt_dict[p][0] for p in pts]
            errs = [jetpt_dict[p][1] for p in pts]

            if component_styles and comp in component_styles:
                style = component_styles[comp]
            else:
                style = dict(color=default_colors[i % len(default_colors)],
                            marker=default_markers[i % len(default_markers)],
                            linestyle="-", label=comp)

            ax.errorbar(pts, mus, yerr=errs, capsize=3, markersize=7, **style)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        fig.tight_layout()

        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename))
        plt.close(fig)


    def plot_all_mpv_summaries(self):
        """Produce all MPV summary PDFs after data has been collected."""
        base = os.path.join(self.base_plot_dir, "mpv_summary")

        # ---- Component styles ----
        basic_styles = {
            "full": dict(color="gray", marker="o", linestyle="-", label="full (passed cut)"),
            "rad":  dict(color="black",    marker="s", linestyle="-", label="radiator"), #color="C3"
            "AA":   dict(color="C0",    marker="D", linestyle="-", label="AxA"),
            "BB":   dict(color="C1",    marker="^", linestyle="-", label="BxB"),
            "AB":   dict(color="C2",    marker="v", linestyle="-", label="AxB"),
        }

        def hvp_styles():
            s = {}
            comp_colors = {"full": "gray", "rad": "black", "AA": "C0", "BB": "C1", "AB": "C2"} #"rad": "C3"
            for comp, color in comp_colors.items():
                s[f"{comp}_pythia"] = dict(color=color, marker="o", linestyle="-",
                                        label=f"{comp} (PYTHIA)")
                s[f"{comp}_herwig"] = dict(color=color, marker="s", linestyle="--",
                                        markerfacecolor="none", markeredgecolor=color,
                                        label=f"{comp} (HERWIG)")
            return s

        def qvg_styles():
            s = {}
            comp_colors = {"full": "gray", "rad": "black", "AA": "C0", "BB": "C1", "AB": "C2"} #"rad": "C3"
            for comp, color in comp_colors.items():
                s[f"{comp}_quark"] = dict(color=color, marker="o", linestyle="-",
                                        label=f"{comp} (quark)")
                s[f"{comp}_gluon"] = dict(color=color, marker="s", linestyle="--",
                                        markerfacecolor="none", markeredgecolor=color,
                                        label=f"{comp} (gluon)")
            return s

        # ---- Iterate over all stored case keys and dispatch to plotter ----
        for case_key in sorted(self.mpv_data.keys()):
            # Parse case key to determine plot type, labels, output directory
            parts = case_key.split("_")
            kind = parts[0]  # 'basic', 'hvp', 'qvg'

            if kind == "basic":
                # basic_{gen}_{partontype}_{cut_suffix}_ww{den_weight}
                # or basic_ptRL_{gen}_{partontype}_{cut_suffix}_ww{den_weight}
                ptRL = (parts[1] == "ptRL")
                offset = 2 if ptRL else 1
                gen        = parts[offset]
                partontype = parts[offset + 1]
                cut_suffix = parts[offset + 2]
                den_token  = parts[offset + 3]  # e.g. 'wwjetpt' or 'wwradpt'
                obs = "p_TR_L" if ptRL else "R_L"
                ylabel = f"Peak position in {obs}"
                title = (f"MPV summary — {gen.upper()} {partontype}, "
                        f"{cut_suffix}, {den_token}" + (" (ptRL)" if ptRL else ""))
                outdir = os.path.join(base, "basic", gen, cut_suffix,
                                    "ptRL" if ptRL else "RL")
                fname  = (f"mpv_basic{'_ptRL' if ptRL else ''}_{gen}_{partontype}"
                        f"_{cut_suffix}_{den_token}.pdf")
                self._plot_mpv_summary(case_key, title, "Jet p_T [GeV/c]",
                                    ylabel, outdir, fname,
                                    component_styles=basic_styles)

            elif kind == "hvp":
                ptRL = (parts[1] == "ptRL")
                offset = 2 if ptRL else 1
                partontype = parts[offset]
                cut_suffix = parts[offset + 1]
                den_token  = parts[offset + 2]
                obs = "p_TR_L" if ptRL else "R_L"
                ylabel = f"Peak position in {obs}"
                title = (f"MPV summary — PYTHIA vs HERWIG, {partontype}, "
                        f"{cut_suffix}, {den_token}" + (" (ptRL)" if ptRL else ""))
                outdir = os.path.join(base, "pythia_vs_herwig", cut_suffix,
                                    "ptRL" if ptRL else "RL")
                fname  = (f"mpv_hvp{'_ptRL' if ptRL else ''}_{partontype}"
                        f"_{cut_suffix}_{den_token}.pdf")
                self._plot_mpv_summary(case_key, title, "Jet p_T [GeV/c]",
                                    ylabel, outdir, fname,
                                    component_styles=hvp_styles())

            elif kind == "qvg":
                ptRL = (parts[1] == "ptRL")
                offset = 2 if ptRL else 1
                gen        = parts[offset]
                cut_suffix = parts[offset + 1]
                den_token  = parts[offset + 2]
                obs = "p_TR_L" if ptRL else "R_L"
                ylabel = f"Peak position in {obs}"
                title = (f"MPV summary — quark vs gluon ({gen.upper()}), "
                        f"{cut_suffix}, {den_token}" + (" (ptRL)" if ptRL else ""))
                outdir = os.path.join(base, "quark_vs_gluon", gen, cut_suffix,
                                    "ptRL" if ptRL else "RL")
                fname  = (f"mpv_qvg{'_ptRL' if ptRL else ''}_{gen}"
                        f"_{cut_suffix}_{den_token}.pdf")
                self._plot_mpv_summary(case_key, title, "Jet p_T [GeV/c]",
                                    ylabel, outdir, fname,
                                    component_styles=qvg_styles())

        print(f"MPV summary plots saved under: {base}")

    

    # -------------------------------------------------------------------------
    # Main plot loop
    # -------------------------------------------------------------------------

    def plot(self):
        self._init_mpv_storage()   # initialize MPV storage before the loop

        for i, target_jetpt in enumerate(self.target_jet_pts):      # Loop over jet pt
            for cut_mode, z_cut in self.cut_modes:                  # Loop over cut modes
                self.cut_mode = cut_mode
                self.z_cut    = z_cut
                print(f"Processing {cut_mode} mode, z_cut={z_cut}, jetpt={target_jetpt}...")

                for partontype in self.partontypes:                 # Loop over parton types
                    # self.plot_jetpt(partontype, target_jetpt, z_cut) # This is currently wrong...

                    for gen in ["pythia", "herwig"]:                # Loop over generators
                        for den_weight in ["jet", "rad"]:           # Loop over weight type
                            
                            if not (partontype in ("quark", "gluon") and gen == "herwig"):
                                self.plot_basic(gen, partontype, target_jetpt,
                                                z_cut, den_weight)
                                    
                                # --- MPV collection: basic (and ptRL) ---
                                self._collect_mpv_basic(gen, partontype, target_jetpt,
                                                        z_cut, den_weight)
                                
                                # --- GET R_G ---
                                if self.cut_mode == "sd":
                                    self.plot_rg(gen, partontype, target_jetpt, z_cut, den_weight)
                            
                            # --- PLOT C_AB,etc FOR PYTHIA ---
                            if gen == "pythia":
                                self.plot_CAB("pythia", partontype, target_jetpt, z_cut)
                                self.plot_CAB_ptRL("pythia", partontype, target_jetpt, z_cut)
                                self.plot_rad_prop("pythia", "pt",   partontype, target_jetpt, z_cut)
                                self.plot_rad_prop("pythia", "lnkt", partontype, target_jetpt, z_cut)
                            
                            # --- PLOT C_AB,etc FOR HERWIG ---
                            if gen == "herwig" and partontype == "inclusive":
                                self.plot_CAB("herwig", partontype, target_jetpt, z_cut)
                                self.plot_CAB_ptRL("herwig", partontype, target_jetpt, z_cut)
                                self.plot_rad_prop("herwig", "pt",   partontype, target_jetpt, z_cut)
                                self.plot_rad_prop("herwig", "lnkt", partontype, target_jetpt, z_cut)
                            
                            if partontype == "inclusive" and gen == "pythia":
                                # --- PYTHIA VS HERWIG ---
                                self.plot_herwig_vs_pythia(partontype, target_jetpt,
                                                        z_cut, den_weight)
                                                
                                # --- QUARKS VS GLUONS ---
                                self.plot_q_vs_g(gen, target_jetpt, z_cut, den_weight)

                            # --- PLOT ACROSS PT ---
                            if i == 0 and partontype == "inclusive" and gen == "pythia":
                                self.plot_herwig_vs_pythia_ratio_acrosspt(z_cut, den_weight)

                    
                
                # ---- All basic fits for this (jetpt, z_cut) are now cached ----
                # NOW it's safe for consumers to read the cache.

                # --- MPV collection: pythia vs herwig ---
                for den_weight in ["jet", "rad"]:
                    self._collect_mpv_hvp("inclusive", target_jetpt, z_cut, den_weight)
                # --- MPV collection: quark vs gluon (after partontype loop, so files are loaded) ---
                # Done per gen, using already-loaded files for this jet pT
                for cut_mode_check in [cut_mode]:  # current cut_mode only
                    for gen in ["pythia"]:
                        for den_weight in ["jet", "rad"]:
                            self._collect_mpv_qvg(gen, target_jetpt, z_cut, den_weight)

        # ---- After all jet pTs processed: render MPV summaries ----
        print("\nGenerating MPV summary plots...")
        self.plot_all_mpv_summaries()



# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    plotter = PlotCurves()
    plotter.plot()

        