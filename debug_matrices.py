#!/usr/bin/env python3
"""
Draw a RooUnfoldResponse (or a plain TH2) and print diagnostics.

USAGE (after roounfold_preload_on / heppy env is set up):
  python draw_response.py
  python draw_response.py -f file.root -n roounfold_response_1D_groomed -o response.pdf
  python draw_response.py -f unfolded1D.root -n matrix1D -o matrix1D.pdf

  python debug_matrices.py -n roounfold_response_1D_groomed_oldversion -o oldversion.pdf
"""

import argparse
import sys
import ROOT
import os

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPaintTextFormat(".2f")

DEFAULT_FILE = ("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/"
                "blianggi/jse/rms/58919321/response_groomed_forunfolding_merged.root")

ROOUNFOLD_DIR = "/global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current"

def load_roounfold():
    # 1) Load the shared library from the heppy RooUnfold build
    lib_candidates = [
        os.path.join(ROOUNFOLD_DIR, "lib", "libRooUnfold.so"),
        os.path.join(ROOUNFOLD_DIR, "lib64", "libRooUnfold.so"),
        os.path.join(ROOUNFOLD_DIR, "libRooUnfold.so"),
        "libRooUnfold",  # fall back to LD_LIBRARY_PATH
    ]
    loaded = False
    for lib in lib_candidates:
        if lib != "libRooUnfold" and not os.path.exists(lib):
            continue
        if ROOT.gSystem.Load(lib) >= 0:
            print(f"Loaded {lib}")
            loaded = True
            break
    if not loaded:
        sys.exit("Error: could not load libRooUnfold. Run roounfold_preload_on or fix ROOUNFOLD_DIR.")

    # 2) Make sure the interpreter knows the class methods (same headers as the C++ macro)
    inc = os.path.join(ROOUNFOLD_DIR, "include")
    ROOT.gInterpreter.AddIncludePath(inc)
    ROOT.gInterpreter.Declare('#include "RooUnfoldResponse.h"')

def print_table(h):
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    print(f"Bin contents (rows = truth/Y bin, cols = measured/X bin; "
          f"0 = underflow, {nx + 1}/{ny + 1} = overflow)")
    print("%8s" % "Y\\X" + "".join(f"{ix:>11d}" for ix in range(nx + 2)))
    for iy in range(ny + 1, -1, -1):
        row = "".join(f"{h.GetBinContent(ix, iy):>11.3g}" for ix in range(nx + 2))
        print(f"{iy:>8d}{row}")


def normalize_by_truth(h_resp, h_true):
    """P(det | part): each truth (Y) row divided by the matching Htruth bin."""
    h_norm = h_resp.Clone("hNorm_draw")
    h_norm.SetDirectory(0)
    for iy in range(1, h_norm.GetNbinsY() + 1):
        t = h_true.GetBinContent(iy)
        for ix in range(1, h_norm.GetNbinsX() + 1):
            h_norm.SetBinContent(ix, iy, h_resp.GetBinContent(ix, iy) / t if t > 0 else 0.0)
            h_norm.SetBinError(ix, iy, 0.0)
    return h_norm


def main():
    parser = argparse.ArgumentParser(description="Draw a RooUnfold response matrix")
    parser.add_argument("-f", "--file", default=DEFAULT_FILE)
    parser.add_argument("-n", "--name", default="roounfold_response_1D_groomed")
    parser.add_argument("-o", "--outpdf", default="response_matrix.pdf")
    args = parser.parse_args()

    load_roounfold()

    fin = ROOT.TFile.Open(args.file)
    if not fin or fin.IsZombie():
        sys.exit(f"Error: cannot open {args.file}")

    obj = fin.Get(args.name)
    if not obj:
        print(f"Error: {args.name} not found. File contents:")
        fin.ls()
        sys.exit(1)
    if obj.InheritsFrom("RooUnfoldResponse") and not hasattr(obj, "Hresponse"):
        sys.exit("Error: RooUnfoldResponse methods not available -> RooUnfold library/dictionary "
                    "not loaded correctly (class is only emulated from the file).")

    # ---------------- Extract histograms ----------------
    if obj.InheritsFrom("RooUnfoldResponse"):
        h_resp = obj.Hresponse().Clone("hResp_draw")
        h_meas = obj.Hmeasured().Clone("hMeas_draw")
        h_true = obj.Htruth().Clone("hTrue_draw")
    elif obj.InheritsFrom("TH2"):
        h_resp = obj.Clone("hResp_draw")
        h_meas = h_resp.ProjectionX("hMeas_draw", 0, -1)
        h_true = h_resp.ProjectionY("hTrue_draw", 0, -1)
    else:
        sys.exit(f"Error: {args.name} is a {obj.ClassName()}, not a RooUnfoldResponse or TH2")

    for h in (h_resp, h_meas, h_true):
        h.SetDirectory(0)

    # ---------------- Diagnostics ----------------
    xa, ya = h_resp.GetXaxis(), h_resp.GetYaxis()
    print(f"Object: {args.name} ({obj.ClassName()})")
    print(f"Hresponse: {h_resp.GetNbinsX()} x {h_resp.GetNbinsY()}, "
          f"X [{xa.GetXmin()}, {xa.GetXmax()}], Y [{ya.GetXmin()}, {ya.GetXmax()}]")
    print(f"  entries                = {h_resp.GetEntries():.6g}")
    print(f"  integral (in range)    = {h_resp.Integral():.6g}")
    print(f"  integral (incl. flow)  = {h_resp.Integral(0, -1, 0, -1):.6g}")
    print(f"  min / max bin content  = {h_resp.GetMinimum():.6g} / {h_resp.GetMaximum():.6g}")
    print(f"Hmeasured integral = {h_meas.Integral():.6g} (incl. flow {h_meas.Integral(0, -1):.6g})")
    print(f"Htruth    integral = {h_true.Integral():.6g} (incl. flow {h_true.Integral(0, -1):.6g})")
    print_table(h_resp)

    h_norm = normalize_by_truth(h_resp, h_true)

    # ---------------- Draw ----------------
    c = ROOT.TCanvas("c", "c", 1200, 1000)
    c.Divide(2, 2)

    c.cd(1)
    ROOT.gPad.SetRightMargin(0.15)
    if h_resp.GetMaximum() > 0:
        ROOT.gPad.SetLogz()
    h_resp.SetTitle("Response matrix (raw);p_{T}^{det} (GeV/c);p_{T}^{part} (GeV/c)")
    h_resp.Draw("colz")

    c.cd(2)
    ROOT.gPad.SetRightMargin(0.15)
    h_norm.SetTitle("Normalized: P(det | part);p_{T}^{det} (GeV/c);p_{T}^{part} (GeV/c)")
    h_norm.SetMinimum(0)
    h_norm.Draw("colz text")

    c.cd(3)
    ROOT.gPad.SetLogy()
    h_meas.SetTitle("Hmeasured;p_{T}^{det} (GeV/c);Counts")
    h_meas.SetLineColor(ROOT.kBlue)
    h_meas.Draw("hist e")

    c.cd(4)
    ROOT.gPad.SetLogy()
    h_true.SetTitle("Htruth;p_{T}^{part} (GeV/c);Counts")
    h_true.SetLineColor(ROOT.kRed)
    h_true.Draw("hist e")

    c.SaveAs(args.outpdf)
    print(f"Saved {args.outpdf}")

    fin.Close()


if __name__ == "__main__":
    main()