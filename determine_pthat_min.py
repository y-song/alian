#!/usr/bin/env python3
import os
import sys
import re
import ROOT
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

input_dir = "/global/cfs/cdirs/alice/blianggi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/jse/testing"
output_dir = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/pthat_determ"
filenames = {
    "40": "AnalysisResults_pthatmin40.root",
    "45": "AnalysisResults_pthatmin45.root",
    "48": "AnalysisResults_pthatmin48.root",
    "50": "AnalysisResults_pthatmin50.root",
}
log_files = {
    "40": os.path.join(input_dir, "output_pthatmin40.txt"),
    "45": os.path.join(input_dir, "output_pthatmin45.txt"),
    "48": os.path.join(input_dir, "output_pthatmin48.txt"),
    "50": os.path.join(input_dir, "output_pthatmin50.txt"),
}
hist_names = ["hjetpT_h", "hjetpT_ha", "hjetpT_chjet"]
hist_titles = ["Full jet p_{T}", "Full jet p_{T} (ALICE)", "Charged jet p_{T}"]
hist_x_labels = ["p_{T, full jet} [GeV/c]", "p_{T, full jet} [GeV/c]", "p_{T, ch jet} [GeV/c]"]
alice_specs = ["", "p_{T}^{track} > 150 MeV, |#eta^{track}| < 0.9", "p_{T}^{track} > 150 MeV, |#eta^{track}| < 0.9"]
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]
alphas = [1.0, 1.0, 1.0, 1.0]

os.makedirs(output_dir, exist_ok=True)


############################################################################################################################################
# Helper functions
############################################################################################################################################


def parse_pythia_xsec(log_file):
    """Parse total cross section from a PYTHIA output log file."""
    with open(log_file, 'r') as f:
        for line in f:
            if re.search(r'\|\s+sum\s+\|', line):
                parts = line.split('|')
                sigma = float(parts[3].split()[0])
                return sigma
    raise ValueError(f"Could not parse cross section from {log_file}")

def get_scale_factor(root_file, sigma):
    """Get scale factor from hNevents histogram and cross section."""
    h_nev = root_file.Get("hNevents")
    if not h_nev:
        raise ValueError("hNevents histogram not found in file")
    n_events = h_nev.GetBinContent(1)
    return sigma / n_events


def rebin_histogram_uniform(hist, rebin_factor=4):
    """Rebin histogram uniformly by a fixed bin factor."""
    new_hist = hist.Rebin(rebin_factor, f"{hist.GetName()}_rebin")
    new_hist.SetDirectory(0)
    new_hist.SetLineColor(hist.GetLineColor())
    new_hist.SetLineWidth(hist.GetLineWidth())
    return new_hist

def compute_scale_factors():
    """Compute scale factors for all pt-hat-min values."""
    scale_factors = {}
    for label, filename in filenames.items():
        path = os.path.join(input_dir, filename)
        root_file = ROOT.TFile.Open(path)
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file: {path}")
        sigma = parse_pythia_xsec(log_files[label])
        scale_factors[label] = get_scale_factor(root_file, sigma)
        print(f"pt-hat-min={label}: sigma={sigma:.3e} mb, n_events={root_file.Get('hNevents').GetBinContent(1):.0f}, scale={scale_factors[label]:.3e} mb/event")
        root_file.Close()
    return scale_factors

def load_histogram(filename, hist_name, scale_factor=None):
    """Load and clone a histogram from a ROOT file, optionally scaling by cross section."""
    path = os.path.join(input_dir, filename)
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Cannot open: {path}")

    hist = root_file.Get(hist_name)
    if not hist:
        raise ValueError(f"Histogram '{hist_name}' not found in {filename}")

    hist_clone = hist.Clone(f"{hist_name}_clone_{id(hist)}")
    hist_clone.SetDirectory(0)
    if scale_factor is not None:
        hist_clone.Scale(scale_factor)
    root_file.Close()
    return hist_clone


############################################################################################################################################
# Entries plots (top-level, existing)
############################################################################################################################################


# Load all histograms (unscaled, for entries plots)
all_histograms = {}
for hist_name in hist_names:
    all_histograms[hist_name] = {}
    for idx, (label, filename) in enumerate(filenames.items()):
        path = os.path.join(input_dir, filename)
        root_file = ROOT.TFile.Open(path)
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file: {path}")

        hist = root_file.Get(hist_name)
        if not hist:
            raise ValueError(f"Histogram '{hist_name}' not found in file: {path}")

        hist_clone = hist.Clone(f"{hist_name}_{label}")
        hist_clone.SetDirectory(0)
        hist_clone.SetLineColor(colors[idx])
        hist_clone.SetLineWidth(2)
        all_histograms[hist_name][label] = hist_clone
        root_file.Close()

# Create canvas with 3 subpads - larger
canvas = ROOT.TCanvas("c_pthatmin", "pthatmin comparison", 2800, 800)
canvas.Divide(3, 1)
all_lines = []

for pad_idx, (hist_name, hist_title, hist_x_label, alice_spec) in enumerate(zip(hist_names, hist_titles, hist_x_labels, alice_specs)):
    pad = canvas.cd(pad_idx + 1)
    pad.SetLeftMargin(0.16)
    pad.SetBottomMargin(0.16)
    pad.SetRightMargin(0.05)
    pad.SetTopMargin(0.12)

    histograms = list(all_histograms[hist_name].items())
    first_hist = histograms[0][1]
    first_hist.Draw("hist")
    first_hist.GetXaxis().SetTitle(hist_x_label)
    first_hist.GetYaxis().SetTitle("Entries")
    first_hist.GetXaxis().SetTitleSize(0.06)
    first_hist.GetYaxis().SetTitleSize(0.06)
    first_hist.GetXaxis().SetLabelSize(0.04)
    first_hist.GetYaxis().SetLabelSize(0.05)
    first_hist.SetTitle(hist_title)
    first_hist.GetXaxis().CenterTitle()
    first_hist.GetYaxis().CenterTitle()

    legend = ROOT.TLegend(0.52, 0.58, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.045)

    for label, hist in histograms:
        if hist is not first_hist:
            hist.Draw("hist same")
        legend.AddEntry(hist, f"#hat{{p}}_{{T,min}} = {label} GeV", "l")

    legend.Draw("same")

    pad.SetLogy(1)
    first_hist.SetMinimum(1)

    pad.Update()
    pad.Modified()
    pad.Update()
    hist_min = first_hist.GetMinimum()
    hist_max = first_hist.GetMaximum()
    line = ROOT.TLine(50.0, hist_min, 50.0, hist_max)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.SetLineStyle(ROOT.kDashed)
    all_lines.append(line)
    all_lines[pad_idx].Draw("same")

    # Add ALICE specifications text if present
    if alice_spec:
        spec_text = ROOT.TLatex()
        spec_text.SetNDC()
        spec_text.SetTextSize(0.04)
        spec_text.DrawLatex(0.5, 0.52, alice_spec)

canvas.Update()
output_path = os.path.join(output_dir, "hjetpT_all_pthatmin_comparison.pdf")
canvas.SaveAs(output_path)
print(f"Saved plot to: {output_path}")


############################################################################################################################################
# Functions
############################################################################################################################################


def plot_pthat_comparison():
    # Load all histograms
    all_histograms = {}
    for hist_name in hist_names:
        all_histograms[hist_name] = {}
        for idx, (label, filename) in enumerate(filenames.items()):
            path = os.path.join(input_dir, filename)
            root_file = ROOT.TFile.Open(path)
            if not root_file or root_file.IsZombie():
                raise FileNotFoundError(f"Cannot open ROOT file: {path}")

            hist = root_file.Get(hist_name)
            if not hist:
                raise ValueError(f"Histogram '{hist_name}' not found in file: {path}")

            hist_clone = hist.Clone(f"{hist_name}_{label}")
            hist_clone.SetDirectory(0)
            hist_clone.SetLineColor(colors[idx])
            hist_clone.SetLineWidth(2)
            all_histograms[hist_name][label] = hist_clone
            root_file.Close()

    # Create canvas with 3 subpads - larger
    canvas = ROOT.TCanvas("c_pthatmin", "pthatmin comparison", 2800, 800)
    canvas.Divide(3, 1)
    all_lines = []

    for pad_idx, (hist_name, hist_title, hist_x_label, alice_spec) in enumerate(zip(hist_names, hist_titles, hist_x_labels, alice_specs)):
        pad = canvas.cd(pad_idx + 1)
        pad.SetLeftMargin(0.16)
        pad.SetBottomMargin(0.16)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.12)

        histograms = list(all_histograms[hist_name].items())
        first_hist = histograms[0][1]
        first_hist.Draw("hist")
        first_hist.GetXaxis().SetTitle(hist_x_label)
        first_hist.GetYaxis().SetTitle("Entries")
        first_hist.GetXaxis().SetTitleSize(0.06)
        first_hist.GetYaxis().SetTitleSize(0.06)
        first_hist.GetXaxis().SetLabelSize(0.04)
        first_hist.GetYaxis().SetLabelSize(0.05)
        first_hist.SetTitle(hist_title)
        first_hist.GetXaxis().CenterTitle()
        first_hist.GetYaxis().CenterTitle()

        legend = ROOT.TLegend(0.52, 0.58, 0.90, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.045)

        for label, hist in histograms:
            if hist is not first_hist:
                hist.Draw("hist same")
            legend.AddEntry(hist, f"#hat{{p}}_{{T,min}} = {label} GeV", "l")

        legend.Draw("same")

        pad.SetLogy(1)
        first_hist.SetMinimum(1)

        pad.Update()
        pad.Modified()
        pad.Update()
        hist_min = first_hist.GetMinimum()
        hist_max = first_hist.GetMaximum()
        line = ROOT.TLine(50.0, hist_min, 50.0, hist_max)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(2)
        line.SetLineStyle(ROOT.kDashed)
        all_lines.append(line)
        all_lines[pad_idx].Draw("same")

        # Add ALICE specifications text if present
        if alice_spec:
            spec_text = ROOT.TLatex()
            spec_text.SetNDC()
            spec_text.SetTextSize(0.04)
            spec_text.DrawLatex(0.5, 0.52, alice_spec)

    canvas.Update()
    output_path = os.path.join(output_dir, "hjetpT_all_pthatmin_comparison.pdf")
    canvas.SaveAs(output_path)
    print(f"Saved plot to: {output_path}")

def plot_pthat_comparison_xsec(scale_factors):
    """Same as plot_pthat_comparison but scaled to cross section."""
    # Load all histograms scaled by cross section
    all_histograms = {}
    for hist_name in hist_names:
        all_histograms[hist_name] = {}
        for idx, (label, filename) in enumerate(filenames.items()):
            path = os.path.join(input_dir, filename)
            root_file = ROOT.TFile.Open(path)
            if not root_file or root_file.IsZombie():
                raise FileNotFoundError(f"Cannot open ROOT file: {path}")

            hist = root_file.Get(hist_name)
            if not hist:
                raise ValueError(f"Histogram '{hist_name}' not found in file: {path}")

            hist_clone = hist.Clone(f"{hist_name}_{label}_xsec")
            hist_clone.SetDirectory(0)
            hist_clone.Scale(scale_factors[label])
            hist_clone.SetLineColor(colors[idx])
            hist_clone.SetLineWidth(2)
            all_histograms[hist_name][label] = hist_clone
            root_file.Close()

    # Create canvas with 3 subpads - larger
    canvas = ROOT.TCanvas("c_pthatmin_xsec", "pthatmin cross section comparison", 2800, 800)
    canvas.Divide(3, 1)
    all_lines = []

    for pad_idx, (hist_name, hist_title, hist_x_label, alice_spec) in enumerate(zip(hist_names, hist_titles, hist_x_labels, alice_specs)):
        pad = canvas.cd(pad_idx + 1)
        pad.SetLeftMargin(0.16)
        pad.SetBottomMargin(0.16)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.12)

        histograms = list(all_histograms[hist_name].items())
        first_hist = histograms[0][1]
        first_hist.Draw("hist")
        first_hist.GetXaxis().SetTitle(hist_x_label)
        first_hist.GetYaxis().SetTitle("d#sigma/dp_{T} [mb/(GeV/c)]")
        first_hist.GetXaxis().SetTitleSize(0.06)
        first_hist.GetYaxis().SetTitleSize(0.05)
        first_hist.GetXaxis().SetLabelSize(0.04)
        first_hist.GetYaxis().SetLabelSize(0.05)
        first_hist.SetTitle(hist_title)
        first_hist.GetXaxis().CenterTitle()
        first_hist.GetYaxis().CenterTitle()

        legend = ROOT.TLegend(0.52, 0.58, 0.90, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.045)

        for label, hist in histograms:
            if hist is not first_hist:
                hist.Draw("hist same")
            legend.AddEntry(hist, f"#hat{{p}}_{{T,min}} = {label} GeV", "l")

        legend.Draw("same")

        pad.SetLogy(1)
        first_hist.SetMinimum(1e-7)

        pad.Update()
        pad.Modified()
        pad.Update()
        hist_min = first_hist.GetMinimum()
        hist_max = first_hist.GetMaximum()
        line = ROOT.TLine(50.0, hist_min, 50.0, hist_max)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(2)
        line.SetLineStyle(ROOT.kDashed)
        all_lines.append(line)
        all_lines[pad_idx].Draw("same")

        # Add ALICE specifications text if present
        if alice_spec:
            spec_text = ROOT.TLatex()
            spec_text.SetNDC()
            spec_text.SetTextSize(0.04)
            spec_text.DrawLatex(0.5, 0.52, alice_spec)

    canvas.Update()
    output_path = os.path.join(output_dir, "hjetpT_all_pthatmin_xsec_comparison.pdf")
    canvas.SaveAs(output_path)
    print(f"Saved plot to: {output_path}")

def plot_leftmost_pthat_xsec(scale_factors):
    """Create a single-canvas PDF containing only the leftmost (first) xsec sub-plot."""
    # Load and scale histograms for the first hist_name only
    hist_name = hist_names[0]
    all_histograms = {}
    all_histograms[hist_name] = {}
    for idx, (label, filename) in enumerate(filenames.items()):
        path = os.path.join(input_dir, filename)
        root_file = ROOT.TFile.Open(path)
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file: {path}")

        hist = root_file.Get(hist_name)
        if not hist:
            raise ValueError(f"Histogram '{hist_name}' not found in file: {path}")

        hist_clone = hist.Clone(f"{hist_name}_{label}_xsec_single")
        hist_clone.SetDirectory(0)
        hist_clone.Scale(scale_factors[label])
        hist_clone = rebin_histogram_uniform(hist_clone, rebin_factor=5)
        hist_clone.SetLineColor(colors[idx])
        hist_clone.SetLineWidth(2)
        all_histograms[hist_name][label] = hist_clone
        root_file.Close()

    histograms = list(all_histograms[hist_name].items())
    first_hist = histograms[0][1]
    base_hist = all_histograms[hist_name]["40"]

    # Original xsec plot
    canvas = ROOT.TCanvas("c_pthatmin_left", "left pthatmin cross section", 900, 800)
    canvas.SetLeftMargin(0.16)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.12)
    canvas.SetBottomMargin(0.16)

    first_hist.Draw("hist")
    first_hist.GetXaxis().SetTitle(hist_x_labels[0])
    first_hist.GetYaxis().SetTitle("d#sigma/dp_{T} [mb/(GeV/c)]")
    first_hist.GetXaxis().SetTitleSize(0.06)
    first_hist.GetYaxis().SetTitleSize(0.05)
    first_hist.GetXaxis().SetLabelSize(0.04)
    first_hist.GetYaxis().SetLabelSize(0.05)
    first_hist.SetTitle(hist_titles[0])
    first_hist.GetXaxis().CenterTitle()
    first_hist.GetYaxis().CenterTitle()

    legend = ROOT.TLegend(0.52, 0.58, 0.90, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.045)

    ratio_histograms = []
    for label, hist in histograms:
        if hist is not first_hist:
            hist.Draw("hist same")
        legend.AddEntry(hist, f"#hat{{p}}_{{T,min}} = {label} GeV", "l")

        ratio = hist.Clone(f"{hist.GetName()}_ratio_40")
        ratio.SetDirectory(0)
        ratio.Divide(base_hist)
        ratio.SetLineColor(hist.GetLineColor())
        ratio.SetLineWidth(hist.GetLineWidth())
        ratio_histograms.append((label, ratio))

    legend.Draw("same")
    first_hist.SetMinimum(1e-7)
    canvas.SetLogy(1)

    canvas.Update()
    hist_min = first_hist.GetMinimum()
    hist_max = first_hist.GetMaximum()
    line = ROOT.TLine(50.0, hist_min, 50.0, hist_max)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.SetLineStyle(ROOT.kDashed)
    line.Draw("same")

    output_path = os.path.join(output_dir, "hjetpT_pthatmin_xsec_comparison.pdf")
    canvas.SaveAs(output_path)
    print(f"Saved plot to: {output_path}")

    # Ratio-only plot
    ratio_canvas = ROOT.TCanvas("c_pthatmin_ratio", "left pthatmin cross section ratio", 900, 800)
    ratio_canvas.SetLeftMargin(0.16)
    ratio_canvas.SetRightMargin(0.05)
    ratio_canvas.SetTopMargin(0.12)
    ratio_canvas.SetBottomMargin(0.16)
    ratio_canvas.SetGridy(1)

    ratio_histograms[0][1].SetTitle("Ratio to pthat,min = 40 GeV")
    ratio_histograms[0][1].GetXaxis().SetTitle(hist_x_labels[0])
    ratio_histograms[0][1].GetYaxis().SetTitle("Ratio to #hat{p}_{T,min} = 40 GeV")
    ratio_histograms[0][1].GetXaxis().SetTitleSize(0.06)
    ratio_histograms[0][1].GetYaxis().SetTitleSize(0.06)
    ratio_histograms[0][1].GetXaxis().SetLabelSize(0.04)
    ratio_histograms[0][1].GetYaxis().SetLabelSize(0.05)
    ratio_histograms[0][1].GetYaxis().SetNdivisions(505)
    ratio_histograms[0][1].SetMinimum(0.2)
    ratio_histograms[0][1].SetMaximum(1.8)
    ratio_histograms[0][1].Draw("hist")

    for label, ratio in ratio_histograms[1:]:
        ratio.Draw("hist same")

    line_one = ROOT.TLine(ratio_histograms[0][1].GetXaxis().GetXmin(), 1.0,
                         ratio_histograms[0][1].GetXaxis().GetXmax(), 1.0)
    line_one.SetLineColor(ROOT.kBlack)
    line_one.SetLineStyle(ROOT.kDashed)
    line_one.Draw("same")

    ratio_min = ratio_histograms[0][1].GetMinimum()
    ratio_max = ratio_histograms[0][1].GetMaximum()
    line_50 = ROOT.TLine(50.0, ratio_min, 50.0, ratio_max)
    line_50.SetLineColor(ROOT.kBlack)
    line_50.SetLineWidth(2)
    line_50.SetLineStyle(ROOT.kDashed)
    line_50.Draw("same")

    ratio_output_path = os.path.join(output_dir, "hjetpT_pthatmin_xsec_comparison_ratio.pdf")
    ratio_canvas.SaveAs(ratio_output_path)
    print(f"Saved ratio plot to: {ratio_output_path}")

def plot_matched_comparison(hist_name_unmatched, hist_name_matched, title, x_label, output_name, scale_factors=None):
    """Plot unmatched vs matched histograms, one pad per pt_hat_min value"""
    canvas = ROOT.TCanvas("c_matched", title, 1600, 1200)
    canvas.Divide(2, 2)  # 2x2 grid for 4 pt_hat_min values

    all_objects = []  # Keep references alive

    for idx, (label, filename) in enumerate(filenames.items()):
        pad = canvas.cd(idx + 1)  # ROOT pads are 1-indexed
        pad.SetLeftMargin(0.14)
        pad.SetBottomMargin(0.14)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.12)
        pad.SetLogy()

        # Load histograms
        scale_factor = scale_factors[label] if scale_factors is not None else None
        hist_unmatched = load_histogram(filename, hist_name_unmatched, scale_factor)
        hist_matched = load_histogram(filename, hist_name_matched, scale_factor)

        hist_unmatched.SetLineColorAlpha(colors[idx], alphas[idx])
        hist_unmatched.SetLineWidth(2)
        hist_unmatched.SetLineStyle(ROOT.kSolid)

        hist_matched.SetLineColorAlpha(colors[idx], alphas[idx])
        hist_matched.SetLineWidth(2)
        hist_matched.SetLineStyle(ROOT.kDashed)

        all_objects.extend([hist_unmatched, hist_matched])

        # Draw
        hist_unmatched.Draw("hist")
        hist_matched.Draw("hist same")

        hist_max = max(hist_unmatched.GetMaximum(), hist_matched.GetMaximum())

        # Setup axes
        hist_unmatched.GetXaxis().SetTitle(x_label)
        y_label = "d#sigma/dp_{T} [mb/(GeV/c)]" if scale_factors is not None else "Entries"
        hist_unmatched.GetYaxis().SetTitle(y_label)
        hist_unmatched.GetXaxis().SetTitleSize(0.05)
        hist_unmatched.GetYaxis().SetTitleSize(0.05)
        hist_unmatched.GetXaxis().SetLabelSize(0.04)
        hist_unmatched.GetYaxis().SetLabelSize(0.04)
        hist_unmatched.SetTitle(f"{title}, #hat{{p}}_{{T,min}}={label} GeV")

        # Legend per pad
        legend = ROOT.TLegend(0.40, 0.65, 0.92, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.04)
        legend.AddEntry(hist_unmatched, f"All, #hat{{p}}_{{T,min}}={label} GeV", "l")
        legend.AddEntry(hist_matched,   f"Matched, #hat{{p}}_{{T,min}}={label} GeV", "l")
        legend.Draw()
        all_objects.append(legend)

        # Draw vertical line
        hist_min = hist_unmatched.GetMinimum()
        line = ROOT.TLine(50.0, hist_min, 50.0, hist_max * 1.5)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(2)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw("same")
        all_objects.append(line)

        pad.Update()

    canvas.Update()
    output_path = os.path.join(output_dir, output_name)
    canvas.SaveAs(output_path)
    print(f"Saved: {output_path}")


############################################################################################################################################
# Run all plots
############################################################################################################################################


# Compute scale factors once for all cross section plots
scale_factors = compute_scale_factors()

# Entries plots
plot_pthat_comparison()
plot_matched_comparison("hjetpT_chjet", "hjetpT_chjet_matchedtoparton",
                        "Charged jet p_{T}: All vs Matched to Parton",
                        "p_{T, ch jet} [GeV/c]",
                        "hjetpT_chjet_matched_comparison.pdf")
plot_matched_comparison("hjetpT_h", "hjetpT_h_matchedtoparton",
                        "Full jet p_{T}: All vs Matched to Parton",
                        "p_{T, full jet} [GeV/c]",
                        "hjetpT_h_matched_comparison.pdf")

# Cross section plots
plot_pthat_comparison_xsec(scale_factors)
plot_leftmost_pthat_xsec(scale_factors)
plot_matched_comparison("hjetpT_chjet", "hjetpT_chjet_matchedtoparton",
                        "Charged jet p_{T}: All vs Matched to Parton",
                        "p_{T, ch jet} [GeV/c]",
                        "hjetpT_chjet_matched_xsec_comparison.pdf",
                        scale_factors=scale_factors)
plot_matched_comparison("hjetpT_h", "hjetpT_h_matchedtoparton",
                        "Full jet p_{T}: All vs Matched to Parton",
                        "p_{T, full jet} [GeV/c]",
                        "hjetpT_h_matched_xsec_comparison.pdf",
                        scale_factors=scale_factors)

print("Done!")