#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>

// Example Usage:
// root -l 'analysis/test/plot_eec_ratio.C("output/eec_R04_100_120_PbPb_100GeV.root", "output/eec_R04_100_120_pp_100GeV.root", "100", "120")'

using namespace std;

void SetStyle(Bool_t graypalette = true)
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    if (graypalette)
        gStyle->SetPalette(8, 0);
    else
        gStyle->SetPalette(1);
    gStyle->SetCanvasColor(10);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetFrameLineWidth(1);
    gStyle->SetFrameFillColor(kWhite);
    gStyle->SetPadColor(10);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetHistLineWidth(1);
    gStyle->SetHistLineColor(kRed);
    gStyle->SetFuncWidth(2);
    gStyle->SetFuncColor(kGreen);
    gStyle->SetLineWidth(1);
    gStyle->SetLabelSize(0.045, "xyz");
    gStyle->SetLabelOffset(0.01, "y");
    gStyle->SetLabelOffset(0.01, "x");
    gStyle->SetLabelColor(kBlack, "xyz");
    gStyle->SetTitleSize(0.05, "xyz");
    gStyle->SetTitleOffset(1.25, "y");
    gStyle->SetTitleOffset(1.2, "x");
    gStyle->SetTitleFillColor(kWhite);
    gStyle->SetTextSizePixels(26);
    gStyle->SetTextFont(42);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    gStyle->SetLegendFont(42);
}

void ProcessCanvas(TCanvas *Canvas)
{
    gStyle->SetOptStat(0);
    Canvas->SetHighLightColor(1);
    Canvas->SetFillColor(0);
    Canvas->SetBorderMode(0);
    Canvas->SetBorderSize(2);
    Canvas->SetTickx(1);
    Canvas->SetTicky(1);
    Canvas->SetFrameBorderMode(0);
    Canvas->SetFrameLineWidth(1);
    Canvas->SetFrameBorderMode(1);
    Canvas->SetGridx(1);
    Canvas->SetGridy(1);
}

void FormatHist(TLegend *l, TH1 *hist, std::string text, int markercolor = 1, int markerstyle = 8)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(0.5);
    l->AddEntry(hist, text.c_str(), "pl");

    hist->GetYaxis()->SetTitleOffset(1.05);
    hist->GetYaxis()->SetTitleSize(0.042);
    hist->GetYaxis()->SetLabelSize(0.042);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetXaxis()->SetTitleSize(0.042);
    hist->GetXaxis()->SetLabelSize(0.042);
}

void addLegendInfo(TLegend *l, TString jetR, TString plot_text, TString pt_min, TString pt_max)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "JEWEL " + plot_text, "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 2.76 TeV, #hat{#it{p}}_{T} > 100 GeV", "h");
    l->AddEntry("NULL", "Anti-#it{k}_{T} full jets, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", "#it{p}_{T}^{track} > 1 GeV/#it{c}", "h");
    l->AddEntry("NULL", pt_min + " < #it{p}_{T}^{jet} < " + pt_max + " GeV/#it{c}", "h");
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
}

// Load a TH1D by name from file; warn and return nullptr if missing.
TH1D *LoadHist(TFile *f, const char *histName)
{
    TH1D *h = (TH1D *)f->Get(histName);
    if (!h)
        std::cerr << "WARNING: histogram \"" << histName << "\" not found in " << f->GetName() << std::endl;
    return h;
}

// Compute numerator / denominator ratio; returns nullptr if either is missing.
// The returned histogram is a clone owned by the caller.
TH1D *MakeRatio(TH1D *num, TH1D *den, const char *name)
{
    if (!num || !den) return nullptr;
    TH1D *ratio = (TH1D *)num->Clone(name);
    ratio->Divide(den);
    return ratio;
}

void plot_eec_ratio(std::string pbpb_file, std::string pp_file,
                    std::string pt_min, std::string pt_max)
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string jetR_str = "0.4";

    std::cout << "PbPb file: " << pbpb_file << std::endl;
    std::cout << "pp   file: " << pp_file   << std::endl;
    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << std::endl;

    TFile *f_pbpb = new TFile(pbpb_file.c_str(), "READ");
    TFile *f_pp   = new TFile(pp_file.c_str(),   "READ");

    if (!f_pbpb || f_pbpb->IsZombie()) { std::cerr << "ERROR: cannot open " << pbpb_file << std::endl; return; }
    if (!f_pp   || f_pp->IsZombie())   { std::cerr << "ERROR: cannot open " << pp_file   << std::endl; return; }

    // Load numerator (PbPb) and denominator (pp) histograms
    struct HistConfig {
        const char *name;
        const char *label;
        int         color;
        int         marker;
        TH1D       *pp_hist;
        TH1D       *pbpb_hist;
        TH1D       *ratio_hist;
    };
    std::vector<HistConfig> configs = {
        // {"h_eec_clone", "inclusive EEC", kBlue, kFullCircle, nullptr, nullptr, nullptr},
        // {"h_eec_sdjet_clone", "SD jet EEC", kGray+2, kFullCircle, nullptr, nullptr, nullptr},
        {"h_sdjet_groomed_eec", "SD jet groomed EEC",    kBlack,   kFullCircle, nullptr, nullptr, nullptr},
        {"h_eec_aa_clone",  "EEC,aa", kBlue,    kFullSquare, nullptr, nullptr, nullptr},
        {"h_eec_bb_clone",  "EEC,bb", kRed,     kFullTriangleUp, nullptr, nullptr, nullptr},
        {"h_eec_ab_clone",  "EEC,ab", kGreen+2, kFullDiamond, nullptr, nullptr, nullptr},
    };

    // TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);
    // ProcessCanvas(c2);
    // c2->cd();
    // gPad->SetLogx();
    // gPad->SetLogy();

    // TLegend *l2 = new TLegend(0.2, 0.2, 0.5, 0.4);

    for (auto &cfg : configs) {
        TH1D *h_pp = LoadHist(f_pp, cfg.name);
        if (!h_pp) continue;
        double norm = h_pp->Integral();

        // FormatHist(l2, h_pp, Form("#splitline{%s (%s", cfg.label, "pp)}{normalized by pp}"), cfg.color, cfg.marker);
        h_pp->SetLineStyle(2);
        h_pp->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h_pp->Scale(1.0 / norm);
        // h_pp->Draw("L");
        cfg.pp_hist = h_pp;

        TH1D *h_pbpb = LoadHist(f_pbpb, cfg.name);
        if (!h_pbpb) continue;

        // FormatHist(l2, h_pbpb, Form("#splitline{%s (%s", cfg.label, "PbPb)}{normalized by pp}"), cfg.color, cfg.marker);
        h_pbpb->SetLineStyle(1);
        h_pbpb->GetXaxis()->SetRangeUser(0.005, 0.4);
        h_pbpb->GetXaxis()->SetTitle("#it{R}_{L}");
        h_pbpb->GetYaxis()->SetTitle("#Sigma_{EEC}(#it{R}_{L})");
        // h_pbpb->Scale(1.0 / norm);
        // h_pbpb->Draw("L same");
        cfg.pbpb_hist = h_pbpb;

        TH1D *ratio = MakeRatio(h_pbpb, h_pp, Form("ratio_%s", cfg.name));
        cfg.ratio_hist = ratio;
    }
    // l2->Draw("same");

    // std::string outstem2 = pbpb_file.substr(0, pbpb_file.rfind(".root"));
    // outstem2.replace(outstem2.find("PbPb"), 4, "compare");
    // c2->SaveAs((outstem2 + ".pdf").c_str());
    // delete c2;
    // delete l2;  

    // Canvas
    TCanvas *c = new TCanvas("c", "c", 800, 600);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogx();
    gPad->SetLogy(0);

    // Reference line at 1
    TLine *unity = nullptr; // drawn after first histogram sets the frame

    // Legend
    TLegend *l = new TLegend(0.5, 0.45, 0.9, 0.88);
    // addLegendInfo(l, jetR_str, "", pt_min, pt_max);

    bool first = true;
    for (auto &cfg : configs) {
        TH1D *ratio = cfg.ratio_hist;

        FormatHist(l, ratio, cfg.label, cfg.color, cfg.marker);
        ratio->GetXaxis()->SetRangeUser(0.001, 0.4);
        ratio->GetYaxis()->SetRangeUser(0.5, 2.5);
        ratio->GetXaxis()->SetTitle("#it{R}_{L}");
        ratio->GetYaxis()->SetTitle("PbPb / pp");

        if (first) {
            ratio->Draw("L");
            // Draw unity line after frame is established
            unity = new TLine(0.001, 1.0, 0.4, 1.0);
            unity->SetLineColor(kGray+1);
            unity->SetLineStyle(2);
            unity->SetLineWidth(5);
            unity->Draw("same");
            first = false;
        } else {
            ratio->Draw("L same");
        }
    }
    // l->Draw("same");

    // Derive output path from PbPb input path
    std::string pbpb_base = pbpb_file.substr(pbpb_file.rfind("/") + 1);
    pbpb_base = pbpb_base.substr(0, pbpb_base.rfind(".root"));
    std::string pp_base = pp_file.substr(pp_file.rfind("/") + 1);
    pp_base = pp_base.substr(0, pp_base.rfind(".root"));
    std::string outdir  = pbpb_file.substr(0, pbpb_file.rfind("/") + 1);
    std::string outstem = outdir + pbpb_base + "_ratio_" + pp_base;

    std::string outpdf  = outstem + ".pdf";
    std::string outroot = outstem + ".root";
    c->SaveAs(outpdf.c_str());
    delete c;
    delete l;

    // Save ratio histograms
    TFile *f_out = new TFile(outroot.c_str(), "RECREATE");
    f_out->cd();
    for (auto &cfg : configs)
        if (cfg.ratio_hist) cfg.ratio_hist->Write();
    f_out->Close();

    f_pbpb->Close();
    f_pp->Close();
    delete f_pbpb;
    delete f_pp;
}