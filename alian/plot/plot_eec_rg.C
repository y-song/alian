#include <typeinfo>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>

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
    gStyle->SetHistLineWidth(2);
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

TLine *drawVertLine(double x1, double y1, double y2, int color, int linestyle = 2)
{
    auto fvertline = new TLine(x1, y1, x1, y2);
    fvertline->SetLineWidth(1);
    fvertline->SetLineColor(color);
    fvertline->SetLineStyle(linestyle);
    return fvertline;
}

TLine *drawHoriLine(double x1, double x2, double y1, int color, int linestyle = 2)
{
    auto fhoriline = new TLine(x1, y1, x2, y1);
    fhoriline->SetLineWidth(1);
    fhoriline->SetLineColor(color);
    fhoriline->SetLineStyle(linestyle);
    return fhoriline;
}

void addLegendInfo(TLegend *l, TString jetR, TString plot_text)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "JEWEL " + plot_text, "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 2.76 TeV, #hat{#it{p}}_{T} > 100 GeV", "h");
    l->AddEntry("NULL", "Anti-#it{k}_{T} full jets, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", "#it{p}_{T}^{track} > 1 GeV/#it{c}", "h");
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
}

TH1D *DivideByBinWidth(TH1D *input_hist)
{
    TH1D *output_hist = (TH1D *)input_hist->Clone(Form("%s_clone", input_hist->GetName()));
    for (int ibin = 1; ibin < input_hist->GetNbinsX() + 1; ibin++)
    {
        double bincontent = input_hist->GetBinContent(ibin);
        double binerror   = input_hist->GetBinError(ibin);
        double binwidth   = input_hist->GetBinWidth(ibin);
        output_hist->SetBinContent(ibin, bincontent / binwidth);
        output_hist->SetBinError(ibin, binerror / binwidth);
    }
    return output_hist;
}

// Helper: load a named TH2 from file, project onto y-axis for the given pT range,
// divide by bin width, and normalize by njets. Returns nullptr if histogram not found.
TH1D *LoadAndProcess(TFile *f, const char *histName, double pt_lo, double pt_hi, double njets, double scale=1.0)
{
    TH2 *h2 = (TH2 *)f->Get(histName);
    if (!h2) {
        std::cerr << "WARNING: histogram \"" << histName << "\" not found in file, skipping." << std::endl;
        return nullptr;
    }
    TH2 *h2_cut = (TH2 *)h2->Clone(Form("%s_cut", histName));
    h2_cut->GetXaxis()->SetRangeUser(pt_lo, pt_hi);
    TH1D *proj  = h2_cut->ProjectionY(Form("%s_proj", histName));
    TH1D *hdivw = DivideByBinWidth((TH1D *)proj->Clone(Form("h_%s", histName)));
    hdivw->Scale(scale / njets);
    return hdivw;
}

void plot_eec_rg(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "")
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string infile  = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    std::string jetR    = "04";
    std::string jetR_str = "0.4";
    std::string outfile = "/home/youqi/alian/alian/output/rg_eec_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + ".root";

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << std::endl;
    std::cout << "R = " << jetR_str << std::endl;

    double pt_lo = stof(pt_min);
    double pt_hi = stof(pt_max);

    TFile *f     = new TFile(infile.c_str(), "READ");
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    // Get jet pT histogram and compute njets once — shared normalization for all EEC variants
    TH1 *h_JetPt = (TH1 *)f->Get("jet_pT");
    std::cout << "Number of jets from pT bins " << pt_lo + 1 << "-" << pt_hi << ": ";
    double njets = h_JetPt->Integral((int)h_JetPt->FindBin(pt_lo), (int)h_JetPt->FindBin(pt_hi));
    std::cout << njets << std::endl;

    TH1D *h_jet_eec = LoadAndProcess(f, "eec",pt_lo, pt_hi, njets);
    TH1D *h_sdjet_eec = LoadAndProcess(f, "sdjet_eec",pt_lo, pt_hi, njets);
    TH1D *h_aa  = LoadAndProcess(f, "eec_aa", pt_lo, pt_hi, njets);
    TH1D *h_bb  = LoadAndProcess(f, "eec_bb", pt_lo, pt_hi, njets);
    TH1D *h_ab  = LoadAndProcess(f, "eec_ab", pt_lo, pt_hi, njets, 2.0);
    TH1D *h_ab_noew  = LoadAndProcess(f, "eec_ab_noew", pt_lo, pt_hi, njets, 2.0);
    TH1D *h_rg  = LoadAndProcess(f, "rg_log", pt_lo, pt_hi, njets);
    TH1D *h_zg  = LoadAndProcess(f, "zg", pt_lo, pt_hi, njets);

    TH1D *h_sdjet_groomed_eec;
    if (h_aa && h_bb && h_ab) {
        h_sdjet_groomed_eec = (TH1D *)h_aa->Clone("h_sdjet_groomed_eec");
        h_sdjet_groomed_eec->Add(h_bb);
        h_sdjet_groomed_eec->Add(h_ab); // eec_ab was already scaled by 2
    }
    // {hist name in file, legend label, color, marker style}
    struct HistConfig {
        const char *label;
        int         color;
        int         marker;
        TH1D       *hist;
    };
    std::vector<HistConfig> configs = {
        // {"inclusive EEC", kBlue, kFullCircle , h_jet_eec},
        // {"SD jet EEC", kGray+2, kFullCircle , h_sdjet_eec},
        // {"SD jet groomed EEC", kBlack, kFullCircle, h_sdjet_groomed_eec},
        // {"EEC,aa", kBlue,     kFullSquare    , h_aa},
        // {"EEC,bb", kRed,      kFullTriangleUp, h_bb},
        {"Rg",     kRed,      kFullTriangleUp, h_rg},
        {"EEC,ab", kGreen+2,  kFullDiamond   , h_ab},
        {"EEC,ab, no weight", kBlack,  kFullDiamond, h_ab_noew},
    };

    // Canvas
    TCanvas *c = new TCanvas("c", "c", 800, 600);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogx();

    TH1F *frame = new TH1F("frame", "My Normalized Plot", 100, 0.001, 0.5);
    frame->SetMaximum(0.1); 
    frame->SetMinimum(0.0);
    frame->GetXaxis()->SetTitle("#it{R}_{L}");
    frame->Draw();

    // Legend — info block + per-histogram entries
    TLegend *l = new TLegend(0.5, 0.45, 0.9, 0.88);
    std::string ptbin = pt_min + " < #it{p}_{T}^{jet} < " + pt_max + " GeV/#it{c}";
    addLegendInfo(l, jetR_str, plot_text);
    l->AddEntry("NULL", ptbin.c_str(), "h");

    // Draw: first valid histogram uses "L" to set the frame; rest use "L same"
    bool first = true;
    for (auto &cfg : configs) {
        if (!cfg.hist)
            continue;

        TH1D *h = cfg.hist;
        FormatHist(l, h, cfg.label, cfg.color, cfg.marker);
        h->GetXaxis()->SetRangeUser(0.001, 0.7);
        h->GetYaxis()->SetRangeUser(0, 8);

        if (first) {
            h->DrawNormalized("same");
            first = false;
        } else {
            h->DrawNormalized("same");
        }
    }
    // h_total->Draw("same");
    // l->Draw("same");

    // Save canvas
    std::string fname = "/home/youqi/alian/alian/output/rg_eec_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + ".pdf";
    c->SaveAs(fname.c_str());
    delete c;
    delete l;

    // Save processed histograms to output file
    f_out->cd();

    h_jet_eec->Write();
    h_sdjet_eec->Write();
    h_sdjet_groomed_eec->Write();

    h_aa->Write();
    h_bb->Write();
    h_ab->Write();
    h_ab_noew->Write();
    h_rg->Write(); 
    h_zg->Write(); 

    f_out->Close();
    f->Close();
    delete f;
}