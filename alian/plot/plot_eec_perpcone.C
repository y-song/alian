#include <typeinfo>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>
// Perp-cone background subtraction closure for the matched combined subjet EECs
// filled by analysis/test/embed_perpcone.py (combined_matched only).
// For each of aa, bb, ab and their sum (aa + bb + 2ab = groomed EEC), draws
//   truth:     sig-sig, sig-bkg, bkg-bkg (split of the combined EEC by track origin), all comb.
//   estimate:  jet-perp - perp1-perp2      (compare to sig-bkg)
//              perp-perp                   (compare to bkg-bkg)
//              all - (jet-perp - perp1-perp2) - perp-perp   (compare to sig-sig)
// Example Usage:
// root -l 'plot/plot_eec_perpcone.C("2013681", "40", "60", "OO")'

using namespace std;

void SetStyle(Bool_t graypalette = true)
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(1);
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
    // l->AddEntry(hist, text.c_str(), "pl");

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
    l->AddEntry("NULL", "pp jets + OO 0#minus100% " + plot_text, "h");
    // l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV, #hat{#it{p}}_{T} > 90 GeV", "h");
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

void plot_eec_perpcone(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "")
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string infile = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    // std::string infile  = "/home/youqi/alian/alian/output/" + file_name + "embed.root";
    std::string jetR    = "04";
    std::string jetR_str = "0.4";
    const std::string sample = "combined_matched";

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << std::endl;
    std::cout << "R = " << jetR_str << std::endl;

    double pt_lo = stof(pt_min);
    double pt_hi = stof(pt_max);
    TFile *f = new TFile(infile.c_str(), "READ");
    if (f->IsZombie())
    {
        std::cerr << "Cannot open " << infile << std::endl;
        delete f;
        return;
    }

    // Same jet-count normalization as plot_eec.C for combined_matched.
    TH1 *h_JetPt = (TH1 *)f->Get("combined_pTg_matched");
    if (!h_JetPt)
    {
        std::cerr << "Missing jet pT histogram combined_pTg_matched" << std::endl;
        f->Close();
        delete f;
        return;
    }
    double njets = h_JetPt->Integral(h_JetPt->FindBin(pt_lo), h_JetPt->FindBin(pt_hi));
    std::cout << sample << ": number of jets = " << njets << std::endl;
    if (njets <= 0)
    {
        f->Close();
        delete f;
        return;
    }

    // "all" is the combined EEC itself; the others are suffixes of it.
    const std::vector<std::string> terms = {"all", "sigsig", "sigbkg", "bkgbkg", "jetperp", "perpperp", "perp1perp2"};
    const std::vector<std::string> sets = {"aa", "bb", "ab", "sum", "sdjet"};
    std::map<std::string, std::map<std::string, TH1D *>> h; // h[set][term]

    for (const std::string &xy : {std::string("aa"), std::string("bb"), std::string("ab")})
    {
        double scale = (xy == "ab") ? 2.0 : 1.0; // eec_ab is filled in one orientation
        for (const std::string &term : terms)
        {
            std::string hname = sample + "_eec_" + xy + (term == "all" ? "" : "_" + term);
            h[xy][term] = LoadAndProcess(f, hname.c_str(), pt_lo, pt_hi, njets, scale);
        }
    }
    // Groomed EEC: aa + bb + 2ab (ab already scaled by 2 above).
    for (const std::string &term : terms)
    {
        if (!h["aa"][term] || !h["bb"][term] || !h["ab"][term])
        {
            h["sum"][term] = nullptr;
            continue;
        }
        h["sum"][term] = (TH1D *)h["aa"][term]->Clone(("h_" + sample + "_eec_sum" + (term == "all" ? "" : "_" + term)).c_str());
        h["sum"][term]->Add(h["bb"][term]);
        h["sum"][term]->Add(h["ab"][term]);
    }
    // Inclusive EEC for jets that passed SD
    for (const std::string &term : terms)
    {
        std::string hname = sample + "_sdjet_eec" + (term == "all" ? "" : "_" + term);
        h["sdjet"][term] = LoadAndProcess(f, hname.c_str(), pt_lo, pt_hi, njets);
    }

    std::string outfile = "/home/youqi/alian/alian/output/eec_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    for (const std::string &set : sets)
    {
        std::map<std::string, TH1D *> &t = h[set];
        bool missing = false;
        for (const std::string &term : terms)
            if (!t[term]) missing = true;
        if (missing)
        {
            std::cerr << "WARNING: missing histograms for " << set << ", skipping." << std::endl;
            continue;
        }

        // Perp-cone estimates, as in pyjetty's plot_eec_perpcone.C.
        TH1D *h_cross = (TH1D *)t["jetperp"]->Clone(("h_eec_perpcone_" + set + "_cross").c_str());
        h_cross->Add(t["perp1perp2"], -1); // jet-perp - perp1-perp2
        TH1D *h_bkgbkg_est = (TH1D *)t["perpperp"]->Clone(("h_eec_perpcone_" + set + "_perpperp").c_str());
        TH1D *h_sub = (TH1D *)t["all"]->Clone(("h_eec_perpcone_" + set + "_sub").c_str());
        h_sub->Add(h_cross, -1);
        h_sub->Add(h_bkgbkg_est, -1); // all - (jet-perp - perp1-perp2) - perp-perp

        struct HistConfig
        {
            const char *label;
            int color;
            int marker;
            TH1D *hist;
        };
        std::vector<HistConfig> configs =
        {
            {"sig-sig", kBlue, kFullCircle, t["sigsig"]},
            {"sig-bkg", kGreen+2, 33, t["sigbkg"]},
            {"bkg-bkg", kRed, 21, t["bkgbkg"]},
            // {"all comb.", kBlack, 29, t["all"]},
            {"all comb. #minus [(jet-perp #minus perp1-perp2) + perp-perp]", kBlue+1, 24, h_sub},
            {"jet-perp #minus perp1-perp2", kGreen+3, 24, h_cross},
            {"perp-perp", kRed+1, 24, h_bkgbkg_est},
        };

        TCanvas *c = new TCanvas("c", "c", 800, 600);
        ProcessCanvas(c);
        c->cd();
        gPad->SetLogx();

        TLegend *l = new TLegend(0.2, 0.65, 0.6, 0.88);
        std::string ptbin = pt_min + " < #it{p}_{T,g}^{sub} < " + pt_max + " GeV/#it{c}";
        addLegendInfo(l, jetR_str, plot_text);
        // l->AddEntry("NULL", (sample + ", EEC " + set).c_str(), "h");
        l->AddEntry("NULL", ptbin.c_str(), "h");

        bool first = true;
        for (auto &cfg : configs)
        {
            TH1D *h1 = cfg.hist;
            FormatHist(l, h1, cfg.label, cfg.color, cfg.marker);
            h1->GetXaxis()->SetRangeUser(0.01, 0.4);
            h1->GetYaxis()->SetRangeUser(0, 6);
            h1->GetXaxis()->SetTitle("#it{R}_{L}");
            h1->GetYaxis()->SetTitle("#Sigma_{EEC}(#it{R}_{L})");
            h1->SetTitle(("EEC " + set).c_str());
            if (first)
            {
                h1->Draw("L");
                first = false;
            }
            else
            {
                h1->Draw("L same");
            }
        }
        l->Draw("same");

        std::string fname = "/home/youqi/alian/alian/output/eec_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + "_" + set + ".pdf";
        c->SaveAs(fname.c_str());
        delete c;
        delete l;

        f_out->cd();
        for (const std::string &term : terms)
            t[term]->Write();
        h_cross->Write();
        h_bkgbkg_est->Write();
        h_sub->Write();
    }
    f_out->Close();
    delete f_out;
    f->Close();
    delete f;
}
