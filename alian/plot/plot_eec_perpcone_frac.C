#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>
// Background fractions of the matched combined subjet EECs, read from the output
// ROOT file written by plot/plot_eec_perpcone.C (run that macro first with the
// same arguments). For each of aa, bb, ab, sum (aa + bb + 2ab) and sdjet, draws
//   sig-bkg / sig-sig
//   bkg-bkg / sig-sig
// using the truth split of the combined EEC by track origin.
// Example Usage:
// root -l 'plot/plot_eec_perpcone_frac.C("2013681", "40", "60", "OO")'

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
    l->AddEntry("NULL", "Anti-#it{k}_{T} full jets, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", "#it{p}_{T}^{track} > 1 GeV/#it{c}", "h");
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
}

// Name of a truth term as written by plot_eec_perpcone.C. Histograms that went
// through DivideByBinWidth there carry a "_clone" suffix (aa, bb, ab, sdjet);
// the sum is built by Clone and does not, so try both.
TH1D *LoadTerm(TFile *f, const std::string &sample, const std::string &set, const std::string &term)
{
    std::string base = (set == "sdjet") ? "h_" + sample + "_sdjet_eec_" + term
                                        : "h_" + sample + "_eec_" + set + "_" + term;
    TH1D *h = (TH1D *)f->Get(base.c_str());
    if (!h)
        h = (TH1D *)f->Get((base + "_clone").c_str());
    if (!h)
        std::cerr << "WARNING: histogram \"" << base << "\" (or \"" << base << "_clone\") not found." << std::endl;
    return h;
}

void plot_eec_perpcone_frac(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "")
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string jetR = "04";
    std::string jetR_str = "0.4";
    const std::string sample = "combined_matched";
    const double x_lo = 0.01;
    const double x_hi = 0.4;

    // Output of plot_eec_perpcone.C with the same arguments.
    std::string infile = "/home/youqi/alian/alian/output/eec_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + ".root";
    TFile *f = new TFile(infile.c_str(), "READ");
    if (f->IsZombie())
    {
        std::cerr << "Cannot open " << infile << ". Run plot_eec_perpcone.C first." << std::endl;
        delete f;
        return;
    }

    const std::vector<std::string> sets = {"aa", "bb", "ab", "sum", "sdjet"};

    std::string outfile = "/home/youqi/alian/alian/output/eec_perpcone_frac_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    for (const std::string &set : sets)
    {
        TH1D *h_sigsig = LoadTerm(f, sample, set, "sigsig");
        TH1D *h_sigbkg = LoadTerm(f, sample, set, "sigbkg");
        TH1D *h_bkgbkg = LoadTerm(f, sample, set, "bkgbkg");
        if (!h_sigsig || !h_sigbkg || !h_bkgbkg)
        {
            std::cerr << "WARNING: missing histograms for " << set << ", skipping." << std::endl;
            continue;
        }

        TH1D *h_frac_sigbkg = (TH1D *)h_sigbkg->Clone(("h_eec_frac_" + set + "_sigbkg_over_sigsig").c_str());
        h_frac_sigbkg->Divide(h_sigsig);
        TH1D *h_frac_bkgbkg = (TH1D *)h_bkgbkg->Clone(("h_eec_frac_" + set + "_bkgbkg_over_sigsig").c_str());
        h_frac_bkgbkg->Divide(h_sigsig);

        struct HistConfig
        {
            const char *label;
            int color;
            int marker;
            TH1D *hist;
        };
        std::vector<HistConfig> configs =
        {
            {"sig-bkg / sig-sig", kGreen+2, 33, h_frac_sigbkg},
            {"bkg-bkg / sig-sig", kRed, 21, h_frac_bkgbkg},
        };

        // y range from the largest fraction inside the drawn R_L window.
        double ymax = 0;
        for (auto &cfg : configs)
        {
            cfg.hist->GetXaxis()->SetRangeUser(x_lo, x_hi);
            ymax = std::max(ymax, cfg.hist->GetMaximum());
        }
        if (ymax <= 0)
            ymax = 1;

        TCanvas *c = new TCanvas("c", "c", 800, 600);
        ProcessCanvas(c);
        c->cd();
        gPad->SetLogx();

        TLegend *l = new TLegend(0.2, 0.6, 0.6, 0.88);
        std::string ptbin = pt_min + " < #it{p}_{T,g}^{sub} < " + pt_max + " GeV/#it{c}";
        addLegendInfo(l, jetR_str, plot_text);
        l->AddEntry("NULL", ptbin.c_str(), "h");

        bool first = true;
        for (auto &cfg : configs)
        {
            TH1D *h1 = cfg.hist;
            FormatHist(l, h1, cfg.label, cfg.color, cfg.marker);
            l->AddEntry(h1, cfg.label, "pl");
            h1->GetXaxis()->SetRangeUser(0.01, 0.4);
            h1->GetYaxis()->SetRangeUser(0, 1);
            h1->GetXaxis()->SetTitle("#it{R}_{L}");
            h1->GetYaxis()->SetTitle("Ratio to sig-sig");
            h1->SetTitle(("EEC " + set).c_str());
            if (first)
            {
                h1->Draw("PE");
                first = false;
            }
            else
            {
                h1->Draw("PE same");
            }
        }
        TLine *line = nullptr;
        if (1.6 * ymax > 1)
        {
            line = drawHoriLine(x_lo, x_hi, 1, kBlack);
            line->Draw("same");
        }
        l->Draw("same");

        std::string fname = "/home/youqi/alian/alian/output/eec_perpcone_frac_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + "_" + set + ".pdf";
        c->SaveAs(fname.c_str());
        delete c;
        delete l;
        delete line;

        f_out->cd();
        h_frac_sigbkg->Write();
        h_frac_bkgbkg->Write();
    }
    f_out->Close();
    delete f_out;
    f->Close();
    delete f;
}
