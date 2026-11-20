#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>
// Perp-cone closure ratios (estimate / truth) of the matched combined subjet EECs,
// read from the output ROOT file written by plot/plot_eec_perpcone.C (run that
// macro first with the same arguments). In that macro each color pairs an
// estimate (empty marker) with the truth it should reproduce (solid marker).
// For each of aa, bb, ab, sum (aa + bb + 2ab) and sdjet, draws
//   blue:  [all - (jet-perp - perp1-perp2) - perp-perp] / sig-sig
//   green: (jet-perp - perp1-perp2) / sig-bkg
//   red:   perp-perp / bkg-bkg
// Errors come from TH1::Divide and treat numerator and denominator as
// uncorrelated, although both are filled from the same events.
// perp_suffix must match the one given to plot_eec_perpcone.C (e.g. "_fullpt" for
// the full-pT perp A/B assignment of embed_cs.py); it selects that macro's output
// file and is appended to the output file names. sdjet is skipped when it is set.
// Example Usage:
// root -l 'plot/plot_eec_perpcone_ratio.C("2013681", "40", "60", "OO")'
// root -l 'plot/plot_eec_perpcone_ratio.C("2013681", "40", "60", "OO", "_fullpt")'

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

// Truth term as written by plot_eec_perpcone.C. Histograms that went through
// DivideByBinWidth there carry a "_clone" suffix (aa, bb, ab, sdjet); the sum
// is built by Clone and does not, so try both.
TH1D *LoadTerm(TFile *f, const std::string &sample, const std::string &set, const std::string &term)
{
    std::string base = "h_" + sample + "_eec_" + set + "_" + term;
    if (set == "sdjet")
        base = "h_" + sample + "_sdjet_eec_" + term;
    TH1D *h = (TH1D *)f->Get(base.c_str());
    if (!h)
        h = (TH1D *)f->Get((base + "_clone").c_str());
    if (!h)
        std::cerr << "WARNING: histogram \"" << base << "\" (or \"" << base << "_clone\") not found." << std::endl;
    return h;
}

// Perp-cone estimate as written by plot_eec_perpcone.C (h_eec_perpcone_<set>_<name>).
TH1D *LoadEstimate(TFile *f, const std::string &set, const std::string &name)
{
    std::string hname = "h_eec_perpcone_" + set + "_" + name;
    TH1D *h = (TH1D *)f->Get(hname.c_str());
    if (!h)
        std::cerr << "WARNING: histogram \"" << hname << "\" not found." << std::endl;
    return h;
}

void plot_eec_perpcone_ratio(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "", std::string perp_suffix = "")
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string jetR = "04";
    std::string jetR_str = "0.4";
    const std::string sample = "combined_matched";
    const double x_lo = 0.01;
    const double x_hi = 0.4;

    // Output of plot_eec_perpcone.C with the same arguments.
    std::string infile = "/home/youqi/alian/alian/output/eec_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + perp_suffix + ".root";
    TFile *f = new TFile(infile.c_str(), "READ");
    if (f->IsZombie())
    {
        std::cerr << "Cannot open " << infile << ". Run plot_eec_perpcone.C first." << std::endl;
        delete f;
        return;
    }

    std::vector<std::string> sets = {"aa", "bb", "ab", "sum"};
    if (perp_suffix.empty())
    {
        sets.push_back("sdjet");
    }

    std::string outfile = "/home/youqi/alian/alian/output/eec_perpcone_ratio_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + perp_suffix + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    for (const std::string &set : sets)
    {
        TH1D *h_sigsig = LoadTerm(f, sample, set, "sigsig");
        TH1D *h_sigbkg = LoadTerm(f, sample, set, "sigbkg");
        TH1D *h_bkgbkg = LoadTerm(f, sample, set, "bkgbkg");
        TH1D *h_sub = LoadEstimate(f, set, "sub");
        TH1D *h_cross = LoadEstimate(f, set, "cross");
        TH1D *h_perpperp = LoadEstimate(f, set, "perpperp");
        if (!h_sigsig || !h_sigbkg || !h_bkgbkg || !h_sub || !h_cross || !h_perpperp)
        {
            std::cerr << "WARNING: missing histograms for " << set << ", skipping." << std::endl;
            continue;
        }

        // Empty marker (estimate) over solid marker (truth), per color.
        TH1D *h_ratio_sub = (TH1D *)h_sub->Clone(("h_eec_ratio_" + set + "_sub_over_sigsig").c_str());
        h_ratio_sub->Divide(h_sigsig);
        TH1D *h_ratio_cross = (TH1D *)h_cross->Clone(("h_eec_ratio_" + set + "_cross_over_sigbkg").c_str());
        h_ratio_cross->Divide(h_sigbkg);
        TH1D *h_ratio_perpperp = (TH1D *)h_perpperp->Clone(("h_eec_ratio_" + set + "_perpperp_over_bkgbkg").c_str());
        h_ratio_perpperp->Divide(h_bkgbkg);

        struct HistConfig
        {
            const char *label;
            int color;
            int marker;
            TH1D *hist;
        };
        std::vector<HistConfig> configs =
        {
            {"[all comb. #minus (jet-perp #minus perp1-perp2) #minus perp-perp] / sig-sig", kBlue, kFullCircle, h_ratio_sub},
            {"(jet-perp #minus perp1-perp2) / sig-bkg", kGreen+2, 33, h_ratio_cross},
            {"perp-perp / bkg-bkg", kRed, 21, h_ratio_perpperp},
        };

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
            h1->GetXaxis()->SetRangeUser(x_lo, x_hi);
            h1->GetYaxis()->SetRangeUser(0, 2);
            h1->GetXaxis()->SetTitle("#it{R}_{L}");
            h1->GetYaxis()->SetTitle("Estimate / truth");
            h1->SetTitle(("EEC " + set + (perp_suffix.empty() ? "" : ", perp terms " + perp_suffix)).c_str());
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
        TLine *line = drawHoriLine(x_lo, x_hi, 1, kBlack);
        line->Draw("same");
        l->Draw("same");

        std::string fname = "/home/youqi/alian/alian/output/eec_perpcone_ratio_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name + perp_suffix + "_" + set + ".pdf";
        c->SaveAs(fname.c_str());
        delete c;
        delete l;
        delete line;

        f_out->cd();
        h_ratio_sub->Write();
        h_ratio_cross->Write();
        h_ratio_perpperp->Write();
    }
    f_out->Close();
    delete f_out;
    f->Close();
    delete f;
}
