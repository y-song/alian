#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>
// Ghost-based A/B assignment diagnostics for the perp-cone subjet EECs filled by
// analysis/test/embed_perpcone.py (combined_matched only). Draws projections of
//   perp_ghost_dR_{a,b,none}   dR of perp-cone particle to its closest ghost, in the pT_g range
//                              (unit-normalized shapes)
//   perp_assign                fraction of perp-cone particles assigned to A, B, neither vs pT_g
//   perp_assign_rg             same vs R_g (all pT_g)
//   ghost_closure[_bkg]        fraction of the jet's own A/B tracks (all / OO-origin) that the
//                              ghosts assign correctly, swapped or dropped vs R_g (all pT_g)
// Compare outputs for decreasing oo.jet_finder.ghost_area to check convergence.
// Example Usage:
// root -l 'plot/plot_ghost_assign.C("2013681", "40", "60", "OO")'
// Optional 5th argument rebins the pT_g axis of the fraction-vs-pT_g plot (2 GeV bins by default).

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
    hist->SetMarkerSize(0.8);
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

// Y-projection of h2 over the x-bins covering [x_lo, x_hi], normalized to unit area.
// Returns nullptr if the histogram is missing or empty in the range.
TH1D *ProjectYShape(TH2 *h2, double x_lo, double x_hi, const char *name)
{
    int ilo = h2->GetXaxis()->FindBin(x_lo + 1e-6);
    int ihi = h2->GetXaxis()->FindBin(x_hi - 1e-6);
    TH1D *proj = h2->ProjectionY(name, ilo, ihi, "e");
    if (proj->Integral() <= 0)
    {
        std::cerr << "WARNING: " << h2->GetName() << " is empty in [" << x_lo << ", " << x_hi << "]." << std::endl;
        return nullptr;
    }
    proj->Scale(1. / proj->Integral());
    return proj;
}

// Fraction of entries in y-bin ybin of h2, per x-bin (after rebinning x by rebin), with binomial errors.
TH1D *CategoryFraction(TH2 *h2, int ybin, const char *name, int rebin = 1)
{
    TH1D *num = h2->ProjectionX(name, ybin, ybin, "e");
    TH1D *den = h2->ProjectionX(Form("%s_den", name), 1, h2->GetNbinsY(), "e");
    if (rebin > 1)
    {
        num->Rebin(rebin);
        den->Rebin(rebin);
    }
    num->Divide(num, den, 1., 1., "B");
    delete den;
    return num;
}

// Fraction of all entries of h2 in [x_lo, x_hi] that fall in y-bin ybin.
double IntegratedFraction(TH2 *h2, int ybin, double x_lo, double x_hi)
{
    int ilo = h2->GetXaxis()->FindBin(x_lo + 1e-6);
    int ihi = h2->GetXaxis()->FindBin(x_hi - 1e-6);
    double total = h2->Integral(ilo, ihi, 1, h2->GetNbinsY());
    return total > 0 ? h2->Integral(ilo, ihi, ybin, ybin) / total : 0.;
}

TH2 *GetHist2D(TFile *f, const std::string &name)
{
    TH2 *h2 = (TH2 *)f->Get(name.c_str());
    if (!h2)
        std::cerr << "WARNING: histogram \"" << name << "\" not found in file, skipping." << std::endl;
    return h2;
}

struct HistConfig
{
    const char *label;
    int color;
    int marker;
    TH1D *hist;
};

// Draw configs on a new canvas with the standard legend and save as pdf.
void DrawAndSave(std::vector<HistConfig> &configs, const std::string &title, const std::string &xtitle,
                 const std::string &ytitle, double ymin, double ymax, bool logx, bool logy,
                 const std::string &jetR_str, const std::string &plot_text, const std::string &extra_text,
                 const std::string &fname)
{
    TCanvas *c = new TCanvas("c", "c", 800, 600);
    ProcessCanvas(c);
    c->cd();
    if (logx)
        gPad->SetLogx();
    if (logy)
        gPad->SetLogy();

    TLegend *l = new TLegend(0.2, 0.55, 0.6, 0.88);
    addLegendInfo(l, jetR_str, plot_text);
    if (!extra_text.empty())
        l->AddEntry("NULL", extra_text.c_str(), "h");

    bool first = true;
    for (auto &cfg : configs)
    {
        if (!cfg.hist)
            continue;
        TH1D *h1 = cfg.hist;
        FormatHist(l, h1, cfg.label, cfg.color, cfg.marker);
        h1->GetYaxis()->SetRangeUser(ymin, ymax);
        h1->GetXaxis()->SetTitle(xtitle.c_str());
        h1->GetYaxis()->SetTitle(ytitle.c_str());
        h1->SetTitle(title.c_str());
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
    if (!first)
    {
        l->Draw("same");
        c->SaveAs(fname.c_str());
    }
    delete c;
    delete l;
}

void plot_ghost_assign(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "", int pt_rebin = 1)
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string infile = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    // std::string infile  = "/home/youqi/alian/alian/output/" + file_name + "embed.root";
    std::string jetR = "04";
    std::string jetR_str = "0.4";
    const std::string sample = "combined_matched";
    const std::string outdir = "/home/youqi/alian/alian/output/";
    const std::string tag = "R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name;

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

    std::string ptbin = pt_min + " < #it{p}_{T,g}^{sub} < " + pt_max + " GeV/#it{c}";
    std::string outfile = outdir + "ghost_assign_" + tag + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    // Category bins of the 'assign' axis (bin = category + 1), as in embed_perpcone.py:
    // ASSIGN_A, ASSIGN_B, ASSIGN_NONE and CLOSURE_CORRECT, CLOSURE_SWAPPED, CLOSURE_DROPPED.
    const int kColors[3] = {kBlue, kGreen + 2, kRed};
    const int kMarkers[3] = {kFullCircle, 33, 21};
    const int kOpenMarkers[3] = {24, 27, 25};
    const char *assign_labels[3] = {"A", "B", "neither"};
    const char *closure_labels[3] = {"correct", "swapped", "dropped"};

    // 1) dR of perp-cone particle to its closest ghost, in the pT_g range.
    {
        const char *suffixes[3] = {"a", "b", "none"};
        std::vector<HistConfig> configs;
        for (int i = 0; i < 3; i++)
        {
            TH2 *h2 = GetHist2D(f, sample + "_perp_ghost_dR_" + suffixes[i]);
            TH1D *h1 = h2 ? ProjectYShape(h2, pt_lo, pt_hi, Form("h_%s_perp_ghost_dR_%s_%s_%s", sample.c_str(), suffixes[i], pt_min.c_str(), pt_max.c_str())) : nullptr;
            std::string label = std::string(assign_labels[i]);
            if (h1)
            {
                label += Form(", #LT#Delta#it{R}#GT = %.3f", h1->GetMean());
                std::cout << "perp_ghost_dR_" << suffixes[i] << ": mean = " << h1->GetMean() << ", RMS = " << h1->GetRMS() << std::endl;
                f_out->cd();
                h1->Write();
            }
            configs.push_back({strdup(label.c_str()), kColors[i], kMarkers[i], h1});
        }
        DrawAndSave(configs, "Perp-cone particle to closest ghost", "#Delta#it{R}(particle, closest ghost)",
                    "normalized counts", 1e-4, 1., false, true, jetR_str, plot_text, ptbin,
                    outdir + "ghost_dR_" + tag + ".pdf");
    }

    // 2) Perp-cone assignment fractions vs pT_g; integrated fractions in the pT_g range printed.
    TH2 *h2_assign = GetHist2D(f, sample + "_perp_assign");
    if (h2_assign)
    {
        std::vector<HistConfig> configs;
        for (int i = 0; i < 3; i++)
        {
            TH1D *h1 = CategoryFraction(h2_assign, i + 1, Form("h_%s_perp_assign_frac_%s", sample.c_str(), assign_labels[i]), pt_rebin);
            h1->GetXaxis()->SetRangeUser(10., 200.);
            std::cout << "perp_assign " << assign_labels[i] << " fraction in pT_g range: " << IntegratedFraction(h2_assign, i + 1, pt_lo, pt_hi) << std::endl;
            f_out->cd();
            h1->Write();
            configs.push_back({assign_labels[i], kColors[i], kMarkers[i], h1});
        }
        DrawAndSave(configs, "Perp-cone particle assignment", "#it{p}_{T,g}^{sub} (GeV/#it{c})",
                    "fraction of perp-cone particles", 0., 1.2, false, false, jetR_str, plot_text, "",
                    outdir + "perp_assign_" + tag + ".pdf");
    }

    // 3) Perp-cone assignment fractions vs R_g (all pT_g).
    TH2 *h2_assign_rg = GetHist2D(f, sample + "_perp_assign_rg");
    if (h2_assign_rg)
    {
        std::vector<HistConfig> configs;
        for (int i = 0; i < 3; i++)
        {
            TH1D *h1 = CategoryFraction(h2_assign_rg, i + 1, Form("h_%s_perp_assign_rg_frac_%s", sample.c_str(), assign_labels[i]));
            h1->GetXaxis()->SetRangeUser(0.01, 0.4);
            f_out->cd();
            h1->Write();
            configs.push_back({assign_labels[i], kColors[i], kMarkers[i], h1});
        }
        DrawAndSave(configs, "Perp-cone particle assignment", "#it{R}_{g}",
                    "fraction of perp-cone particles", 0., 1.2, true, false, jetR_str, plot_text, "all #it{p}_{T,g}^{sub}",
                    outdir + "perp_assign_rg_" + tag + ".pdf");
    }

    // 4) Ghost-assignment closure of the jet's own A/B tracks vs R_g (all pT_g):
    //    all tracks (filled markers) and OO-origin tracks (open markers).
    TH2 *h2_closure = GetHist2D(f, sample + "_ghost_closure");
    TH2 *h2_closure_bkg = GetHist2D(f, sample + "_ghost_closure_bkg");
    if (h2_closure || h2_closure_bkg)
    {
        std::vector<HistConfig> configs;
        for (int i = 0; i < 3; i++)
        {
            if (h2_closure)
            {
                TH1D *h1 = CategoryFraction(h2_closure, i + 1, Form("h_%s_ghost_closure_frac_%s", sample.c_str(), closure_labels[i]));
                h1->GetXaxis()->SetRangeUser(0.01, 0.4);
                f_out->cd();
                h1->Write();
                configs.push_back({strdup(Form("%s, all tracks", closure_labels[i])), kColors[i], kMarkers[i], h1});
            }
            if (h2_closure_bkg)
            {
                TH1D *h1 = CategoryFraction(h2_closure_bkg, i + 1, Form("h_%s_ghost_closure_bkg_frac_%s", sample.c_str(), closure_labels[i]));
                h1->GetXaxis()->SetRangeUser(0.01, 0.4);
                f_out->cd();
                h1->Write();
                configs.push_back({strdup(Form("%s, OO tracks", closure_labels[i])), kColors[i], kOpenMarkers[i], h1});
            }
        }
        DrawAndSave(configs, "Ghost assignment of jet A/B tracks", "#it{R}_{g}",
                    "fraction of subjet tracks", 0., 1.4, true, false, jetR_str, plot_text, "all #it{p}_{T,g}^{sub}",
                    outdir + "ghost_closure_" + tag + ".pdf");
    }

    f_out->Close();
    delete f_out;
    f->Close();
    delete f;
}
