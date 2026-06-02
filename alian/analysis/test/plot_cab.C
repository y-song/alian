#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>

using namespace std;

// ── hardcode file pairs here ────────────────────────────────────────────
struct FilePair {
    std::string pbpb_file;
    std::string pp_file;
    std::string label;
    int         color;
    int         style;  // line style
};

const std::vector<FilePair> kPairs = {
    {"output/eec_R04_100_120_1652502.root",
     "output/eec_R04_100_120_1652402.root",
     "z_{cut} = 0.1", kRed,   1},
    {"output/eec_R04_100_120_1652717.root",
     "output/eec_R04_100_120_1652617.root",
     "z_{cut} = 0.2", kBlue,  1},
    {"output/eec_R04_100_120_1652917.root",
     "output/eec_R04_100_120_1652817.root",
     "z_{cut} = 0.3", kGreen+2, 1},
    {"output/eec_R04_100_120_1656144.root",
     "output/eec_R04_100_120_1656044.root",
     "z_{cut} = 0.45", kOrange, 1},
};
// ─────────────────────────────────────────────────────────────────────────────

void SetStyle(Bool_t graypalette = true)
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    if (graypalette) gStyle->SetPalette(8, 0);
    else             gStyle->SetPalette(1);
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
    gStyle->SetLineWidth(2);
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

TH1D *GetCab(TFile *f)
{
    TH1D *h_aa = (TH1D *)f->Get("h_eec_aa_clone");
    TH1D *h_bb = (TH1D *)f->Get("h_eec_bb_clone");
    TH1D *h_ab = (TH1D *)f->Get("h_eec_ab_clone");
    if (!h_aa || !h_bb || !h_ab) {
        std::cerr << "ERROR: missing histogram in " << f->GetName() << std::endl;
        return nullptr;
    }

    TH1D *h_denom = (TH1D *)h_aa->Clone("h_denom_tmp");
    h_denom->SetDirectory(0);
    h_denom->Multiply(h_bb);
    for (int i = 1; i <= h_denom->GetNbinsX(); i++) {
        double val = h_denom->GetBinContent(i);
        double err = h_denom->GetBinError(i);
        double sqrtval = (val > 0) ? sqrt(val) : 0.;
        double sqrterr = (val > 0) ? err / (2. * sqrtval) : 0.;
        h_denom->SetBinContent(i, sqrtval);
        h_denom->SetBinError(i, sqrterr);
    }

    TH1D *h_cab = (TH1D *)h_ab->Clone("h_cab_tmp");
    h_cab->SetDirectory(0);
    h_cab->Divide(h_denom);
    delete h_denom;
    return h_cab;
}

TH1D *MakeRatio(TH1D *num, TH1D *den, const char *name)
{
    if (!num || !den) return nullptr;
    TH1D *ratio = (TH1D *)num->Clone(name);
    ratio->SetDirectory(0);
    ratio->Divide(den);
    return ratio;
}

void plot_cab()
{
    gStyle->SetOptStat(0);
    SetStyle();

    TCanvas *c = new TCanvas("c", "c", 800, 600);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogx();
    gPad->SetLogy(0);

    TLegend *l = new TLegend(0.6, 0.65, 0.9, 0.88);
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);

    bool first = true;
    std::vector<TH1D *> ratios; // keep alive until SaveAs

    for (size_t ip = 0; ip < kPairs.size(); ip++) {
        const FilePair &p = kPairs[ip];

        TFile *f_pbpb = new TFile(p.pbpb_file.c_str(), "READ");
        TFile *f_pp   = new TFile(p.pp_file.c_str(),   "READ");
        if (!f_pbpb || f_pbpb->IsZombie()) {
            std::cerr << "ERROR: cannot open " << p.pbpb_file << std::endl; continue;
        }
        if (!f_pp || f_pp->IsZombie()) {
            std::cerr << "ERROR: cannot open " << p.pp_file   << std::endl; continue;
        }

        TH1D *h_pbpb = GetCab(f_pbpb);
        TH1D *h_pp   = GetCab(f_pp);
        if (!h_pbpb || !h_pp) continue;

        std::string rname = "ratio_" + std::to_string(ip);
        TH1D *ratio = MakeRatio(h_pbpb, h_pp, rname.c_str());
        if (!ratio) continue;
        ratio->SetDirectory(0);

        ratio->SetLineColor(p.color);
        ratio->SetMarkerColor(p.color);
        ratio->SetLineStyle(p.style);
        ratio->SetLineWidth(2);
        ratio->SetMarkerStyle(20 + ip);
        ratio->SetMarkerSize(0.5);

        ratio->GetXaxis()->SetRangeUser(0.005, 0.4);
        ratio->GetYaxis()->SetRangeUser(0.5, 2.5);
        ratio->GetXaxis()->SetTitle("#it{R}_{L}");
        ratio->GetYaxis()->SetTitle("C_{AB}");

        ratio->Draw(first ? "E" : "E same");
        l->AddEntry(ratio, p.label.c_str(), "pl");
        ratios.push_back(ratio);

        f_pbpb->Close(); delete f_pbpb;
        f_pp->Close();   delete f_pp;
        first = false;
    }

    // Unity line
    TLine *unity = new TLine(0.005, 1.0, 0.4, 1.0);
    unity->SetLineColor(kGray+1);
    unity->SetLineStyle(2);
    unity->SetLineWidth(2);
    unity->Draw("same");

    l->Draw("same");

    c->SaveAs("output/cab_overlay.pdf");
    delete c;
    delete l;
    delete unity;
}