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
    gStyle->SetLabelSize(0.02, "xyz");
    gStyle->SetLabelOffset(0.01, "y");
    gStyle->SetLabelOffset(0.01, "x");
    gStyle->SetLabelColor(kBlack, "xyz");
    gStyle->SetTitleSize(0.025, "xyz");
    gStyle->SetTitleOffset(1.5, "y");
    gStyle->SetTitleOffset(1.2, "x");
    gStyle->SetTitleFillColor(kWhite);
    gStyle->SetTextSizePixels(26);
    gStyle->SetTextFont(42);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    gStyle->SetLegendFont(42);
}

void FormatHist(TLegend *l, TH1 *hist, TString text, int color)
{
    hist->SetMarkerSize(0.8);
    hist->SetMarkerStyle(8);
    hist->SetMarkerColor(color);
    hist->SetLineColor(color);
    hist->GetYaxis()->SetTitle("Probability density");
    hist->GetYaxis()->SetTitleOffset(1.5);
    hist->GetYaxis()->SetTitleSize(0.035);
    hist->GetYaxis()->SetLabelSize(0.035);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.1);
    hist->GetXaxis()->SetTitleSize(0.035);
    hist->GetXaxis()->SetLabelSize(0.035);

    l->AddEntry(hist, text, "pl");

    return;
}

// second legend row under a histogram entry with its mean and standard deviation
void addStatsEntry(TLegend *l, double mean, double sigma)
{
    l->AddEntry("NULL", TString::Format("#LT#deltap_{T}#GT = %.2f GeV, #sigma(#deltap_{T}) = %.2f", mean, sigma), "h");
}

void addLegendInfo(TLegend *l, string pt_min, string pt_max, string jetR)
{
    l->SetTextSize(0.032);
    l->AddEntry("NULL", "pp jets + OO 0#minus100%", "h");
    l->AddEntry("NULL", ("charged jets, anti-#it{k}_{T}, #it{R} =" + jetR).c_str(), "h");
    l->AddEntry("NULL", "", "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void check_bump_fraction(vector<TH1D *> hists, vector<string> labels, double xcut = 5.)
{
    // thin dotted vertical lines at +/- xcut on the current pad
    gPad->Update(); // pad range is not computed until the pad is painted
    double ymin = gPad->GetUymin();
    double ymax = gPad->GetUymax();
    if (gPad->GetLogy())
    {
        ymin = pow(10, ymin);
        ymax = pow(10, ymax);
    }
    for (double x : {-xcut, xcut})
    {
        TLine *line = new TLine(x, ymin, x, ymax);
        line->SetLineStyle(3);
        line->SetLineWidth(1);
        line->SetLineColor(kBlack);
        line->Draw("same");
    }

    // fraction of area outside [-xcut, xcut] over the whole range
    for (size_t i = 0; i < hists.size(); i++)
    {
        TH1D *h = hists[i];
        double total = h->Integral(1, h->GetNbinsX());
        if (total <= 0)
            continue;
        int binLo = h->GetXaxis()->FindBin(-xcut + 1e-6);
        int binHi = h->GetXaxis()->FindBin(xcut - 1e-6);
        // binLo > binHi means an empty window (e.g. xcut = 0); ROOT would swap the bins instead of returning 0
        double inside = (binLo <= binHi) ? h->Integral(binLo, binHi) : 0.0;
        cout << labels[i] << ": fraction outside |#deltap_{T}| > " << xcut << " = " << (total - inside) / total << endl;
    }
}

void plot_dpt(string jobID)
{
    SetStyle();

    const string jetR = "04";
    const string jetRPoint = "0.4";

    TFile *f = new TFile(("/rstorage/youqi/" + jobID + "/AnalysisResultsFinal.root").c_str(), "READ");
    // TFile *f = new TFile("/home/youqi/alian/alian/output/embed_111226.root", "READ");
    TH2D *h_residual_pp_pT_matched = (TH2D *)f->Get("residual_pp_pT_matched");
    TH2D *h_residual_sub_pT_matched = (TH2D *)f->Get("residual_sub_pT_matched");

    TH2D *h1 = (TH2D *)h_residual_pp_pT_matched->Clone("h1");
    TH2D *h2 = (TH2D *)h_residual_pp_pT_matched->Clone("h2");
    TH2D *h3 = (TH2D *)h_residual_pp_pT_matched->Clone("h3");
    TH2D *h4 = (TH2D *)h_residual_sub_pT_matched->Clone("h4");
    TH2D *h5 = (TH2D *)h_residual_sub_pT_matched->Clone("h5");
    TH2D *h6 = (TH2D *)h_residual_sub_pT_matched->Clone("h6");
    TH2D *h7 = (TH2D *)h_residual_pp_pT_matched->Clone("h7");
    TH2D *h8 = (TH2D *)h_residual_sub_pT_matched->Clone("h8");

    h1->GetXaxis()->SetRangeUser(20.0, 40.0);
    h2->GetXaxis()->SetRangeUser(40.0, 60.0);
    h3->GetXaxis()->SetRangeUser(60.0, 80.0);
    h4->GetXaxis()->SetRangeUser(20.0, 40.0);
    h5->GetXaxis()->SetRangeUser(40.0, 60.0);
    h6->GetXaxis()->SetRangeUser(60.0, 80.0);
    h7->GetXaxis()->SetRangeUser(80.0, 100.0);
    h8->GetXaxis()->SetRangeUser(80.0, 100.0);

    TH1D *h1_proj = h1->ProjectionY();
    TH1D *h2_proj = h2->ProjectionY();
    TH1D *h3_proj = h3->ProjectionY();
    TH1D *h4_proj = h4->ProjectionY();
    TH1D *h5_proj = h5->ProjectionY();
    TH1D *h6_proj = h6->ProjectionY();
    TH1D *h7_proj = h7->ProjectionY();
    TH1D *h8_proj = h8->ProjectionY();

    cout << "mean: " << h1_proj->GetMean() << ", " << h2_proj->GetMean() << ", " << h3_proj->GetMean() << ", " << h7_proj->GetMean() << endl;
    cout << "sigma: " << h1_proj->GetStdDev() << ", " << h2_proj->GetStdDev() << ", " << h3_proj->GetStdDev() << ", " << h7_proj->GetStdDev() << endl;

    cout << "mean: " << h4_proj->GetMean() << ", " << h5_proj->GetMean() << ", " << h6_proj->GetMean() << ", " << h8_proj->GetMean() << endl;
    cout << "sigma: " << h4_proj->GetStdDev() << ", " << h5_proj->GetStdDev() << ", " << h6_proj->GetStdDev() << ", " << h8_proj->GetStdDev() << endl;

    // store the full-range values printed above; SetRangeUser(-40, 40) below would change GetMean/GetStdDev
    vector<TH1D *> projs = {h1_proj, h2_proj, h3_proj, h4_proj, h5_proj, h6_proj, h7_proj, h8_proj};
    vector<double> means;
    vector<double> sigmas;
    for (TH1D *h : projs)
    {
        means.push_back(h->GetMean());
        sigmas.push_back(h->GetStdDev());
    }

    h1_proj->Scale(1.0 / h1_proj->Integral());
    h2_proj->Scale(1.0 / h2_proj->Integral());
    h3_proj->Scale(1.0 / h3_proj->Integral());
    h4_proj->Scale(1.0 / h4_proj->Integral());
    h5_proj->Scale(1.0 / h5_proj->Integral());
    h6_proj->Scale(1.0 / h6_proj->Integral());
    h7_proj->Scale(1.0 / h7_proj->Integral());
    h8_proj->Scale(1.0 / h8_proj->Integral());

    // First canvas
    TCanvas *c1 = new TCanvas();
    c1->SetCanvasSize(700, 500);
    c1->cd();
    gPad->SetLogy();

    TLegend *leg1 = new TLegend(0.16, 0.40, 0.4662155, 0.88, "");
    addLegendInfo(leg1, "", "", jetRPoint);
    h1_proj->GetXaxis()->SetTitle("#deltap_{T} = p_{T}^{combined sub} #minus p_{T}^{pp} [GeV]");
    h1_proj->GetXaxis()->SetRangeUser(-40, 40);
    h1_proj->GetYaxis()->SetRangeUser(0.0001, 1.0);
    FormatHist(leg1, h1_proj, "20 < #it{p}_{T}^{pp} < 40 GeV", kGreen+2);
    addStatsEntry(leg1, means[0], sigmas[0]);
    FormatHist(leg1, h2_proj, "40 < #it{p}_{T}^{pp} < 60 GeV", kRed+2);
    addStatsEntry(leg1, means[1], sigmas[1]);
    FormatHist(leg1, h3_proj, "60 < #it{p}_{T}^{pp} < 80 GeV", kBlue+2);
    addStatsEntry(leg1, means[2], sigmas[2]);
    FormatHist(leg1, h7_proj, "80 < #it{p}_{T}^{pp} < 100 GeV", kMagenta+2);
    addStatsEntry(leg1, means[6], sigmas[6]);

    h1_proj->Draw();
    h2_proj->Draw("same");
    h3_proj->Draw("same");
    h7_proj->Draw("same");
    leg1->Draw("same");
    check_bump_fraction({h1_proj, h2_proj, h3_proj, h7_proj}, {"pp 20-40", "pp 40-60", "pp 60-80", "pp 80-100"});

    c1->SaveAs(("output/dpT_vs_pp_pT_R" + jetR + "_" + jobID + "_matched.pdf").c_str());

    // Second canvas
    TCanvas *c2 = new TCanvas();
    c2->SetCanvasSize(700, 500);
    c2->cd();
    gPad->SetLogy();
    
    TLegend *leg2 = new TLegend(0.16, 0.40, 0.4662155, 0.88, "");
    addLegendInfo(leg2, "", "", jetRPoint);
    h4_proj->GetXaxis()->SetTitle("#deltap_{T} = p_{T}^{combined sub} #minus p_{T}^{pp} [GeV]");
    h4_proj->GetXaxis()->SetRangeUser(-40, 40);
    h4_proj->GetYaxis()->SetRangeUser(0.0001, 1.0);
    FormatHist(leg2, h4_proj, "20 < #it{p}_{T}^{combined sub} < 40 GeV", kGreen+2);
    addStatsEntry(leg2, means[3], sigmas[3]);
    FormatHist(leg2, h5_proj, "40 < #it{p}_{T}^{combined sub} < 60 GeV", kRed+2);
    addStatsEntry(leg2, means[4], sigmas[4]);
    FormatHist(leg2, h6_proj, "60 < #it{p}_{T}^{combined sub} < 80 GeV", kBlue+2);
    addStatsEntry(leg2, means[5], sigmas[5]);
    FormatHist(leg2, h8_proj, "80 < #it{p}_{T}^{combined sub} < 100 GeV", kMagenta+2);
    addStatsEntry(leg2, means[7], sigmas[7]);

    h4_proj->Draw();
    h5_proj->Draw("same");
    h6_proj->Draw("same");
    h8_proj->Draw("same");
    leg2->Draw("same");
    check_bump_fraction({h4_proj, h5_proj, h6_proj, h8_proj}, {"sub 20-40", "sub 40-60", "sub 60-80", "sub 80-100"});

    // c2->SaveAs(("output/dpT_vs_sub_pT_R" + jetR + "_matched.pdf").c_str());
    c2->SaveAs(("output/dpT_vs_sub_pT_R" + jetR + "_" + jobID + "_matched.pdf").c_str());
}