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

void addLegendInfo(TLegend *l, string pt_min, string pt_max, string jetR)
{
    l->SetTextSize(0.032);
    l->AddEntry("NULL", "pp jets + OO 0#minus10%", "h");
    l->AddEntry("NULL", ("charged jets, anti-#it{k}_{T}, #it{R} =" + jetR).c_str(), "h");
    l->AddEntry("NULL", "", "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void plot_dpt()
{
    SetStyle();

    const string jetR = "04";
    const string jetRPoint = "0.4";
    const string jobID = "1791701";

    TFile *f = new TFile(("/rstorage/youqi/" + jobID + "/AnalysisResultsFinal.root").c_str(), "READ");
    TH2D *h_grid_med = (TH2D *)f->Get("delta_pT_grid_pp_jet_pT");
    TH2D *h_jet_med = (TH2D *)f->Get("delta_pT_jet_pp_jet_pT");

    TH2D *h1 = (TH2D *)h_grid_med->Clone("h1");
    TH2D *h2 = (TH2D *)h_grid_med->Clone("h2");
    TH2D *h3 = (TH2D *)h_grid_med->Clone("h3");
    TH2D *h4 = (TH2D *)h_jet_med->Clone("h4");
    TH2D *h5 = (TH2D *)h_jet_med->Clone("h5");
    TH2D *h6 = (TH2D *)h_jet_med->Clone("h6");

    h1->GetXaxis()->SetRangeUser(20.0, 40.0);
    h2->GetXaxis()->SetRangeUser(40.0, 60.0);
    // h3->GetXaxis()->SetRangeUser(30.0, 40.0);
    h4->GetXaxis()->SetRangeUser(20.0, 40.0);
    h5->GetXaxis()->SetRangeUser(40.0, 60.0);
    // h6->GetXaxis()->SetRangeUser(30.0, 40.0);

    TH1D *h1_proj = h1->ProjectionY();
    TH1D *h2_proj = h2->ProjectionY();
    // TH1D *h3_proj = h3->ProjectionY();
    TH1D *h4_proj = h4->ProjectionY();
    TH1D *h5_proj = h5->ProjectionY();
    // TH1D *h6_proj = h6->ProjectionY();

    // cout << "Mean: " << h7_proj->GetMean() << ", " << h8_proj->GetMean() << ", " << h9_proj->GetMean() << endl;
    // cout << "Sigma: " << h7_proj->GetStdDev() << ", " << h8_proj->GetStdDev() << ", " << h9_proj->GetStdDev() << endl;

    h1_proj->Scale(1.0 / h1_proj->Integral());
    h2_proj->Scale(1.0 / h2_proj->Integral());
    // h3_proj->Scale(1.0 / h3_proj->Integral());
    h4_proj->Scale(1.0 / h4_proj->Integral());
    h5_proj->Scale(1.0 / h5_proj->Integral());
    // h6_proj->Scale(1.0 / h6_proj->Integral());

    // First canvas
    TCanvas *c1 = new TCanvas();
    c1->SetCanvasSize(700, 500);
    c1->cd();
    
    TLegend *leg1 = new TLegend(0.16, 0.56, 0.4662155, 0.88, "");
    addLegendInfo(leg1, "", "", jetRPoint);
    h1_proj->GetXaxis()->SetTitle("#deltap_{T} = p_{T}^{combined sub} #minus p_{T}^{pp} [GeV]");
    h1_proj->GetYaxis()->SetRangeUser(0, 0.05);
    h1_proj->SetTitle("Grid median #rho");
    FormatHist(leg1, h1_proj, "20 < #it{p}_{T}^{pp} < 40 GeV", kGreen+2);
    FormatHist(leg1, h2_proj, "40 < #it{p}_{T}^{pp} < 60 GeV", kRed+2);
    // FormatHist(leg1, h3_proj, "30 < #it{p}_{T}^{pp} < 40 GeV", kBlue+2);

    h1_proj->Draw();
    h2_proj->Draw("same");
    // h3_proj->Draw("same");  
    leg1->Draw("same");

    c1->SaveAs(("output/dpt_grid_med_vs_pp_pt_R" + jetR + "_" + jobID + ".pdf").c_str());

    // Second canvas
    TCanvas *c2 = new TCanvas();
    c2->SetCanvasSize(700, 500);
    c2->cd();
    
    TLegend *leg2 = new TLegend(0.16, 0.56, 0.4662155, 0.88, "");
    addLegendInfo(leg2, "", "", jetRPoint);
    h4_proj->GetXaxis()->SetTitle("#deltap_{T} = p_{T}^{combined sub} #minus p_{T}^{pp} [GeV]");
    h4_proj->GetYaxis()->SetRangeUser(0, 0.05);
    h4_proj->SetTitle("Jet median #rho");
    FormatHist(leg2, h4_proj, "20 < #it{p}_{T}^{pp} < 40 GeV", kGreen+2);
    FormatHist(leg2, h5_proj, "40 < #it{p}_{T}^{pp} < 60 GeV", kRed+2);
    // FormatHist(leg2, h6_proj, "30 < #it{p}_{T}^{pp} < 40 GeV", kBlue+2);

    h4_proj->Draw();
    h5_proj->Draw("same");
    // h6_proj->Draw("same");  
    leg2->Draw("same");

    c2->SaveAs(("output/dpt_jet_med_vs_pp_pt_R" + jetR + "_" + jobID + ".pdf").c_str());
}