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

void SetStyle()
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
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

void plot_pt()
{
    SetStyle();
    
    TFile *f_pp = new TFile("/rstorage/youqi/1707049/AnalysisResultsFinal.root", "READ");
    TFile *f_hybrid = new TFile("/rstorage/youqi/hybrid/vac_hadrons/AnalysisResultsFinal.root", "READ");
    TH1D *h_pp = (TH1D *)f_pp->Get("jet_pT");
    TH1D *h = (TH1D *)f_hybrid->Get("jet_pT");

    // h_pp->GetXaxis()->SetRangeUser(100,120);
    // h->GetXaxis()->SetRangeUser(100,120);
    h_pp->Scale(1.0/h_pp->Integral(h_pp->FindBin(100), h_pp->FindBin(120)));
    h->Scale(1.0/h->Integral(h->FindBin(100), h->FindBin(120)));
    h_pp->SetLineColor(1);
    h->SetLineColor(4);

    TCanvas *c = new TCanvas();
    c->cd();
    gPad->SetLogy();

    TH1F *frame = new TH1F("frame", "My Normalized Plot", 100, 100, 120);
    // frame->SetMaximum(0.001);
    // frame->SetMinimum(0.02);
    frame->GetXaxis()->SetTitle("#it{p}_{T}^{jet} (GeV/#it{c})");
    // frame->Draw();
    
    h_pp->Draw();
    h->Draw("same");
    c->SaveAs("output/pt_hybrid_jewel.png");
}