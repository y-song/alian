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

void plot_raa()
{
    SetStyle();
    
    TFile *f_pp = new TFile("/rstorage/youqi/1707049/AnalysisResultsFinal.root", "READ");
    TFile *f    = new TFile("/rstorage/youqi/1703291/AnalysisResultsFinal.root", "READ");
    TH1D *h_pp = (TH1D *)f_pp->Get("jet_pT");
    TH1D *h = (TH1D *)f->Get("jet_pT");

    double scale = 1.0024369111/1889506.0;
    double scale_pp = 0.38707892174/1999187.0;
    h->Scale(scale);
    h_pp->Scale(scale_pp);

    h_pp->Rebin(5);
    h->Rebin(5);
    
    h->Divide(h_pp);
    h->GetXaxis()->SetRangeUser(100,200);
    h->GetYaxis()->SetRangeUser(0,1);
    h->GetYaxis()->SetTitle("#it{R}_{AA}");

    TCanvas *c = new TCanvas();
    h->Draw();
    c->SaveAs("output/raa.png");
}