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
    hist->SetMarkerSize(0.3);
    hist->SetMarkerStyle(8);
    hist->SetMarkerColor(color);
    hist->SetLineColor(color);
    hist->GetYaxis()->SetTitle("Counts");
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
    l->AddEntry("NULL", "OO 0#minus10%", "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void plot_track_pt()
{
    SetStyle();
    
    TFile *f = new TFile("~/temp/track_qa/AnalysisResultsFinal.root", "READ");
    TH1D *h1 = (TH1D *)f->Get("track_pT");
    TH1D *h2 = (TH1D *)f->Get("track_in_jet_event_pT");
    TH1D *h3 = (TH1D *)f->Get("track_in_jet_pT");

    TCanvas *c = new TCanvas();
    c->SetCanvasSize(700, 500);
    c->cd();
    gPad->SetLogy();
    
    TLegend *leg1 = new TLegend(0.4, 0.56, 0.9, 0.88, "");
    addLegendInfo(leg1, "", "", "0.4");
    FormatHist(leg1, h1, "all tracks", kRed+2);
    FormatHist(leg1, h2, "all tracks in events containing jets", kGreen+2);
    FormatHist(leg1, h3, "all tracks in jets", kBlue+2);
    
    h1->Draw();
    h2->Draw("same");
    h3->Draw("same");
    leg1->Draw("same");

    c->SaveAs("output/oo_track_pt.png");
}