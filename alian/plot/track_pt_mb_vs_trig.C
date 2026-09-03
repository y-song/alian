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
    l->AddEntry("NULL", "OO data, 0#minus100%", "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void ProcessCanvas(TCanvas *Canvas)
{
    if (!Canvas) return;
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
void track_pt_mb_vs_trig()
{
    SetStyle();
    
    TFile *f_trig = new TFile("/rstorage/youqi/1927065/AnalysisResultsFinal.root", "READ");
    TFile *f = new TFile("/rstorage/youqi/1934255/AnalysisResultsFinal.root", "READ");
    // TH1D *h1 = (TH1D *)f->Get("track_pT");
    TH1D *h2_trig = (TH1D *)f_trig->Get("track_in_jet_event_pT");
    TH1D *h3_trig = (TH1D *)f_trig->Get("track_in_jet_pT");
    TH1D *h2 = (TH1D *)f->Get("track_in_jet_event_pT");
    TH1D *h3 = (TH1D *)f->Get("track_in_jet_pT");
    h2_trig->GetXaxis()->SetRangeUser(0,20);
    h3_trig->GetXaxis()->SetRangeUser(0,20);
    h2->GetXaxis()->SetRangeUser(0,20);
    h3->GetXaxis()->SetRangeUser(0,20);

    h2_trig->Rebin(5);
    h3_trig->Rebin(5);
    h2->Rebin(5);
    h3->Rebin(5);
    h2_trig->Scale(1.0/h2_trig->Integral(0,20));
    h3_trig->Scale(1.0/h3_trig->Integral(0,20));
    h2->Scale(1.0/h2->Integral(0,20));
    h3->Scale(1.0/h3->Integral(0,20));

    TCanvas *c = new TCanvas();
    c->SetCanvasSize(700, 500);
    c->cd();
    ProcessCanvas(c);
    TString pad1Name = "pad1";
    TString pad2Name = "pad2";

    TPad *pad1 = new TPad(pad1Name, "pad1", 0, 0.5, 1, 1.0);
    pad1->SetLogy();
    pad1->Draw();

    TPad *pad2 = new TPad(pad2Name, "pad2", 0, 0, 1, 0.5);
    pad2->SetTopMargin(0.);
    pad2->SetBottomMargin(0.15);
    pad2->Draw();

    
    pad1->cd();
    TLegend *leg1 = new TLegend(0.4, 0.56, 0.9, 0.88, "");
    addLegendInfo(leg1, "", "", "0.4");
    // FormatHist(leg1, h1, "all tracks", kRed+2);
    FormatHist(leg1, h2_trig, "all tracks in events containing jets (trig)", kRed);
    FormatHist(leg1, h3_trig, "all tracks in jets (MB)", kRed);
    FormatHist(leg1, h2, "all tracks in events containing jets (MB)", kGreen+2);
    FormatHist(leg1, h3, "all tracks in jets (MB)", kBlue+2);
    
    // h1->Draw();
    // h2_trig->Draw("same");
    h3_trig->Draw("same");
    // h2->Draw("same");
    h3->Draw("same");
    leg1->Draw("same");

    pad2->cd();
    h3_trig->Divide(h3);
    h3_trig->Draw();

    c->SaveAs("output/track_pt_mb_vs_trig.png");
}