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

void FormatHist(TH1 *hist, int markercolor = 1, int markerstyle = 8)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(0.5);

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
}

void plot_data_pt(std::string file_name)
{
    SetStyle();
    
    std::string infile  = "/home/youqi/alian/alian/output/embed.root";//"/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    TFile *f = new TFile(infile.c_str(), "READ");

    TH1D *h_sub_pT_all = (TH1D *)f->Get("sub_pT");
    // TH1D *h_sub_pT_all = (TH1D *)f->Get("pp_pT"); // for pp

    TH2 *h_combined_pT_sub_pT_matched = (TH2 *)f->Get("combined_pT_sub_pT_matched");
    TH1D *h_sub_pT_matched = h_combined_pT_sub_pT_matched->ProjectionX(Form("%s_projx", h_combined_pT_sub_pT_matched->GetName()));
    // TH2 *h_combined_pT_sub_pT_matched = (TH2 *)f->Get("sub_pT_pp_pT_matched"); // for pp
    // TH1D *h_sub_pT_matched = h_combined_pT_sub_pT_matched->ProjectionX(Form("%s_projx", h_combined_pT_sub_pT_matched->GetName())); // for pp

    TCanvas *c = new TCanvas("c", "c", 700, 500);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogy();
    
    FormatHist(h_sub_pT_all, kBlue, kFullSquare);
    // FormatHist(h_JetPt, kBlue, kFullTriangleUp);
    FormatHist(h_sub_pT_matched, kRed, kFullCircle);
    h_sub_pT_all->Draw();
    // h_JetPt->Draw("same");
    h_sub_pT_matched->Draw("same");

    c->SaveAs("output/embed_pt.png");
}