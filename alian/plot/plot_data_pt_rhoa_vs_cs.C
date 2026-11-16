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

TH1D *RebinToMatch(TH1D *h, TH1D *h_template, const char *name)
{
    TH1D *h_new = (TH1D *)h_template->Clone(name);
    h_new->Reset();
    for (int i = 0; i <= h_new->GetNbinsX() + 1; i++)
    {
        int j = h->FindBin(h_new->GetBinCenter(i));
        h_new->SetBinContent(i, h->GetBinContent(j));
        h_new->SetBinError(i, h->GetBinError(j));
    }
    return h_new;
}

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

void FormatHist(TH1 *hist, int markercolor = 1, int markerstyle = 8, double markersize = 0.5)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(markersize);

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

void plot_data_pt_rhoa_vs_cs(std::string file_name_rhoa, std::string file_name_cs)
{
    SetStyle();
    
    std::string infile_rhoa = "/rstorage/youqi/" + file_name_rhoa + "/AnalysisResultsFinal.root";
    std::string infile_cs = "/rstorage/youqi/" + file_name_cs + "/AnalysisResultsFinal.root";
    TFile *f_rhoa = new TFile(infile_rhoa.c_str(), "READ");
    TFile *f_cs = new TFile(infile_cs.c_str(), "READ");

    TH1D *h_rhoa = (TH1D *)f_rhoa->Get("sub_pT");
    TH1D *h_cs = (TH1D *)f_cs->Get("sub_pT");

    TH2 *h_rhoa_sub_pT_pp_pT_matched = (TH2 *)f_rhoa->Get("sub_pT_pp_pT_matched");
    TH2 *h_cs_sub_pT_pp_pT_matched = (TH2 *)f_cs->Get("sub_pT_pp_pT_matched");
    TH2 *h_rhoa_sub_pTg_pp_pTg_matched = (TH2 *)f_rhoa->Get("sub_pTg_pp_pTg_matched");
    TH2 *h_cs_sub_pTg_pp_pTg_matched = (TH2 *)f_cs->Get("sub_pTg_pp_pTg_matched");
    TH1D *h_rhoa_matched = h_rhoa_sub_pT_pp_pT_matched->ProjectionY("h_rhoa_matched");
    TH1D *h_rhoa_g_matched = h_rhoa_sub_pTg_pp_pTg_matched->ProjectionY("h_rhoa_g_matched");
    TH1D *h_cs_matched = h_cs_sub_pT_pp_pT_matched->ProjectionY("h_cs_matched");
    TH1D *h_cs_g_matched = h_cs_sub_pTg_pp_pTg_matched->ProjectionY("h_cs_g_matched");
    
    h_rhoa->GetXaxis()->SetRangeUser(0, 200);
    h_rhoa_matched->GetXaxis()->SetRangeUser(0, 200);
    h_rhoa_g_matched->GetXaxis()->SetRangeUser(0, 200);
    h_cs->GetXaxis()->SetRangeUser(0, 200);
    h_cs_matched->GetXaxis()->SetRangeUser(0, 200);
    h_cs_g_matched->GetXaxis()->SetRangeUser(0, 200);

    TCanvas *c = new TCanvas("c", "c", 700, 500);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogy();
    
    FormatHist(h_rhoa, kBlue, kFullCircle, 1);
    FormatHist(h_rhoa_matched, kRed, kFullCircle, 1);
    FormatHist(h_rhoa_g_matched, kGreen+2, kFullCircle, 1);
    FormatHist(h_cs, kBlue, 25);
    FormatHist(h_cs_matched, kBlack, 25);
    FormatHist(h_cs_g_matched, kBlack, 25);

    // h_rhoa->Draw();
    h_rhoa_matched->Draw();
    h_rhoa_g_matched->Draw("same");

    // h_cs->Draw("same");
    h_cs_matched->Draw("same");
    h_cs_g_matched->Draw("same");

    cout << h_rhoa_g_matched->Integral(h_rhoa_g_matched->FindBin(40), h_rhoa_g_matched->FindBin(60)) << endl;
    cout << h_cs_g_matched->Integral(h_cs_g_matched->FindBin(40), h_cs_g_matched->FindBin(60)) << endl;
    c->SaveAs(("output/embed_pt_rhoa_vs_cs_" + file_name_rhoa + "_" + file_name_cs + ".png").c_str());

    TH1D *h_ratio = RebinToMatch(h_cs_g_matched, h_rhoa_g_matched, "h_cs_g_matched_over_rhoa");
    h_ratio->Divide(h_rhoa_g_matched);
    // h_ratio->Rebin(5);
    h_ratio->SetMinimum(0.5);
    h_ratio->SetMaximum(1.5);
    
    TCanvas *c_ratio = new TCanvas("c_ratio", "c_ratio", 700, 500);
    ProcessCanvas(c_ratio);
    c_ratio->cd();

    FormatHist(h_ratio, kBlack, 25);
    h_ratio->SetTitle(";p_{T} (GeV/c);CS / #rhoA");
    h_ratio->Draw("E");

    c_ratio->SaveAs(("output/embed_pt_ratio_rhoa_over_cs_" + file_name_rhoa + "_" + file_name_cs + ".png").c_str());
}