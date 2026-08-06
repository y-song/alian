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
    Canvas->SetGridx(1);
    Canvas->SetGridy(1);
}

void plot_data_track_eta(std::string file_name)
{
    SetStyle();

    std::string infile  = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    TFile *f = new TFile(infile.c_str(), "READ");

    TH2D *h_track_eta_runnumber = (TH2D *)f->Get("track_eta_runnumber");
    if (!h_track_eta_runnumber) {
        std::cerr << "Error: track_eta_runnumber not found in " << infile << std::endl;
        return;
    }

    TCanvas *c = new TCanvas("c", "c", 700, 500);
    ProcessCanvas(c);
    c->cd();

    TLegend *leg = new TLegend(0.2, 0.17, 0.4, 0.55);
    leg->SetBorderSize(0);

    int bins[] = {57, 60, 74, 75, 88, 101, 115, 131, 146};
    int colors[] = {kRed, kBlue, kGreen+2, kMagenta, kOrange, kCyan, kRed+3, kOrange+2, kGreen};
    int styles[] = {kFullCircle, kFullSquare, kFullTriangleUp, kFullStar, kFullDiamond, kFullCross, kFullSquare, kFullCircle, kFullTriangleDown};

    for (int i = 0; i < 9; ++i) {
        TString name = Form("proj_%d", bins[i]);
        TH1D *proj = h_track_eta_runnumber->ProjectionY(name, bins[i]);

        FormatHist(proj, colors[i], styles[i]);

        if (i == 0) {
            proj->DrawNormalized();
        } else {
            proj->DrawNormalized("same");
        }
        leg->AddEntry(proj, Form("run number %d", i + 1), "lp");
    }
    leg->Draw();

    c->SaveAs(Form("output/data_track_eta_runnumber_%s.png", file_name.c_str()));
}
