#include <typeinfo>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <TStyle.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>
#include <TSystem.h>

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
    gStyle->SetFrameLineWidth(2);
    gStyle->SetFrameFillColor(kWhite);
    gStyle->SetPadColor(10);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetHistLineWidth(2);
    gStyle->SetHistLineColor(kRed);
    gStyle->SetFuncWidth(2);
    gStyle->SetFuncColor(kGreen);
    gStyle->SetLineWidth(2);
    gStyle->SetLabelColor(kBlack, "xyz");
    gStyle->SetTitleFillColor(kWhite);
    gStyle->SetTitleFontSize(0.04);
    gStyle->SetTextSizePixels(30);
    gStyle->SetTextFont(42);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    gStyle->SetLegendFont(42);
}

void FormatHist(TLegend *l, TH1 *hist, TString text, int color)
{
    hist->SetMarkerSize(0.6);
    hist->SetMarkerStyle(8);
    hist->SetMarkerColor(color);
    hist->SetLineColor(color);
    hist->SetLineWidth(2);

    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);

    l->AddEntry(hist, text, "pl");

    return;
}

void addLegendInfo(TLegend *l, std::string pt_min, std::string pt_max, std::string jetR)
{
    l->SetTextSize(0.038);
    l->AddEntry("NULL", "OO data, 0#minus100%", "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void ProcessCanvas(TCanvas *Canvas)
{
    if (!Canvas)
        return;

    gStyle->SetOptStat(0);
    Canvas->SetHighLightColor(1);
    Canvas->SetFillColor(0);
    Canvas->SetBorderMode(0);
    Canvas->SetBorderSize(2);
    Canvas->SetTickx(1);
    Canvas->SetTicky(1);
    Canvas->SetFrameBorderMode(0);
    Canvas->SetFrameLineWidth(2);
    Canvas->SetFrameBorderMode(1);
}

/*
 * Rebin hist so that it has the same bin edges as reference.
 *
 * This assumes that the reference bin edges coincide with existing
 * bin boundaries in hist.
 */
TH1D *RebinToMatch(TH1D *hist, const TH1D *reference, const char *newName)
{
    const int nBins = reference->GetNbinsX();
    std::vector<double> binEdges(nBins + 1);

    for (int i = 1; i <= nBins; ++i) {
        binEdges[i - 1] =
            reference->GetXaxis()->GetBinLowEdge(i);
    }

    binEdges[nBins] =
        reference->GetXaxis()->GetBinUpEdge(nBins);

    return static_cast<TH1D *>(
        hist->Rebin(nBins, newName, binEdges.data())
    );
}

void MakeComparisonPlot(TH1D *hTrig, TH1D *hMB, const char *canvasName, const char *ratioName, const char *trigLegend, const char *mbLegend, const char *outputName)
{
    TCanvas *c = new TCanvas(canvasName, canvasName, 1200, 1000);

    ProcessCanvas(c);

    TString pad1Name = TString::Format("%s_pad1", canvasName);
    TString pad2Name = TString::Format("%s_pad2", canvasName);

    /*
     * The pads share the boundary at y = 0.35.
     * Setting both margins at that boundary to zero makes
     * the upper and lower frames touch.
     */
    TPad *pad1 = new TPad(pad1Name, "pad1", 0.0, 0.35, 1.0, 1.0);

    pad1->SetLeftMargin(0.12);
    pad1->SetRightMargin(0.03);
    pad1->SetTopMargin(0.08);
    pad1->SetBottomMargin(0.0);
    pad1->SetLogy();
    pad1->SetTickx(1);
    pad1->SetTicky(1);
    pad1->Draw();

    TPad *pad2 = new TPad(pad2Name, "pad2", 0.0, 0.0, 1.0, 0.35);

    pad2->SetLeftMargin(0.12);
    pad2->SetRightMargin(0.03);
    pad2->SetTopMargin(0.0);
    pad2->SetBottomMargin(0.27);
    pad2->SetTickx(1);
    pad2->SetTicky(1);
    pad2->Draw();

    pad1->cd();

    TLegend *leg1 = new TLegend(0.4, 0.6, 0.92, 0.90, "");

    addLegendInfo(leg1, "", "", "0.4");

    FormatHist(leg1, hTrig, trigLegend, kRed+2);
    FormatHist(leg1, hMB, mbLegend, kBlue + 2);

    hTrig->GetYaxis()->SetTitle("Self-normalized counts");
    hTrig->GetYaxis()->SetTitleOffset(0.95);
    hTrig->GetYaxis()->SetTitleSize(0.055);
    hTrig->GetYaxis()->SetLabelSize(0.05);

    /*
     * The x-axis is labeled only on the bottom panel because
     * the two panels touch.
     */
    hTrig->GetXaxis()->SetTitleSize(0.0);
    hTrig->GetXaxis()->SetLabelSize(0.0);

    hTrig->Draw("E");
    hMB->Draw("E SAME");
    leg1->Draw("SAME");

    pad2->cd();

    TH1D *ratio = static_cast<TH1D *>(hTrig->Clone(ratioName));

    ratio->SetTitle("");
    ratio->Divide(hMB);

    ratio->SetMinimum(0.5);
    ratio->SetMaximum(1.5);

    ratio->GetYaxis()->SetTitle("Ratio over MB");
    ratio->GetYaxis()->SetTitleOffset(0.60);
    ratio->GetYaxis()->SetTitleSize(0.085);
    ratio->GetYaxis()->SetLabelSize(0.085);
    ratio->GetYaxis()->SetNdivisions(505);

    ratio->GetXaxis()->SetTitle("p_{T} (GeV/#it{c})");
    ratio->GetXaxis()->SetTitleOffset(1.05);
    ratio->GetXaxis()->SetTitleSize(0.085);
    ratio->GetXaxis()->SetLabelSize(0.085);

    ratio->SetMarkerSize(0.6);
    ratio->SetMarkerStyle(8);
    ratio->SetMarkerColor(kRed+2);
    ratio->SetLineColor(kRed+2);
    ratio->SetLineWidth(2);

    ratio->Draw("E");

    TLine *unityLine = new TLine(ratio->GetXaxis()->GetXmin(), 1.0, 20.0, 1.0);

    unityLine->SetLineColor(kGray + 2);
    unityLine->SetLineStyle(2);
    unityLine->SetLineWidth(2);
    unityLine->Draw("SAME");

    ratio->Draw("E SAME");

    c->SaveAs(outputName);
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

    /*
    * Use the rebinned h3_trig as the reference binning for h3.
    */
    h3_trig->Rebin(6);
    h3 = RebinToMatch(h3, h3_trig, "h3_rebinned");

    /*
    * Use the rebinned h2_trig as the reference binning for h2.
    */
    h2_trig->Rebin(6);
    h2 = RebinToMatch(h2, h2_trig, "h2_rebinned");

    h2_trig->GetXaxis()->SetRangeUser(0, 20);
    h3_trig->GetXaxis()->SetRangeUser(0, 20);
    h2->GetXaxis()->SetRangeUser(0, 20);
    h3->GetXaxis()->SetRangeUser(0, 20);

    h2_trig->Scale(1.0 / h2_trig->Integral(0, 20));
    h3_trig->Scale(1.0 / h3_trig->Integral(0, 20));
    h2->Scale(1.0 / h2->Integral(0, 20));
    h3->Scale(1.0 / h3->Integral(0, 20));

    gSystem->mkdir("output", kTRUE);
    MakeComparisonPlot(h3_trig, h3, "c_track_in_jet", "ratio_h3_trig_over_h3", "#splitline{JE triggered: All tracks in jets with}{R = 0.4, p_{T} #minus #rhoA > 20 GeV}", "#splitline{MB: All tracks in jets with}{R = 0.4, p_{T} #minus #rhoA > 20 GeV}", "output/track_in_jet_pt_mb_vs_trig.png");
    MakeComparisonPlot(h2_trig, h2, "c_track_in_jet_event", "ratio_h2_trig_over_h2", "#splitline{JE triggered: All tracks in events containing jets with}{R = 0.4, p_{T} #minus #rhoA > 20 GeV}", "#splitline{MB: All tracks in events containing jets with}{R = 0.4, p_{T} #minus #rhoA > 20 GeV}", "output/track_in_jet_event_pt_mb_vs_trig.png");
}