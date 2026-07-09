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
    gStyle->SetPalette(kBird);
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

// Project the cent_rho TH2 (x = rho, y = centrality) onto the rho axis for a
// centrality range, then normalize to a probability density (unit area).
TH1D *GetRhoPDF(TH2 *h2, double cent_lo, double cent_hi, int color, const char *name)
{
    TH2 *h2_cut = (TH2 *)h2->Clone(Form("%s_cut", name));
    h2_cut->GetYaxis()->SetRangeUser(cent_lo, cent_hi);
    h2_cut->GetXaxis()->SetRangeUser(0, 50);
    TH1D *h = h2_cut->ProjectionX(name);
    h->Scale(1.0 / h->Integral("width"));
    h->SetLineColor(color);
    h->SetLineWidth(2);
    return h;
}

void plot_rho()
{
    SetStyle();

    TFile *f = new TFile("output/test.root", "READ");
    TH2 *h_cent_rho = (TH2 *)f->Get("cent_rho");

    TH1D *h_0_10   = GetRhoPDF(h_cent_rho, 0, 10, kBlack, "rho_cent_0_10");
    TH1D *h_10_30  = GetRhoPDF(h_cent_rho, 10, 30, kRed, "rho_cent_10_30");
    TH1D *h_30_100 = GetRhoPDF(h_cent_rho, 30, 100, kBlue, "rho_cent_30_100");

    TCanvas *c = new TCanvas();
    c->cd();

    gPad->SetLogy();
    h_0_10->GetXaxis()->SetTitle("#rho (GeV)");
    h_0_10->GetYaxis()->SetTitle("Probability density");
    h_0_10->GetYaxis()->SetRangeUser(1e-5, 1);
    h_0_10->Draw("hist");
    h_10_30->Draw("hist same");
    h_30_100->Draw("hist same");

    TLegend *l = new TLegend(0.6, 0.65, 0.88, 0.85);
    l->AddEntry(h_0_10, "0#minus10%", "l");
    l->AddEntry(h_10_30, "10#minus30%", "l");
    l->AddEntry(h_30_100, "30#minus100%", "l");
    l->Draw("same");

    c->SaveAs("output/rho_pdf.png");

    TH2 *h_sigma_rho = (TH2 *)f->Get("sigma_rho");
    h_sigma_rho->GetXaxis()->SetRangeUser(0, 30);
    h_sigma_rho->GetXaxis()->SetTitle("#rho (GeV)");
    h_sigma_rho->GetYaxis()->SetRangeUser(0, 10);
    h_sigma_rho->GetYaxis()->SetTitle("#sigma (GeV)");
    h_sigma_rho->GetZaxis()->SetRangeUser(1, 1e5);
    h_sigma_rho->GetZaxis()->SetTitle("Event counts");

    TCanvas *c2 = new TCanvas();
    c2->cd();
    gPad->SetLogz();
    gPad->SetRightMargin(0.15);
    h_sigma_rho->Draw("colz");
    c2->SaveAs("output/sigma_rho.png");
}
