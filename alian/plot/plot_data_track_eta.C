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
    if (!hist) return;
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

void plot_data_track_eta(std::string file_name)
{
    SetStyle();

    std::string infile  = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    std::cout << "Opening file: " << infile << std::endl;
    TFile *f = new TFile(infile.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Could not open file " << infile << std::endl;
        return;
    }

    TH2D *h_track_eta_runnumber = (TH2D *)f->Get("track_eta_runnumber");
    if (!h_track_eta_runnumber) {
        std::cerr << "Error: track_eta_runnumber not found in " << infile << std::endl;
        return;
    }

    std::cout << "Creating canvas..." << std::endl;
    TString canvasName = Form("c_track_eta_%s", file_name.c_str());
    TCanvas *c = new TCanvas(canvasName, "c", 700, 800);
    c->cd();
    ProcessCanvas(c);

    TString pad1Name = canvasName + "_pad1";
    TString pad2Name = canvasName + "_pad2";

    TPad *pad1 = new TPad(pad1Name, "pad1", 0, 0.5, 1, 1.0);
    pad1->SetBottomMargin(0.);
    pad1->Draw();

    TPad *pad2 = new TPad(pad2Name, "pad2", 0, 0, 1, 0.5);
    pad2->SetTopMargin(0.);
    pad2->SetBottomMargin(0.15);
    pad2->Draw();

    TLegend *leg = new TLegend(0.2, 0.1, 0.4, 0.55);
    leg->SetBorderSize(0);

    int bins[] = {57, 60, 74, 75, 88, 101, 115, 131, 146};
    int colors[] = {kRed, kBlue, kGreen+2, kMagenta, kOrange, kCyan, kRed+3, kOrange+2, kGreen};
    int styles[] = {kFullCircle, kFullSquare, kFullTriangleUp, kFullStar, kFullDiamond, kFullCross, kFullSquare, kFullCircle, kFullTriangleDown};

    TH1D *lastProj = nullptr;

    for (int i = 0; i < 9; ++i) {
        std::cout << "Processing bin " << bins[i] << " (index " << i << ")..." << std::endl;
        TString name = Form("proj_%d", bins[i]);
        TH1D *proj = h_track_eta_runnumber->ProjectionY(name, bins[i]);

        if (!proj) {
            std::cerr << "Warning: Projection for bin " << bins[i] << " returned nullptr!" << std::endl;
            continue;
        }

        lastProj = proj;
        FormatHist(proj, colors[i], styles[i]);

        if (pad1) {
            pad1->cd();
            if (i == 0) {
                std::cout << "Drawing first projection..." << std::endl;
                proj->DrawNormalized();
            } else {
                std::cout << "Drawing projection " << i+1 << "..." << std::endl;
                proj->DrawNormalized("same");
            }
        }
        leg->AddEntry(proj, Form("run number %d", i + 1), "lp");
    
        // Ratio calculation
        std::cout << "Creating ratio histogram for bin " << bins[i] << "..." << std::endl;
        double xmin = proj->GetXaxis()->GetXmin();
        double xmax = proj->GetXaxis()->GetXmax();
        int nBinsX = proj->GetNbinsX();

        TH1D *h_ratio = new TH1D(Form("ratio_%d", bins[i]), "", nBinsX, xmin, xmax);
        if (!h_ratio) {
            std::cerr << "Warning: Could not create ratio histogram for bin " << bins[i] << std::endl;
            continue;
        }

        h_ratio->SetTitle("");
        h_ratio->GetYaxis()->SetTitle("Ratio with (#eta>0)");

        for (int b = 1; b <= nBinsX; ++b) {
            double eta = proj->GetBinCenter(b);
            if (eta < 0) {
                int reflectedBin = nBinsX - b + 1;
                if (reflectedBin < 1 || reflectedBin > nBinsX) continue;

                double content = proj->GetBinContent(b);
                double err = proj->GetBinError(b);
                double reflectedContent = proj->GetBinContent(reflectedBin);
                double reflectedErr = proj->GetBinError(reflectedBin);
                if (reflectedContent != 0) {
                    h_ratio->SetBinContent(b, content / reflectedContent);
                    h_ratio->SetBinError(b, content / reflectedContent * TMath::Sqrt( std::pow(err/content, 2.0) + std::pow(reflectedErr/reflectedContent, 2.0) ));
                } else {
                    h_ratio->SetBinContent(b, 0);
                }
            } else {
                if (proj->GetBinContent(b) != 0) {
                    h_ratio->SetBinContent(b, 1.0);
                    h_ratio->SetBinError(b, proj->GetBinError(b) / proj->GetBinContent(b));
                } else {
                    h_ratio->SetBinContent(b, 0);
                }
            }
        }

        if (pad2) {
            pad2->cd();
            FormatHist(h_ratio, colors[i], styles[i]);
            h_ratio->GetYaxis()->SetRangeUser(0.9, 1.1);
            if (i == 0) {
                h_ratio->Draw("P");
            } else {
                h_ratio->Draw("P same");
            }
        }
    }

    if (pad1) {
        pad1->cd();
        leg->Draw();
    }

    if (pad2) {
        pad2->cd();
        if (lastProj) {
            TLine *line = new TLine(lastProj->GetXaxis()->GetXmin(), 1.0, lastProj->GetXaxis()->GetXmax(), 1.0);
            line->SetLineStyle(2);
            line->SetLineColor(kBlack);
            line->Draw();
        }
    }

    std::cout << "Saving canvas..." << std::endl;
    c->SaveAs(Form("output/data_track_eta_runnumber_%s.png", file_name.c_str()));
}
