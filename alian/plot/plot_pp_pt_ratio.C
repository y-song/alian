#include <TFile.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TError.h>
#include <cmath>

void plot_pp_pt_ratio()
{
    TFile fNum("/rstorage/youqi/2001196/AnalysisResultsFinal.root", "READ");
    TFile fDen("/rstorage/youqi/1986696/anchmc_simple_merged.root", "READ");

    if (fNum.IsZombie() || fDen.IsZombie()) {
        Error("ratio_pp_pT", "Cannot open an input file.");
        return;
    }

    TH1F *hNum = nullptr;
    TH1D *hDen = nullptr;
    fNum.GetObject("pp_pT", hNum);
    fDen.GetObject("hJetPtALL_det_matched_ungroomed", hDen);

    if (!hNum || !hDen) {
        Error("ratio_pp_pT", "Histogram missing or not a TH1F.");
        return;
    }

    std::cout << hNum->GetNbinsX()  << " " << hDen->GetNbinsX() << std::endl;
    hDen->Rebin(2);

    // Both histograms now have 2-unit bins.
    // Copy only the bins covering 0–200.
    auto *hNumCommon = new TH1D("hNumCommon", "", 100, 0, 200);
    auto *hDenCommon = new TH1D("hDenCommon", "", 100, 0, 200);
    hNumCommon->SetDirectory(nullptr);
    hDenCommon->SetDirectory(nullptr);
    hNumCommon->Sumw2();
    hDenCommon->Sumw2();

    for (int i = 1; i <= 100; ++i) {
        const double x = hNumCommon->GetBinCenter(i);
        const int iNum = hNum->FindBin(x);
        const int iDen = hDen->FindBin(x);

        hNumCommon->SetBinContent(i, hNum->GetBinContent(iNum));
        hNumCommon->SetBinError(i, hNum->GetBinError(iNum));

        hDenCommon->SetBinContent(i, hDen->GetBinContent(iDen));
        hDenCommon->SetBinError(i, hDen->GetBinError(iDen));
    }

    auto *hRatio = static_cast<TH1D*>(hNumCommon->Clone("hRatio_pp_pT"));
    hRatio->SetDirectory(nullptr);
    hRatio->Divide(hNumCommon, hDenCommon);

    hRatio->SetTitle(";p_{T};pp_pT / hJetPtAll_det_matched_ungroomed");
    hRatio->GetXaxis()->SetTitle("p_{T} (GeV)");
    hRatio->SetStats(false);
    hRatio->SetMarkerStyle(20);

    auto *c = new TCanvas("cRatio_pp_pT", "pT ratio", 800, 600);
    c->SetLeftMargin(0.15);
    hRatio->Draw("E1");
    c->SaveAs("output/ratio_pp_pT.pdf");

    TFile output("output/ratio_pp_pT.root", "RECREATE");
    hRatio->Write();
}