#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TColor.h>
#include <TLegend.h>
#include <TFile.h>
// Subjet prong matching fractions vs pp det pT,g, the analogue of Fig. 41 (SD zcut = 0.2)
// of the hardest-kT analysis note (Ehlers), from combined_matched_prong_match filled by
// analysis/test/embed_cs.py: category vs pp pT,g (x) and CS pT,g (y).
// The CS pT,g window [pt_min, pt_max] plays the role of the hybrid 40-120 GeV window.
// Each category is drawn as a fraction of all matched jets in the same pp pT,g bin, so the
// nine curves sum to 1 in every bin. Jets that pass SD at only one level are not in the
// histogram (the "_matched" selection requires SD at both levels).
// rebin merges pp pT,g bins (jet_pT axis is 2 GeV wide).
// Example Usage:
// root -l 'plot/plot_prong_match.C("2031425", "40", "60", "OO")'

using namespace std;

void SetStyle(Bool_t graypalette = true)
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(0);
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

void FormatHist(TH1 *hist, int markercolor = 1, int markerstyle = 8)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(0.8);

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

void addLegendInfo(TLegend *l, TString jetR, TString plot_text)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "pp jets + OO 0#minus100% " + plot_text, "h");
    l->AddEntry("NULL", "Anti-#it{k}_{T} full jets, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", "SD #it{z}_{cut} = 0.2, #beta = 0", "h");
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
}

void plot_prong_match(std::string file_name, std::string pt_min, std::string pt_max, std::string plot_text = "", int rebin = 5)
{
    gStyle->SetOptStat(0);
    SetStyle();

    std::string infile = "/rstorage/youqi/" + file_name + "/AnalysisResultsFinal.root";
    std::string jetR = "04";
    std::string jetR_str = "0.4";
    const std::string hname = "combined_matched_prong_match";

    double pt_lo = stof(pt_min);
    double pt_hi = stof(pt_max);
    TFile *f = new TFile(infile.c_str(), "READ");
    if (f->IsZombie())
    {
        std::cerr << "Cannot open " << infile << std::endl;
        delete f;
        return;
    }

    TH3 *h3 = (TH3 *)f->Get(hname.c_str());
    if (!h3)
    {
        std::cerr << "Missing histogram " << hname << " (filled by embed_cs.py)" << std::endl;
        f->Close();
        delete f;
        return;
    }

    // CS pT,g window on the y axis; drop the upper bin if pt_hi sits on its low edge
    TAxis *ay = h3->GetYaxis();
    int ybin_lo = ay->FindBin(pt_lo);
    int ybin_hi = ay->FindBin(pt_hi);
    if (ay->GetBinLowEdge(ybin_hi) >= pt_hi)
        ybin_hi--;
    std::cout << "CS pT,g window: " << ay->GetBinLowEdge(ybin_lo) << " - " << ay->GetBinUpEdge(ybin_hi) << " GeV" << std::endl;
    ay->SetRange(ybin_lo, ybin_hi);
    TH2D *h2 = (TH2D *)h3->Project3D("zx"); // x: pp pT,g, y: category
    h2->SetName("h_prong_match_ppptg_cat");
    if (rebin > 1)
        h2->RebinX(rebin);

    TH1D *h_total = h2->ProjectionX("h_prong_total", 1, h2->GetNbinsY());
    double ntot = h_total->Integral();
    std::cout << "matched jets in window (weighted): " << ntot << ", entries: " << h_total->GetEntries() << std::endl;
    if (ntot <= 0)
    {
        f->Close();
        delete f;
        return;
    }

    // order and labels follow PRONG_CATEGORY in embed_cs.py; colors are those of the same
    // categories in Fig. 41 of the hardest-kT note (matplotlib tab10)
    struct CatConfig
    {
        const char *label;
        int color;
    };
    std::vector<CatConfig> cats =
    {
        {"pure", TColor::GetColor(31, 119, 180)},
        {"leading matched, subleading unmatched", TColor::GetColor(44, 160, 44)},
        {"leading unmatched, subleading matched", TColor::GetColor(255, 127, 14)},
        {"leading matched, subleading in leading", TColor::GetColor(214, 39, 40)},
        {"leading in subleading, subleading matched", TColor::GetColor(148, 103, 189)},
        {"leading unmatched, subleading in leading", TColor::GetColor(140, 86, 75)},
        {"leading in subleading, subleading unmatched", TColor::GetColor(227, 119, 194)},
        {"swap", TColor::GetColor(127, 127, 127)},
        {"both unmatched", TColor::GetColor(188, 189, 34)},
    };

    std::vector<TH1D *> fracs;
    std::cout << "Integrated fractions in the CS pT,g window:" << std::endl;
    for (size_t icat = 0; icat < cats.size(); icat++)
    {
        int ybin = h2->GetYaxis()->FindBin(icat);
        TH1D *hnum = h2->ProjectionX(Form("h_prong_cat%zu", icat), ybin, ybin);
        std::cout << "  " << icat << " " << cats[icat].label << ": " << hnum->Integral() / ntot << std::endl;
        TH1D *hfrac = (TH1D *)hnum->Clone(Form("h_prong_frac_cat%zu", icat));
        hfrac->Divide(hnum, h_total, 1, 1, "B");
        fracs.push_back(hfrac);
    }
    // purities as defined in the embed_cs.py docstring
    std::vector<double> n_cat;
    for (size_t icat = 0; icat < cats.size(); icat++)
        n_cat.push_back(h2->Integral(1, h2->GetNbinsX(), icat + 1, icat + 1));
    std::cout << "  leading purity (0+1+3): " << (n_cat[0] + n_cat[1] + n_cat[3]) / ntot << std::endl;
    std::cout << "  subleading purity (0+2+4): " << (n_cat[0] + n_cat[2] + n_cat[4]) / ntot << std::endl;

    std::string outbase = "/home/youqi/alian/alian/output/prong_match_R" + jetR + "_" + pt_min + "_" + pt_max + "_" + file_name;
    std::string ptbin = pt_min + " < #it{p}_{T,g}^{sub} < " + pt_max + " GeV/#it{c}";

    // axis ranges as in Fig. 41: log y from 1e-3 to 10, pT,det from 0 to 150
    {
        TCanvas *c = new TCanvas("c", "c", 1100, 600);
        ProcessCanvas(c);
        c->SetRightMargin(0.36);
        c->cd();
        gPad->SetLogy();

        TLegend *l_info = new TLegend(0.17, 0.70, 0.62, 0.88);
        addLegendInfo(l_info, jetR_str, plot_text);
        l_info->AddEntry("NULL", ptbin.c_str(), "h");

        TLegend *l_cat = new TLegend(0.645, 0.25, 0.99, 0.88);
        l_cat->SetTextSize(0.03);
        l_cat->SetBorderSize(0);
        l_cat->SetFillStyle(0);

        for (size_t icat = 0; icat < cats.size(); icat++)
        {
            TH1D *h1 = fracs[icat];
            // no markers: bin-width and error bars only, as in Fig. 41
            FormatHist(h1, cats[icat].color, 1);
            h1->SetMarkerSize(0);
            h1->GetXaxis()->SetRangeUser(0, 150);
            h1->GetYaxis()->SetRangeUser(1e-3, 10);
            h1->GetXaxis()->SetTitle("#it{p}_{T,g}^{pp det} (GeV/#it{c})");
            h1->GetYaxis()->SetTitle("Fraction of matched jets");
            h1->SetTitle("Subjet prong matching");
            h1->Draw(icat == 0 ? "E" : "E same");
            l_cat->AddEntry(h1, cats[icat].label, "le");
        }
        l_info->Draw("same");
        l_cat->Draw("same");

        c->SaveAs((outbase + ".pdf").c_str());
        delete c;
        delete l_info;
        delete l_cat;
    }

    TFile *f_out = new TFile((outbase + ".root").c_str(), "RECREATE");
    h2->Write();
    h_total->Write();
    for (TH1D *h1 : fracs)
        h1->Write();
    f_out->Close();
    delete f_out;
    f->Close();
    delete f;
}
