#include <iostream>
#include <string>
#include <vector>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>

using namespace std;

// ── hardcode file pairs here ──────────────────────────────────────────────────
struct FilePair {
    std::string pbpb_file;
    std::string pp_file;
    std::string label;
    int         color;
    int         style;
};

// const std::vector<FilePair> kPairs = {
//     {"output/eec_R04_100_120_1652502.root",
//      "output/eec_R04_100_120_1652402.root",
//      "z_{cut} = 0.1", kRed,     1},
//     {"output/eec_R04_100_120_1652717.root",
//      "output/eec_R04_100_120_1652617.root",
//      "z_{cut} = 0.2", kBlue,    1},
//     {"output/eec_R04_100_120_1652917.root",
//      "output/eec_R04_100_120_1652817.root",
//      "z_{cut} = 0.3", kGreen+2, 1},
//     {"output/eec_R04_100_120_1656144.root",
//      "output/eec_R04_100_120_1656044.root",
//      "z_{cut} = 0.45", kOrange, 1},
// };
const std::vector<FilePair> kPairs = {
    {"output/eec_R06_100_120_1698385.root",
     "output/eec_R06_100_120_1697883.root",
     "z_{cut} = 0.1", kRed,     1},
    {"output/eec_R06_100_120_1698585.root",
     "output/eec_R06_100_120_1698485.root",
     "z_{cut} = 0.2", kBlue,    1},
    {"output/eec_R06_100_120_1698785.root",
     "output/eec_R06_100_120_1698685.root",
     "z_{cut} = 0.3", kGreen+2, 1},
    // {"output/eec_R06_100_120_1656144.root",
    //  "output/eec_R06_100_120_1656044.root",
    //  "z_{cut} = 0.45", kOrange, 1},
};
// ─────────────────────────────────────────────────────────────────────────────

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

void ProcessCanvas(TCanvas *c)
{
    gStyle->SetOptStat(0);
    c->SetHighLightColor(1);
    c->SetFillColor(0);
    c->SetBorderMode(0);
    c->SetBorderSize(2);
    c->SetTickx(1);
    c->SetTicky(1);
    c->SetFrameBorderMode(0);
    c->SetFrameLineWidth(1);
    c->SetFrameBorderMode(1);
    c->SetGridx(1);
    c->SetGridy(1);
}

// Returns h_ab / sqrt(h_aa * h_bb), detached from the file.
TH1D *GetCab(TFile *f)
{
    TH1D *h_aa = (TH1D *)f->Get("h_eec_aa_clone");
    TH1D *h_bb = (TH1D *)f->Get("h_eec_bb_clone");
    TH1D *h_ab = (TH1D *)f->Get("h_eec_ab_clone");
    if (!h_aa || !h_bb || !h_ab) {
        std::cerr << "ERROR: missing histogram in " << f->GetName() << std::endl;
        return nullptr;
    }

    TH1D *h_denom = (TH1D *)h_aa->Clone("h_denom_tmp");
    h_denom->SetDirectory(0);
    h_denom->Multiply(h_bb);
    for (int i = 1; i <= h_denom->GetNbinsX(); i++) {
        double val    = h_denom->GetBinContent(i);
        double err    = h_denom->GetBinError(i);
        double sqrtval = (val > 0) ? sqrt(val) : 0.;
        double sqrterr = (val > 0) ? err / (2. * sqrtval) : 0.;
        h_denom->SetBinContent(i, sqrtval);
        h_denom->SetBinError(i, sqrterr);
    }

    TH1D *h_cab = (TH1D *)h_ab->Clone("h_cab_tmp");
    h_cab->SetDirectory(0);
    h_cab->Divide(h_denom);
    delete h_denom;
    return h_cab;
}

// Returns h_ab detached from the file.
TH1D *GetEECab(TFile *f)
{
    TH1D *h_ab = (TH1D *)f->Get("h_eec_ab_clone");
    if (!h_ab) {
        std::cerr << "ERROR: missing histogram in " << f->GetName() << std::endl;
        return nullptr;
    }
    TH1D *out = (TH1D *)h_ab->Clone("h_eecab_tmp");
    out->SetDirectory(0);
    return out;
}

double FindCrossing(TH1D *h, int half_window = 2, TF1 **fit_out = nullptr)
{
    int nbins = h->GetNbinsX();
    int cross_bin = -1;
    int first_bin = h->FindBin(0.005);

    for (int i = first_bin; i < nbins; i++) {
        double v0 = h->GetBinContent(i);
        double v1 = h->GetBinContent(i + 1);
        if (v0 <= 0 || v1 <= 0) continue;
        if ((v0 - 1.) * (v1 - 1.) < 0.) {
            cross_bin = i;
            break;
        }
    }
    if (cross_bin < 0) {
        std::cerr << "WARNING: no y=1 crossing found in " << h->GetName() << std::endl;
        return -1.;
    }

    int bin_lo = std::max(1,     cross_bin - half_window);
    int bin_hi = std::min(nbins, cross_bin + half_window);
    double x_lo = h->GetBinLowEdge(bin_lo);
    double x_hi = h->GetBinLowEdge(bin_hi + 1);

    std::string fname = std::string("fline_") + h->GetName();
    // Fit y = a + b*log(x) in the window, using bin errors as weights
    TF1 *fline = new TF1(fname.c_str(), "[0] + [1]*log(x)", x_lo, x_hi);
    h->Fit(fline, "RQN");   // R = use function range, Q = quiet, N = don't draw

    double a = fline->GetParameter(0);
    double b = fline->GetParameter(1);

    if (fit_out)    *fit_out = fline;  // hand ownership to caller
    else    delete fline;

    if (fabs(b) < 1e-30) return -1.;
    return exp((1. - a) / b);
}

struct OverlayResult {
    std::vector<TH1D *> ratios;
    std::vector<TF1  *> fits;
};

OverlayResult DrawOverlay(TLegend *l, TH1D *(*getter)(TFile *), const char *ytitle, double ymin, double ymax)
{
    OverlayResult result;
    bool first = true;

    for (size_t ip = 0; ip < kPairs.size(); ip++) {
        const FilePair &p = kPairs[ip];

        TFile *f_pbpb = new TFile(p.pbpb_file.c_str(), "READ");
        TFile *f_pp   = new TFile(p.pp_file.c_str(),   "READ");
        if (!f_pbpb || f_pbpb->IsZombie()) {
            std::cerr << "ERROR: cannot open " << p.pbpb_file << std::endl; continue;
        }
        if (!f_pp || f_pp->IsZombie()) {
            std::cerr << "ERROR: cannot open " << p.pp_file   << std::endl; continue;
        }

        TH1D *h_pbpb = getter(f_pbpb);
        TH1D *h_pp   = getter(f_pp);

        f_pbpb->Close(); delete f_pbpb;
        f_pp->Close();   delete f_pp;

        if (!h_pbpb || !h_pp) { delete h_pbpb; delete h_pp; continue; }

        std::string rname = "ratio_" + std::to_string(ip);
        TH1D *ratio = (TH1D *)h_pbpb->Clone(rname.c_str());
        ratio->SetDirectory(0);
        ratio->Divide(h_pp);
        delete h_pbpb;
        delete h_pp;

        ratio->SetLineColor(p.color);
        ratio->SetMarkerColor(p.color);
        ratio->SetLineStyle(p.style);
        ratio->SetLineWidth(2);
        ratio->SetMarkerStyle(20 + (int)ip);
        ratio->SetMarkerSize(0.5);
        ratio->GetXaxis()->SetRangeUser(0.005, 0.4);
        ratio->GetYaxis()->SetRangeUser(ymin, ymax);
        ratio->GetXaxis()->SetTitle("#it{R}_{L}");
        ratio->GetYaxis()->SetTitle(ytitle);

        ratio->Draw(first ? "E" : "E same");

        TF1 *fit = nullptr;
        double x_cross = FindCrossing(ratio, 2, &fit);
        if (x_cross > 0)
            std::cout << p.label << ":  crossover = " << x_cross << std::endl;
        if (fit) {
            fit->SetLineColor(p.color);
            fit->SetLineStyle(2);
            fit->SetLineWidth(2);
            fit->Draw("same");
            result.fits.push_back(fit);
        }

        l->AddEntry(ratio, p.label.c_str(), "pl");
        result.ratios.push_back(ratio);
        first = false;
    }
    return result;
}

void MakePlot(const char *canvas_name, TH1D *(*getter)(TFile *), const char *ytitle, double ymin, double ymax, const char *outpath)
{
    TCanvas *c = new TCanvas(canvas_name, canvas_name, 800, 600);
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogx();

    TLegend *l = new TLegend(0.6, 0.65, 0.9, 0.88);
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0);

    auto result = DrawOverlay(l, getter, ytitle, ymin, ymax);

    TLine unity(0.005, 1.0, 0.4, 1.0);
    unity.SetLineColor(kGray+1);
    unity.SetLineStyle(2);
    unity.SetLineWidth(2);
    unity.Draw("same");

    l->Draw("same");
    c->SaveAs(outpath);

    delete c;
    delete l;
    for (auto *h : result.ratios) delete h;
    for (auto *f : result.fits)   delete f;
}

void plot_cab()
{
    gStyle->SetOptStat(0);
    SetStyle();

    MakePlot("c_cab",
             GetCab,
             "C_{AB}(PbPb/pp)",
             0.5, 2.5,
             "output/cab_overlay.pdf");

    MakePlot("c_eecab",
             GetEECab,
             "EEC_{AB}(PbPb/pp)",
             0.5, 2.5,
             "output/eecab_overlay.pdf");
}