#include <iostream>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>
#include <string>
#include <vector>
// #include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldBayes.h>
// #include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldResponse.h>
#include <cmath>

#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// ---------------------------------------------------------------------------
// RL rebinning helpers
// ---------------------------------------------------------------------------

// Bin edges of one axis, copied unchanged (works for uniform and variable bins)
static std::vector<double> copyEdges(const TAxis* ax) {
    std::vector<double> v;
    const TArrayD* b = ax->GetXbins();
    if (b && b->GetSize() == ax->GetNbins() + 1) {
        for (int i = 0; i < b->GetSize(); ++i) v.push_back((*b)[i]);
    } else {
        v.push_back(ax->GetXmin());
        v.push_back(ax->GetXmax());
    }
    return v;
}

// Bin edges of one axis after merging f adjacent bins (1-based: bins {1..f} -> 1, {f+1..2f} -> 2, ...)
static std::vector<double> mergedEdges(const TAxis* ax, int f) {
    const int n = ax->GetNbins();
    std::vector<double> v;
    const TArrayD* b = ax->GetXbins();
    if (b && b->GetSize() == n + 1) {
        for (int i = 0; i <= n / f; ++i) v.push_back((*b)[i * f]);
    } else {
        const double w = (ax->GetXmax() - ax->GetXmin()) / n;
        for (int i = 0; i <= n / f; ++i) v.push_back(ax->GetXmin() + i * f * w);
    }
    return v;
}

// Merge f adjacent bins of the RL (Y) axis of a TH3.
// Contents are summed, errors are propagated in quadrature (Sumw2-style).
// Returns nullptr if the RL bin count is not divisible by f.
static TH3D* rebinY_TH3(const TH3* h, int f, const TString& name) {
    const int nx = h->GetNbinsX(), ny = h->GetNbinsY(), nz = h->GetNbinsZ();
    if (ny % f != 0) {
        std::cerr << "rebinY_TH3: " << h->GetName() << " has " << ny
                  << " RL bins, not divisible by rebin factor " << f << std::endl;
        return nullptr;
    }
    const int ny2 = ny / f;
    std::vector<double> xe = copyEdges(h->GetXaxis());
    std::vector<double> ye = mergedEdges(h->GetYaxis(), f);
    std::vector<double> ze = copyEdges(h->GetZaxis());

    auto* h2 = new TH3D(name, h->GetTitle(), nx, xe.data(), ny2, ye.data(), nz, ze.data());
    h2->SetDirectory(0);
    h2->Sumw2();
    for (int iz = 1; iz <= nz; ++iz)
        for (int iy2 = 1; iy2 <= ny2; ++iy2)
            for (int ix = 1; ix <= nx; ++ix) {
                double c = 0., e2 = 0.;
                for (int k = 0; k < f; ++k) {
                    const int iy = (iy2 - 1) * f + 1 + k;
                    c  += h->GetBinContent(ix, iy, iz);
                    e2 += std::pow(h->GetBinError(ix, iy, iz), 2);
                }
                h2->SetBinContent(ix, iy2, iz, c);
                h2->SetBinError(ix, iy2, iz, std::sqrt(e2));
            }
    return h2;
}

// Regroup a flattened response matrix (x = measured, y = truth) after merging
// rebinRL adjacent RL (Y-axis) bins at BOTH det and truth level.
// Flat convention (0-based), matching RooUnfold's flattening of TH3:
//   flat = ix + nx*iy + nx*ny*iz
// Under/overflow rows/columns are carried over unchanged.
static TH2D* rebinFlatResponse(const TH2* H,
                               int nx_m, int ny_m, int nz_m,
                               int nx_t, int ny_t, int nz_t,
                               int rebinRL, const TString& name) {
    const int nm = nx_m * ny_m * nz_m;
    const int nt = nx_t * ny_t * nz_t;
    const int nm2 = nx_m * (ny_m / rebinRL) * nz_m;
    const int nt2 = nx_t * (ny_t / rebinRL) * nz_t;

    // old in-range flat index (0-based) -> new in-range flat index (0-based)
    auto newflat = [=](int flat, int nx, int ny) {
        const int ix = flat % nx;
        const int r  = flat / nx;
        const int iy = r % ny;
        const int iz = r / ny;
        return ix + nx * (iy / rebinRL) + nx * (ny / rebinRL) * iz;
    };

    auto* H2 = new TH2D(name, H->GetTitle(), nm2, 0., 1., nt2, 0., 1.);
    H2->SetDirectory(0);
    H2->Sumw2();

    for (int j = 0; j <= nt + 1; ++j) {
        for (int i = 0; i <= nm + 1; ++i) {
            const double c = H->GetBinContent(i, j);
            const double e = H->GetBinError(i, j);
            if (c == 0. && e == 0.) continue;
            const int i2 = (i == 0) ? 0 : (i == nm + 1 ? nm2 + 1
                                                       : newflat(i - 1, nx_m, ny_m) + 1);
            const int j2 = (j == 0) ? 0 : (j == nt + 1 ? nt2 + 1
                                                       : newflat(j - 1, nx_t, ny_t) + 1);
            H2->SetBinContent(i2, j2, H2->GetBinContent(i2, j2) + c);
            H2->SetBinError(i2, j2, std::sqrt(std::pow(H2->GetBinError(i2, j2), 2) + e * e));
        }
    }
    return H2;
}

// usage
//
// rebinRL: optional rebinning of the RL bins (Y axis of the 3D histograms) by merging
//          adjacent RL bins. rebinRL=1 -> no rebinning (default binning as produced),
//          rebinRL=2 -> RL bins merged in pairs (e.g. 24 -> 12 bins).
//          The rebin is applied consistently to the response matrix (measured, truth
//          and the flattened 2D matrix), the raw data, and the purity histograms.

// FULLSIM / DATA
// signature: unfold(rebinRL, iter, rm_file, data_file, outfile, do_purity)
// root -q "unfold.C(1, 5, \"/global/cfs/cdirs/alice/kdevero/pp_fullsim_enc/26200088/merged.root\", \"/global/cfs/cdirs/alice/kdevero/pp_enc/25886413/merged.root\", \"unfolded_fr.root\", true)"

// no rebinning, all defaults
// root -q unfold.C

// RL bins merged in pairs (output automatically goes to unfolded_rebinRL2.root)
// root -q "unfold.C(2, 5)"

// const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/58411425/response_groomed_forunfolding_merged.root", 
// const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/58416144/AnalysisResultsMerged_groomedbins.root",

void unfold(int rebinRL=1, int iter=5, 
            const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root", 
            const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60320414/AnalysisResultsMerged_groomedbins.root",
            const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded.root",
            bool do_purity=true) { //iter=9
    
    
    // Set ROOT to batch mod
    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true); // covers both 1D and 3D histogram since TH2 inherits from TH1

    // If rebinning is applied, tag the output filename with the rebin factor
    // (inserted before the extension, e.g. unfolded.root -> unfolded_rebinRL2.root)
    TString outname = outfile;
    if (rebinRL > 1) {
        const Ssiz_t dot = outname.Last('.');
        const Ssiz_t slash = outname.Last('/');
        const TString tag = TString::Format("_rebinRL%d", rebinRL);
        if (dot != kNPOS && (slash == kNPOS || dot > slash))
            outname = TString(outname(0, dot)) + tag + TString(outname(dot, outname.Length() - dot));
        else
            outname += tag;
        cout << "rebinRL=" << rebinRL << ": output file will be " << outname << endl;
    }

    // Open inputs
    TFile* f_data = new TFile(data_file);
    TFile* f = new TFile(rm_file);

    // Create output file ONCE, outside the loop
    TFile* fout = new TFile(outname, "RECREATE");

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;


    std::vector<std::string> objects = {"AA", "AB", "BB", "rad"};

    for (const auto& obj : objects) {

        const char* o = obj.c_str();
        std::cout << "=== processing " << obj << " ===" << std::endl;

        // RM
        f->cd();
        RooUnfoldResponse* response = (RooUnfoldResponse*) f->Get(TString::Format("groomed_%s_roounfold_response", o)); //response");

        // DATA
        f_data->cd();
        TH3D* h3_raw = (TH3D*) f_data->Get(TString::Format("%s_sd0.1_raw", o));
        h3_raw = (TH3D*) h3_raw->Clone(TString::Format("%s_raw_corrected", o));

        // ---- optional rebinning of the RL (Y) axis ----
        const int ny_rl = h3_raw->GetNbinsY();
        if (rebinRL > 1 && ny_rl % rebinRL != 0) {
            std::cerr << "Error: " << h3_raw->GetName() << " has " << ny_rl
                      << " RL bins, not divisible by rebinRL=" << rebinRL << std::endl;
            return;
        }

        bool resp_rebinned = false;
        TH3D* h_meas_r  = nullptr;
        TH3D* h_truth_r = nullptr;
        TH2D* Hmat_r    = nullptr;
        if (rebinRL > 1) {
            const TH3* tmpl_m = (const TH3*) response->Hmeasured();
            const TH3* tmpl_t = (const TH3*) response->Htruth();
            if (tmpl_m->GetNbinsY() % rebinRL != 0 || tmpl_t->GetNbinsY() % rebinRL != 0) {
                std::cerr << "Error: response RL bins (" << tmpl_m->GetNbinsY() << "/"
                          << tmpl_t->GetNbinsY() << ") not divisible by rebinRL=" << rebinRL << std::endl;
                return;
            }

            h_meas_r  = rebinY_TH3(tmpl_m, rebinRL, TString::Format("Hmeasured_%s_rebin%d", o, rebinRL));
            h_truth_r = rebinY_TH3(tmpl_t, rebinRL, TString::Format("Htruth_%s_rebin%d", o, rebinRL));
            Hmat_r    = rebinFlatResponse(response->Hresponse(),
                                          tmpl_m->GetNbinsX(), tmpl_m->GetNbinsY(), tmpl_m->GetNbinsZ(),
                                          tmpl_t->GetNbinsX(), tmpl_t->GetNbinsY(), tmpl_t->GetNbinsZ(),
                                          rebinRL, TString::Format("matrix_%s_rebin%d", o, rebinRL));
            if (!h_meas_r || !h_truth_r || !Hmat_r) { std::cerr << "Error: RL rebinning failed" << std::endl; return; }

            response = new RooUnfoldResponse(h_meas_r, h_truth_r, Hmat_r,
                                             TString::Format("groomed_%s_roounfold_response_rebin%d", o, rebinRL),
                                             TString::Format("groomed_%s response (RL rebin %d)", o, rebinRL));
            resp_rebinned = true;
            cout << "RL rebin " << rebinRL << ": " << tmpl_m->GetNbinsY() << " -> " << tmpl_m->GetNbinsY() / rebinRL
                 << " RL bins (flat " << Hmat_r->GetNbinsX() << " x " << Hmat_r->GetNbinsY() << " matrix)" << endl;
        }

        // rebin the raw data on the RL axis
        if (rebinRL > 1) {
            TH3D* h3_raw_r = rebinY_TH3(h3_raw, rebinRL, TString::Format("%s_rebin%d", h3_raw->GetName(), rebinRL));
            if (!h3_raw_r) { std::cerr << "Error: RL rebinning of raw data failed" << std::endl; return; }
            delete h3_raw;
            h3_raw = h3_raw_r;
        }

        // purity correction
        TH3D* purity = nullptr;
        if (do_purity)
        {
            TH3D* h3_reco_matched           = (TH3D*) f->Get(TString::Format("groomed_%s_reco_matched", o));
            TH3D* h3_reco_all = (TH3D*) f->Get(TString::Format("groomed_%s_reco_all", o));
            if (rebinRL > 1) {
                TH3D* h3_reco_matched_r = rebinY_TH3(h3_reco_matched, rebinRL, TString::Format("groomed_%s_reco_matched_rebin%d", o, rebinRL));
                TH3D* h3_reco_all_r     = rebinY_TH3(h3_reco_all,     rebinRL, TString::Format("groomed_%s_reco_all_rebin%d", o, rebinRL));
                if (!h3_reco_matched_r || !h3_reco_all_r) { std::cerr << "Error: RL rebinning of purity inputs failed" << std::endl; return; }
                h3_reco_matched = h3_reco_matched_r;
                h3_reco_all     = h3_reco_all_r;
            }
            h3_reco_matched->Sumw2(1);
            h3_reco_all->Sumw2(1);
            purity = (TH3D*) h3_reco_matched->Clone(TString::Format("purity_%s", o));
            purity->Divide(h3_reco_all); // (reco_matched / reco_all)
            h3_raw->Multiply(purity);
        }
        

        fout->cd();
        for (int i = 1; i <= iter; ++i) {
            // Unfold the 3D histogram
            RooUnfoldBayes unfold(response, h3_raw, i);
            TH3* hunf = (TH3*)unfold.Hunfold(errorTreatment);
            // TH3* hfold = (TH3*)response->ApplyToTruth(hunf, "");

            cout<<"unfolded"<<endl;

            // Clone and name histograms
            TH3* htempUnf = (TH3*) hunf->Clone(TString::Format("Bayesian_Unfolded_%s_iter%d", o, i));            
            // TH3* htempFold = (TH3*)hfold->Clone(TString::Format("Baysian_Foldediter%d", i));

            // Write histograms to output file
            htempUnf->Write();
            // htempFold->Write();

            cout<<"written"<<endl;
        }

        // Make TH2Ds out of response matricies, add to output file
        TH2D* matrix = (TH2D*) response->Hresponse();
        matrix->SetName(TString::Format("matrix_%s", o));
        matrix->Write();

        // write purity hist
        if (do_purity) purity->Write();
        h3_raw->Write(); // note this raw data is now purity-corrected, use it to compare with folded result

        // cleanup of rebinned intermediates (the rebuilt response keeps its own copies
        // of the measured/truth/matrix histograms, so these can go)
        if (resp_rebinned) {
            delete response;
            delete h_meas_r;
            delete h_truth_r;
            delete Hmat_r;
        }

    }

    // Write output file
    fout->Write();
    fout->Close();
}