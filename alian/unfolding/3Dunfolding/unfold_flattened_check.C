#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <TString.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// USAGE
// root -q unfold.C
// root -q "unfold.C(\"rm.root\", \"data.root\", \"unfolded.root\", 5, true, false)"
//
// Strategy: the 3D RooUnfoldResponse gives bogus Mresponse() normalisation, so we
// flatten everything (x fastest, then y, then z -- same as RooUnfold's convention,
// verified against groomed_X_gen) and unfold a plain 1D problem instead.


// ======================= Helpers =======================

// Flattened 1-based bin index (x fastest, then y, then z)
static inline int flat_bin_3d(int ix, int iy, int iz, int nx, int ny) {
    return ix + nx * (iy - 1) + nx * ny * (iz - 1);
}

// TH3 -> TH1D with nx*ny*nz bins (in-range bins only; 3D under/overflow dropped)
static TH1D* flatten(const TH3* h3, const TString& name) {
    const int nx = h3->GetNbinsX(), ny = h3->GetNbinsY(), nz = h3->GetNbinsZ();
    const int n = nx * ny * nz;
    TH1D* h = new TH1D(name, name, n, 0.5, n + 0.5);
    h->SetDirectory(nullptr);
    h->Sumw2();
    for (int iz = 1; iz <= nz; ++iz)
        for (int iy = 1; iy <= ny; ++iy)
            for (int ix = 1; ix <= nx; ++ix) {
                const int k = flat_bin_3d(ix, iy, iz, nx, ny);
                h->SetBinContent(k, h3->GetBinContent(ix, iy, iz));
                h->SetBinError(k,   h3->GetBinError(ix, iy, iz));
            }
    return h;
}

// TH1 (flat) -> TH3D with the binning of 'templ'
static TH3D* unflatten(const TH1* h1, const TH3* templ, const TString& name) {
    TH3D* h = static_cast<TH3D*>(templ->Clone(name));
    h->SetDirectory(nullptr);
    h->Reset();
    const int nx = templ->GetNbinsX(), ny = templ->GetNbinsY(), nz = templ->GetNbinsZ();
    for (int iz = 1; iz <= nz; ++iz)
        for (int iy = 1; iy <= ny; ++iy)
            for (int ix = 1; ix <= nx; ++ix) {
                const int k = flat_bin_3d(ix, iy, iz, nx, ny);
                h->SetBinContent(ix, iy, iz, h1->GetBinContent(k));
                h->SetBinError(ix, iy, iz,   h1->GetBinError(k));
            }
    return h;
}

// Copy the in-range part of the flattened 3D response into a TH2D with clean 1D axes
static TH2D* build_flat_response(const TH2* H, int n_m, int n_t, const TString& name) {
    TH2D* h = new TH2D(name, name, n_m, 0.5, n_m + 0.5, n_t, 0.5, n_t + 0.5);
    h->SetDirectory(nullptr);
    h->Sumw2();
    for (int j = 1; j <= n_t; ++j)
        for (int i = 1; i <= n_m; ++i) {
            const double c = H->GetBinContent(i, j);
            if (c == 0.) continue;
            h->SetBinContent(i, j, c);
            h->SetBinError(i, j, H->GetBinError(i, j));
        }
    return h;
}

// Column sums of the normalised response = efficiencies, must all be <= 1
static bool check_response_1d(RooUnfoldResponse& resp, const char* tag) {
    const TMatrixD& R = resp.Mresponse();
    double max_colsum = 0., max_el = 0.;
    int j_max = -1, n_bad = 0;
    for (int j = 0; j < R.GetNcols(); ++j) {
        double s = 0.;
        for (int i = 0; i < R.GetNrows(); ++i) {
            s += R(i, j);
            max_el = std::max(max_el, R(i, j));
        }
        if (s > 1. + 1e-6) ++n_bad;
        if (s > max_colsum) { max_colsum = s; j_max = j; }
    }
    std::cout << "[" << tag << "] 1D response check: max column sum = " << max_colsum
              << " (truth bin " << j_max << "), max element = " << max_el
              << ", columns with sum > 1: " << n_bad << " / " << R.GetNcols() << std::endl;
    return n_bad == 0;
}


// ======================= Main =======================

void unfold_flattened_check(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/58919321/response_groomed_forunfolding_merged.root",
            const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60017705/AnalysisResultsMerged_groomedbins.root",
            const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded_flattened_check.root",
            int iter=5, bool do_purity=true, bool do_mc_closure=true) { //TODO: turn mc_closure back to false after done

    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true);
    TH1::AddDirectory(false);   // keep ownership of all histograms with us

    TStopwatch total_timer;
    total_timer.Start();

    // Open inputs
    TFile* f_data = TFile::Open(data_file);
    TFile* f      = TFile::Open(rm_file);
    if (!f || !f_data || f->IsZombie() || f_data->IsZombie()) {
        std::cerr << "Error: cannot open input files" << std::endl;
        return;
    }

    // Output file
    TFile* fout = new TFile(outfile, "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "Error: cannot create output file " << outfile << std::endl;
        return;
    }

    // Error treatment for unfolding (kCovariance is slow for 4500 bins; kErrors is faster)
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    std::vector<std::string> objects = {"AA", "AB", "BB", "rad"};

    for (const auto& obj : objects) {

        const char* o = obj.c_str();
        TStopwatch obj_timer;
        obj_timer.Start();
        std::cout << "\n=== processing " << obj << " ===" << std::endl;

        // ---------------- 3D RM ----------------
        RooUnfoldResponse* response3d = (RooUnfoldResponse*) f->Get(TString::Format("groomed_%s_roounfold_response", o));
        if (!response3d) { std::cerr << "Error: response for " << obj << " not found, skipping" << std::endl; continue; }

        const TH3* h3_truth_all = dynamic_cast<const TH3*>(response3d->Htruth());
        const TH3* h3_meas_resp = dynamic_cast<const TH3*>(response3d->Hmeasured());
        const TH2* h2_resp      = response3d->Hresponse();
        if (!h3_truth_all || !h3_meas_resp || !h2_resp) {
            std::cerr << "Error: response for " << obj << " not built from TH3s, skipping" << std::endl;
            continue;
        }

        const int n_m = h3_meas_resp->GetNbinsX() * h3_meas_resp->GetNbinsY() * h3_meas_resp->GetNbinsZ();
        const int n_t = h3_truth_all->GetNbinsX() * h3_truth_all->GetNbinsY() * h3_truth_all->GetNbinsZ();
        if (h2_resp->GetNbinsX() != n_m || h2_resp->GetNbinsY() != n_t) {
            std::cerr << "Error: Hresponse is " << h2_resp->GetNbinsX() << " x " << h2_resp->GetNbinsY()
                      << " but TH3s flatten to " << n_m << " x " << n_t << ", skipping" << std::endl;
            continue;
        }

        // ---------------- Build flattened 1D response ----------------
        TH2D* h_resp_flat  = build_flat_response(h2_resp, n_m, n_t, TString::Format("resp_flat_%s", o));
        TH1D* h_truth_flat = flatten(h3_truth_all, TString::Format("truth_flat_%s", o));   // all truth incl. misses
        // Measured = row sums of the matched response -> no fakes inside RooUnfold
        // (fakes are removed beforehand via the purity correction on data)
        TH1D* h_meas_flat  = h_resp_flat->ProjectionX(TString::Format("meas_flat_%s", o), 1, n_t);
        h_meas_flat->SetDirectory(nullptr);

        RooUnfoldResponse resp1d(h_meas_flat, h_truth_flat, h_resp_flat,
                                 TString::Format("response1D_%s", o), TString::Format("response1D_%s", o));

        std::cout << "Hresponse integral = " << h_resp_flat->Integral()
                  << ", truth (all) integral = " << h_truth_flat->Integral()
                  << ", integrated efficiency = " << h_resp_flat->Integral() / h_truth_flat->Integral() << std::endl;

        if (!check_response_1d(resp1d, o)) {
            std::cerr << "Error: flattened response for " << obj << " still has column sums > 1, skipping" << std::endl;
            delete h_resp_flat; delete h_truth_flat; delete h_meas_flat;
            continue;
        }

        // ---------------- DATA ----------------
        TH3D* h3_raw_in = (TH3D*) f_data->Get(TString::Format("%s_sd0.1_raw", o));
        if (!h3_raw_in) { std::cerr << "Error: raw histogram for " << obj << " not found, skipping" << std::endl; continue; }
        TH3D* h3_raw = (TH3D*) h3_raw_in->Clone(TString::Format("%s_raw_corrected", o));
        h3_raw->SetDirectory(nullptr);

        // ---------------- Purity correction ----------------
        TH3D* purity = nullptr;
        if (do_purity) {
            TH3D* h3_reco           = (TH3D*) f->Get(TString::Format("groomed_%s_reco", o));
            TH3D* h3_reco_unmatched = (TH3D*) f->Get(TString::Format("groomed_%s_reco_unmatched", o));
            if (!h3_reco || !h3_reco_unmatched) {
                std::cerr << "Error: purity inputs for " << obj << " not found, skipping" << std::endl;
                continue;
            }
            h3_reco->Sumw2(1);
            h3_reco_unmatched->Sumw2(1);
            purity = (TH3D*) h3_reco->Clone(TString::Format("purity_%s", o));
            purity->SetDirectory(nullptr);
            purity->Divide(h3_reco, h3_reco_unmatched, 1., 1., "B");   // matched reco / all reco
            h3_raw->Multiply(purity);
        }

        TH1D* h_raw_flat = flatten(h3_raw, TString::Format("raw_flat_%s", o));
        std::cout << "raw" << (do_purity ? "*purity" : "") << " integral = " << h_raw_flat->Integral()
                  << ", MC matched reco integral = " << h_meas_flat->Integral() << std::endl;

        // ---------------- Unfold data ----------------
        fout->cd();
        for (int i = 1; i <= iter; ++i) {
            TStopwatch it_timer;
            it_timer.Start();

            RooUnfoldBayes unfold(&resp1d, h_raw_flat, i);
            TH1* hunf_flat = (TH1*) unfold.Hunfold(errorTreatment);
            if (!hunf_flat) { std::cerr << "Error: unfolding failed for " << obj << " iter " << i << std::endl; continue; }

            TH3D* hunf3 = unflatten(hunf_flat, h3_truth_all, TString::Format("Bayesian_Unfolded_%s_iter%d", o, i));
            fout->cd();
            hunf3->Write();

            std::cout << "  iter " << i << ": unfolded integral = " << hunf3->Integral()
                      << "  (" << it_timer.RealTime() << " s)" << std::endl;
            delete hunf3;
        }

        // ---------------- Optional: trivial MC closure ----------------
        // Unfold MC matched reco (= h_meas_flat); should reproduce MC all-truth (groomed_X_gen_unmatched)
        if (do_mc_closure) {
            for (int i = 1; i <= iter; ++i) {
                RooUnfoldBayes unfold_mc(&resp1d, h_meas_flat, i);
                TH1* hmc_flat = (TH1*) unfold_mc.Hunfold(errorTreatment);
                if (!hmc_flat) continue;
                TH3D* hmc3 = unflatten(hmc_flat, h3_truth_all, TString::Format("MCClosure_Unfolded_%s_iter%d", o, i));
                fout->cd();
                hmc3->Write();
                std::cout << "  MC closure iter " << i << ": unfolded = " << hmc3->Integral()
                          << ", truth (all) = " << h_truth_flat->Integral() << std::endl;
                delete hmc3;
            }
        }

        // ---------------- Write remaining objects ----------------
        fout->cd();
        TH2D* matrix = (TH2D*) h2_resp->Clone(TString::Format("matrix_%s", o));
        matrix->Write();
        if (do_purity) purity->Write();
        h3_raw->Write();   // purity-corrected raw = exactly what was unfolded; compare refolded result to this
        fout->Flush();     // save progress in case a later object crashes

        std::cout << "=== " << obj << " done in " << obj_timer.RealTime() / 60. << " min ===" << std::endl;

        // ---------------- Cleanup ----------------
        delete matrix;
        delete h3_raw;
        delete purity;
        delete h_raw_flat;
        delete h_meas_flat;
        delete h_truth_flat;
        delete h_resp_flat;
    }

    fout->Close();
    std::cout << "\nWrote output to " << outfile
              << " (total " << total_timer.RealTime() / 60. << " min)" << std::endl;
}