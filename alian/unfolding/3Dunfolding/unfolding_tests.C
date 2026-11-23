#include <iostream>
#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TRandom3.h>
#include <TVectorD.h>
#include <TArrayD.h>
#include <TMatrixD.h>
#include <TSystem.h>
#include <TString.h>
#include <TObjString.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// Inspired from: https://github.com/ezradlesser/pyjetty/blob/09e41284e6bd7f289e457875d9e15224eaa34dc2/pyjetty/alice_analysis/analysis/user/substructure/roounfold_obs.py#L1233

// USAGE
// don't forget to run roounfold_preload_on first! -- prob not needed anymore
// root -q unfolding_tests.C > unfolding_test_output.txt
// OBJ="AB" // root -l -b -q "unfolding_tests.C(\"${OBJ}\")"
// RL bins merged in pairs -> unfolded_tests_AA_rebinRL2.root
// root -l -b -q "unfolding_tests.C(\"AA\", 2)"
//
// rebinRL: optional RL (Y-axis) rebinning by merging adjacent bins, mirroring the
//          rebinRL option in unfold.C. rebinRL=1 -> no rebinning (default binning
//          as produced), rebinRL=2 -> RL bins merged in pairs (e.g. 24 -> 12 bins).
//          Applied consistently to ALL inputs: the response (measured, truth and
//          flattened matrix), the raw data, and the purity/efficiency inputs. The
//          output file name is tagged _rebinRL<n>, and unfolded_file is tagged the
//          same way (unfolded.root -> unfolded_rebinRL2.root): the unfolded results
//          are expected from unfold.C run with the SAME rebinRL and are NOT rebinned
//          again here.
// OBJ="AB" // root -l -b -q "unfolding_tests.C(\"${OBJ}\")"




// Flattened 1-based bin index, matching RooUnfold's convention for 3D histograms:
//   flat = (ix-1) + nx*(iy-1) + nx*ny*(iz-1)   (0-based)
// and the response TH2 bin = flat + 1.
static inline int flat_bin_3d(int ix, int iy, int iz, int nx, int ny) {
    return ix + nx * (iy - 1) + nx * ny * (iz - 1);
}

//------------------------------------------------------------------------------------------------
// RL rebinning helpers (copied from unfold.C)
//------------------------------------------------------------------------------------------------

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

// Insert "_rebinRL<n>" before the extension of a file name (no-op if rebinRL <= 1)
static TString tagRebinFile(const TString& name, int rebinRL) {
    if (rebinRL <= 1) return name;
    const Ssiz_t dot = name.Last('.');
    const Ssiz_t slash = name.Last('/');
    const TString tag = TString::Format("_rebinRL%d", rebinRL);
    if (dot != kNPOS && (slash == kNPOS || dot > slash))
        return TString(name(0, dot)) + tag + TString(name(dot, name.Length() - dot));
    return name + tag;
}



//----// h_truth_train_all is the full truth-level distribution from the MC sample used to fill the migration matrix
//----// h_mig_raw is the flattened (N_meas x N_true) response, e.g. RooUnfoldResponse::Hresponse()
static TH3D* th3_manual_refold_from_raw_migration(TH2* h_mig_raw, TH3* h_meas_template, TH3* h_true_template,
                                                  TH3* h_unfolded_truth, TString name, bool include_reco_overflow=true) {
    int nx_meas = h_meas_template->GetNbinsX();
    int ny_meas = h_meas_template->GetNbinsY();
    int nz_meas = h_meas_template->GetNbinsZ();
    int nx_true = h_true_template->GetNbinsX();
    int ny_true = h_true_template->GetNbinsY();
    int nz_true = h_true_template->GetNbinsZ();

    int n_meas_flat = nx_meas * ny_meas * nz_meas;
    int n_true_flat = nx_true * ny_true * nz_true;

    // Sanity check: response dimensions must match the flattened templates
    if (h_mig_raw->GetNbinsX() != n_meas_flat || h_mig_raw->GetNbinsY() != n_true_flat) {
        Error("th3_manual_refold_from_raw_migration",
              "Response matrix is %d x %d but templates flatten to %d x %d",
              h_mig_raw->GetNbinsX(), h_mig_raw->GetNbinsY(), n_meas_flat, n_true_flat);
        return nullptr;
    }

    // Column sums = matched truth per (flattened) truth bin -> P(reco i | true j, matched)
    int i_m_lo = include_reco_overflow ? 0 : 1;
    int i_m_hi = include_reco_overflow ? n_meas_flat + 1 : n_meas_flat;

    std::vector<double> inv_norm(n_true_flat + 1, 0.0);
    for (int i_t = 1; i_t <= n_true_flat; ++i_t) {
        double norm = 0.0;
        for (int i_m = i_m_lo; i_m <= i_m_hi; ++i_m)
            norm += h_mig_raw->GetBinContent(i_m, i_t);
        if (norm > 0.0) inv_norm[i_t] = 1.0 / norm;
    }

    // Cache unfolded truth in flattened order
    std::vector<double> u(n_true_flat + 1, 0.0);
    std::vector<double> u_err(n_true_flat + 1, 0.0);
    for (int iz_t = 1; iz_t <= nz_true; ++iz_t)
        for (int iy_t = 1; iy_t <= ny_true; ++iy_t)
            for (int ix_t = 1; ix_t <= nx_true; ++ix_t) {
                int i_t = flat_bin_3d(ix_t, iy_t, iz_t, nx_true, ny_true);
                u[i_t]     = h_unfolded_truth->GetBinContent(ix_t, iy_t, iz_t);
                u_err[i_t] = h_unfolded_truth->GetBinError(ix_t, iy_t, iz_t);
            }

    TH3D* h_refold = static_cast<TH3D*>(h_meas_template->Clone(name));
    h_refold->SetDirectory(nullptr);
    h_refold->Reset();

    for (int iz_m = 1; iz_m <= nz_meas; ++iz_m) {
        for (int iy_m = 1; iy_m <= ny_meas; ++iy_m) {
            for (int ix_m = 1; ix_m <= nx_meas; ++ix_m) {

                int i_m = flat_bin_3d(ix_m, iy_m, iz_m, nx_meas, ny_meas);
                double pred = 0.0;
                double pred_error_sq = 0.0;

                for (int i_t = 1; i_t <= n_true_flat; ++i_t) {
                    if (inv_norm[i_t] == 0.0) continue;
                    double mij = h_mig_raw->GetBinContent(i_m, i_t);
                    if (mij <= 0.0) continue;

                    double transfer_coeff = mij * inv_norm[i_t];
                    pred          += transfer_coeff * u[i_t];
                    pred_error_sq += std::pow(transfer_coeff * u_err[i_t], 2.);
                }

                h_refold->SetBinContent(ix_m, iy_m, iz_m, pred);
                h_refold->SetBinError(ix_m, iy_m, iz_m, std::sqrt(pred_error_sq));
            }
        }
    }

    return h_refold;
}


static TH3D* th3_refold(RooUnfoldResponse* resp, TH3* h_meas_template, TH3* h_unfolded_truth, TString name) {
    const TH2*      H  = resp->Hresponse();
    const TVectorD& vt = resp->Vtruth();      // all truth incl. misses, flattened, 0-based

    const int nx_m = h_meas_template->GetNbinsX(), ny_m = h_meas_template->GetNbinsY(), nz_m = h_meas_template->GetNbinsZ();
    const int nx_t = h_unfolded_truth->GetNbinsX(), ny_t = h_unfolded_truth->GetNbinsY(), nz_t = h_unfolded_truth->GetNbinsZ();
    const int n_m = nx_m * ny_m * nz_m, n_t = nx_t * ny_t * nz_t;

    if (H->GetNbinsX() != n_m || H->GetNbinsY() != n_t || vt.GetNrows() != n_t) {
        Error("th3_refold", "dimension mismatch");
        return nullptr;
    }

    std::vector<double> u(n_t, 0.), u_err(n_t, 0.), inv_vt(n_t, 0.);
    for (int iz = 1; iz <= nz_t; ++iz)
        for (int iy = 1; iy <= ny_t; ++iy)
            for (int ix = 1; ix <= nx_t; ++ix) {
                int j = flat_bin_3d(ix, iy, iz, nx_t, ny_t) - 1;
                u[j]     = h_unfolded_truth->GetBinContent(ix, iy, iz);
                u_err[j] = h_unfolded_truth->GetBinError(ix, iy, iz);
                if (vt[j] > 0.) inv_vt[j] = 1. / vt[j];
            }

    TH3D* h = static_cast<TH3D*>(h_meas_template->Clone(name));
    h->SetDirectory(nullptr);
    h->Reset();

    for (int iz = 1; iz <= nz_m; ++iz)
        for (int iy = 1; iy <= ny_m; ++iy)
            for (int ix = 1; ix <= nx_m; ++ix) {
                int i = flat_bin_3d(ix, iy, iz, nx_m, ny_m);   // 1-based for TH2 access
                double pred = 0., err2 = 0.;
                for (int j = 0; j < n_t; ++j) {
                    if (inv_vt[j] == 0.) continue;
                    double m = H->GetBinContent(i, j + 1);
                    if (m <= 0.) continue;
                    double r = m * inv_vt[j];                    // P(reco i | true j), incl. efficiency
                    pred += r * u[j];
                    err2 += r * r * u_err[j] * u_err[j];
                }
                h->SetBinContent(ix, iy, iz, pred);
                h->SetBinError(ix, iy, iz, std::sqrt(err2));
            }
    return h;
}



// REFOLDING TEST
// Feed in unfolded data distribution, "refold" (multiply) if with the RM
// To be compared: refolded distribuion to original raw (detector) data
std::vector<TH3*> refolding_test(std::string obj, TFile * f_unf, RooUnfoldResponse* resp, TH3D * h3_raw, int iter, TH3D * purity, bool do_purity, std::string addname="") {

    std::vector<TH3*> out_hists;

    for (int i = 1; i <= iter; ++i) {

        cout << "------------------------------------------" << endl;
        cout << "iter " << i << endl;


        TString unf_name = TString::Format("Bayesian_Unfolded_%s_iter%d", obj.c_str(), i);
        TH3* h3_unf_in = (TH3*) f_unf->Get(unf_name);
        if (!h3_unf_in) { cout << "Skipping " << unf_name << " (not found)" << endl; continue; }

        // Truth level that M acts on
        TH3* h3_unfolded_truth = (TH3*) h3_unf_in->Clone(TString::Format("%s", addname.c_str()) + unf_name); // + "_Clone"); //TString::Format("h1_jetpt_groomed_unfolded%d", i));
        h3_unfolded_truth->SetDirectory(0);

        // Fold --> manual function b/c I can't get ApplyToTruth() to work :(
        // TH1D* h1_refold = th1_manual_refold_from_raw_migration(RM_2D, h1_raw, h1_truth, h1_truth, TString::Format("%sBaysian_Refolded1Diter%d", addname.c_str(), i));
        // TH3D* h3_refold = th3_manual_refold(resp, h3_raw, h3_unfolded_truth, TString::Format("%sBayesian_Refolded_%s_iter%d", addname.c_str(), obj.c_str(), i));
        TH3D* h3_refold = th3_refold(resp, h3_raw, h3_unfolded_truth, TString::Format("%sBayesian_Refolded_%s_iter%d", addname.c_str(), obj.c_str(), i));
        h3_refold->SetTitle(TString::Format("Refolded, iter %d", i));

        // (a) refold integral BEFORE any purity division
        TH3D* h3_raw_corr = (TH3D*) h3_raw->Clone("tmp_raw_corr");
        h3_raw_corr->Multiply(purity);
        cout << "refold (no /purity) = " << h3_refold->Integral()
            << ", raw*purity = " << h3_raw_corr->Integral() << endl;
        delete h3_raw_corr;

        // (b) column sums of the normalised response = efficiencies, must be <= 1
        if (i == 1) {
            const TMatrixD& R = resp->Mresponse();
            double max_colsum = 0., max_el = 0.; int j_max = -1;
            for (int j = 0; j < R.GetNcols(); ++j) {
                double s = 0.;
                for (int k = 0; k < R.GetNrows(); ++k) { s += R(k, j); max_el = std::max(max_el, R(k, j)); }
                if (s > max_colsum) { max_colsum = s; j_max = j; }
            }
            cout << "max column sum (efficiency) = " << max_colsum << " at truth bin " << j_max
                << ", max element = " << max_el << endl;
            cout << "Hresponse integral = " << resp->Hresponse()->Integral()
                << ", Htruth integral = "   << resp->Htruth()->Integral()
                << ", Hmeasured integral = " << resp->Hmeasured()->Integral() << endl;
        }

        // Undo purity: add matching fakes back so it is comparable to raw
        if (do_purity) h3_refold->Divide(purity); //???

        cout << "unfolded = " << h3_unf_in->Integral()
             << ", refolded = " << h3_refold->Integral()
             << ", raw = " << h3_raw->Integral() << endl;
        cout << endl;

        out_hists.insert(out_hists.end(), {h3_unfolded_truth, h3_refold});
    }

    return out_hists;

}

//================================================================================================
// CLOSURE TEST CONFIGURATION / HELPERS
//================================================================================================

// Error treatment for statistical/shape/prior closure (the trivial test uses kNoError, since
// its errors are meaningless and covariance propagation is slow for 4320 bins)
static RooUnfold::ErrorTreatment kClosureErr = RooUnfold::kCovariance;
static TRandom3 gClosureRng(0);

TH3D* cloneTH3(const TH1* h, const TString& name) {
    TH3D* c = dynamic_cast<TH3D*>(h->Clone(name));
    if (!c) { Error("cloneTH3", "%s is not a TH3D", h->GetName()); return nullptr; }
    c->SetDirectory(0);
    if (c->GetSumw2N() == 0) c->Sumw2();
    return c;
}

bool sameBinning3D(const TH3* a, const TH3* b, const char* what) {
    auto same = [](const TAxis* x, const TAxis* y) {
        if (x->GetNbins() != y->GetNbins()) return false;
        for (int i = 1; i <= x->GetNbins() + 1; ++i) {
            const double ex = x->GetBinLowEdge(i), ey = y->GetBinLowEdge(i);
            if (std::fabs(ex - ey) > 1e-9 * std::max(1., std::fabs(ex))) return false;
        }
        return true;
    };
    const bool ok = same(a->GetXaxis(), b->GetXaxis()) && same(a->GetYaxis(), b->GetYaxis())
                 && same(a->GetZaxis(), b->GetZaxis());
    if (!ok) printf("ERROR: binning mismatch: %s (%s vs %s)\n", what, a->GetName(), b->GetName());
    return ok;
}

// Copy of the flattened response with all under/overflow rows and columns zeroed
// (det or truth outside the RM range). Only the in-range block remains.
TH2* inRangeResponse(const TH2* H, const TString& name) {
    TH2* h = (TH2*) H->Clone(name);
    h->SetDirectory(0);
    const int nm = h->GetNbinsX(), nt = h->GetNbinsY();
    for (int j = 0; j <= nt + 1; ++j)
        for (int i : {0, nm + 1}) { h->SetBinContent(i, j, 0); h->SetBinError(i, j, 0); }
    for (int i = 0; i <= nm + 1; ++i)
        for (int j : {0, nt + 1}) { h->SetBinContent(i, j, 0); h->SetBinError(i, j, 0); }
    return h;
}

// Det-level TH3 = row sums of the in-range flattened response (matched pairs with truth in range)
TH3D* detFromResponse(const TH2* H, const TH3* measTmpl, const TString& name) {
    TH3D* h = (TH3D*) measTmpl->Clone(name);
    h->SetDirectory(0);
    h->Reset();
    const int nx = h->GetNbinsX(), ny = h->GetNbinsY(), nz = h->GetNbinsZ();
    const int nt = H->GetNbinsY();
    for (int iz = 1; iz <= nz; ++iz)
        for (int iy = 1; iy <= ny; ++iy)
            for (int ix = 1; ix <= nx; ++ix) {
                const int i = flat_bin_3d(ix, iy, iz, nx, ny);
                double s = 0., e2 = 0.;
                for (int j = 1; j <= nt; ++j) {
                    s  += H->GetBinContent(i, j);
                    e2 += std::pow(H->GetBinError(i, j), 2);
                }
                h->SetBinContent(ix, iy, iz, s);
                h->SetBinError(ix, iy, iz, std::sqrt(e2));
            }
    return h;
}

// Response from an in-range flattened RM, its det row sums, and a truth TH3 that includes misses.
// fakes = det - row sums = 0; misses = truth - column sums (pair inefficiency + det out of range)
RooUnfoldResponse* makeResponse3D(const TH2* H, const TH3* hDet, const TH3* hTruth, const TString& name) {
    return new RooUnfoldResponse(hDet, hTruth, H, name, name);
}

// How much of the det-level matched sample is NOT covered by the in-range response
void diagnoseResponse(RooUnfoldResponse* r, const TH3* hMeas, const TH3* hDetInRM, const TH3* hTruth) {
    const double meas = hMeas->Integral(1, hMeas->GetNbinsX(), 1, hMeas->GetNbinsY(), 1, hMeas->GetNbinsZ());
    const double inRM = hDetInRM->Integral();
    const double tru  = hTruth->Integral(1, hTruth->GetNbinsX(), 1, hTruth->GetNbinsY(), 1, hTruth->GetNbinsZ());
    printf("  Response diagnostics:\n");
    printf("    Hmeasured (in range)             = %g\n", meas);
    printf("    in-range response (det row sums) = %g  -> %.2f%% of det pairs have truth outside RM range\n",
           inRM, meas > 0 ? 100. * (1. - inRM / meas) : 0.);
    printf("    Htruth (in range, incl. misses)  = %g  -> pair efficiency inside RM = %.4f\n",
           tru, tru > 0 ? inRM / tru : 0.);
    printf("    explicit fakes (Vfakes sum)      = %g\n", r->Vfakes().Sum());
}

void wDiagonality(const TH2* H, const TH3* measTmpl, const TH3* truthTmpl) {
    const int nxm = measTmpl->GetNbinsX(),  nym = measTmpl->GetNbinsY(),  nzm = measTmpl->GetNbinsZ();
    const int nxt = truthTmpl->GetNbinsX(), nyt = truthTmpl->GetNbinsY(), nzt = truthTmpl->GetNbinsZ();
    std::vector<double> diag(nzt + 1, 0.), tot(nzt + 1, 0.);
    for (int j = 1; j <= nxt * nyt * nzt; ++j) {
        const int izt = (j - 1) / (nxt * nyt) + 1;
        for (int i = 1; i <= nxm * nym * nzm; ++i) {
            const double c = H->GetBinContent(i, j);
            if (c <= 0) continue;
            const int izm = (i - 1) / (nxm * nym) + 1;
            tot[izt] += c;
            if (izm == izt) diag[izt] += c;   // assumes same w binning at det and truth level
        }
    }
    for (int iz = 1; iz <= nzt; ++iz)
        printf("  w bin %2d [%g, %g]: diagonal fraction = %.3f\n", iz,
               truthTmpl->GetZaxis()->GetBinLowEdge(iz), truthTmpl->GetZaxis()->GetBinUpEdge(iz),
               tot[iz] > 0 ? diag[iz] / tot[iz] : 0.);
}

//------------------------------------------------------------------------------------------------
// Data-like smearing in 3D
//------------------------------------------------------------------------------------------------
// Relative stat. errors of the data, indexed by GLOBAL bin number (h->GetBin(ix,iy,iz))
std::vector<double> getMeasuredErrors3D(const TH3* hData) {
    std::vector<double> rel(hData->GetNcells(), 0.);
    int nzero = 0;
    for (int iz = 1; iz <= hData->GetNbinsZ(); ++iz)
        for (int iy = 1; iy <= hData->GetNbinsY(); ++iy)
            for (int ix = 1; ix <= hData->GetNbinsX(); ++ix) {
                const int g = hData->GetBin(ix, iy, iz);
                const double c = hData->GetBinContent(g), e = hData->GetBinError(g);
                if (c != 0) rel[g] = e / c;
                else ++nzero;
            }
    printf("  getMeasuredErrors3D: %d of %d in-range data bins are empty (rel. error set to 0)\n",
           nzero, hData->GetNbinsX() * hData->GetNbinsY() * hData->GetNbinsZ());
    return rel;
}

// content -> Gaus(content, content*relErr), error -> content*relErr (pyjetty smearSpectrum)
TH3D* smearSpectrum3D(const TH3* h_in, const std::vector<double>& relErr, const TString& name) {
    if ((int) relErr.size() != h_in->GetNcells()) {
        Error("smearSpectrum3D", "relErr size %zu != ncells %d (binning mismatch?)", relErr.size(), h_in->GetNcells());
        return nullptr;
    }
    TH3D* h = cloneTH3(h_in, name);
    for (int g = 0; g < h->GetNcells(); ++g) h->SetBinError(g, 0);

    int n_nodata = 0, n_neg = 0;
    for (int iz = 1; iz <= h->GetNbinsZ(); ++iz)
        for (int iy = 1; iy <= h->GetNbinsY(); ++iy)
            for (int ix = 1; ix <= h->GetNbinsX(); ++ix) {
                const int g = h->GetBin(ix, iy, iz);
                const double content = h->GetBinContent(g);
                if (content != 0 && relErr[g] == 0) ++n_nodata;
                const double errNew = std::abs(relErr[g] * content);
                const double cNew   = content + gClosureRng.Gaus(0, errNew);
                if (cNew < 0) ++n_neg;
                h->SetBinContent(g, cNew);
                h->SetBinError(g, errNew);
            }
    printf("  smearSpectrum3D(%s): %d MC bins with no data (left unsmeared), %d bins negative after smearing\n",
           name.Data(), n_nodata, n_neg);
    return h;
}

//------------------------------------------------------------------------------------------------
// Prior weights: pt_true^p_pt * RL_true^p_rl * w_true^p_w
//------------------------------------------------------------------------------------------------
struct PriorVariation {
    double p_pt = 0., p_rl = 0., p_w = 0.;
    TString label() const {
        auto f = [](const char* n, double p) {
            TString s = TString::Format("%s%g", n, p);
            s.ReplaceAll(".", "").ReplaceAll("-", "m");
            return s;
        };
        TString l;
        if (p_pt != 0) l += f("pt", p_pt);
        if (p_rl != 0) l += (l.IsNull() ? "" : "_") + f("rl", p_rl);
        if (p_w  != 0) l += (l.IsNull() ? "" : "_") + f("w",  p_w);
        return l.IsNull() ? TString("nominal") : l;
    }
};

static double powInt(double a, double b, double k) {   // integral of x^k on [a,b]
    return (std::abs(k + 1) < 1e-9) ? std::log(b / a)
                                    : (std::pow(b, k + 1) - std::pow(a, k + 1)) / (k + 1);
}

static double binAvgWeight(double a, double b, double n, double p) {   // <x^p> for x^-n
    return powInt(a, b, p - n) / powInt(a, b, -n);
}

// Local power-law index per bin from the log-log slope of the density between neighbors
std::vector<double> localPowerLawIndex(const TH1* h) {
    const TAxis* ax = h->GetXaxis();
    const int n = h->GetNbinsX();
    std::vector<double> x(n + 2, 0.), d(n + 2, 0.), idx(n + 2, 0.);
    for (int j = 1; j <= n; ++j) {
        const double a = ax->GetBinLowEdge(j), b = ax->GetBinUpEdge(j);
        x[j] = (a > 0) ? std::sqrt(a * b) : ax->GetBinCenter(j);
        d[j] = h->GetBinContent(j) / (b - a);
    }
    for (int j = 1; j <= n; ++j) {
        const int lo = std::max(1, j - 1), hi = std::min(n, j + 1);
        if (lo == hi || d[lo] <= 0 || d[hi] <= 0 || x[lo] <= 0) continue;
        idx[j] = -std::log(d[hi] / d[lo]) / std::log(x[hi] / x[lo]);
    }
    return idx;
}

// Weight per bin of one axis (indexed by ROOT bin number)
//   useBinCenter      : x_center^p
//   nLoc given        : bin average assuming a local power law (used for pT)
//   otherwise         : geometric bin center^p (log-binned RL and w)
std::vector<double> axisPriorWeights(const TAxis* ax, double p, bool useBinCenter,
                                     const std::vector<double>* nLoc) {
    std::vector<double> w(ax->GetNbins() + 2, 1.);
    if (std::abs(p) <= 1e-5) return w;
    for (int i = 1; i <= ax->GetNbins(); ++i) {
        const double a = ax->GetBinLowEdge(i), b = ax->GetBinUpEdge(i);
        if (useBinCenter || a <= 0) w[i] = std::pow(ax->GetBinCenter(i), p);
        else if (nLoc)              w[i] = binAvgWeight(a, b, (*nLoc)[i], p);
        else                        w[i] = std::pow(std::sqrt(a * b), p);
    }
    return w;
}

struct Scaled3D {
    TH2*  H      = nullptr;   // scaled in-range flattened RM (large: delete as soon as possible)
    TH3D* hDet   = nullptr;   // det row sums of scaled RM
    TH3D* hTruth = nullptr;   // scaled truth incl. misses
};

// Scale every truth column of the in-range RM, and the truth (incl. misses), by the prior weight
Scaled3D scaleResponse3D(const TH2* H_in, const TH3* hTruth_in, const TH3* measTmpl,
                         const PriorVariation& pv, bool useBinCenter, const TString& name) {
    Scaled3D s;
    const int nx = hTruth_in->GetNbinsX(), ny = hTruth_in->GetNbinsY(), nz = hTruth_in->GetNbinsZ();
    const int nm = H_in->GetNbinsX();

    std::vector<double> nLoc;
    if (!useBinCenter && std::abs(pv.p_pt) > 1e-5) {
        TH1D* hp = hTruth_in->ProjectionX(name + "_ptproj_tmp", 1, ny, 1, nz);
        nLoc = localPowerLawIndex(hp);
        delete hp;
    }
    const auto wx = axisPriorWeights(hTruth_in->GetXaxis(), pv.p_pt, useBinCenter, nLoc.empty() ? nullptr : &nLoc);
    const auto wy = axisPriorWeights(hTruth_in->GetYaxis(), pv.p_rl, useBinCenter, nullptr);
    const auto wz = axisPriorWeights(hTruth_in->GetZaxis(), pv.p_w,  useBinCenter, nullptr);

    printf("  scaleResponse3D(%s):\n", name.Data());
    for (int ix = 1; ix <= nx; ++ix)
        printf("    pT [%g, %g]  n_local = %6.2f  weight = %g\n",
               hTruth_in->GetXaxis()->GetBinLowEdge(ix), hTruth_in->GetXaxis()->GetBinUpEdge(ix),
               nLoc.empty() ? 0. : nLoc[ix], wx[ix]);
    printf("    RL weights: %g ... %g,  w weights: %g ... %g\n", wy[1], wy[ny], wz[1], wz[nz]);

    s.hTruth = cloneTH3(hTruth_in, name + "_truth");
    s.H = (TH2*) H_in->Clone(name + "_H");
    s.H->SetDirectory(0);

    for (int iz = 1; iz <= nz; ++iz)
        for (int iy = 1; iy <= ny; ++iy)
            for (int ix = 1; ix <= nx; ++ix) {
                const double f = wx[ix] * wy[iy] * wz[iz];
                if (f == 1.) continue;
                s.hTruth->SetBinContent(ix, iy, iz, s.hTruth->GetBinContent(ix, iy, iz) * f);
                s.hTruth->SetBinError  (ix, iy, iz, s.hTruth->GetBinError  (ix, iy, iz) * f);
                const int j = flat_bin_3d(ix, iy, iz, nx, ny);
                for (int i = 1; i <= nm; ++i) {
                    const double c = s.H->GetBinContent(i, j);
                    if (c == 0) continue;
                    s.H->SetBinContent(i, j, c * f);
                    s.H->SetBinError(i, j, s.H->GetBinError(i, j) * f);
                }
            }
    s.hDet = detFromResponse(s.H, measTmpl, name + "_det");
    return s;
}

//------------------------------------------------------------------------------------------------
// Unfolding + comparison
//------------------------------------------------------------------------------------------------
TH3D* unfold3D(RooUnfoldResponse* resp, const TH3* hDet, int it,
               RooUnfold::ErrorTreatment err, const TString& name) {
    RooUnfoldBayes unf(resp, hDet, it);
    unf.SetVerbose(0);
    TH1* hu = (TH1*) unf.Hunfold(err);
    TH3D* h = cloneTH3(hu, name);
    delete hu;
    return h;
}

TH3D* ratio3D(const TH3* num, const TH3* den, const TString& name) {   // numerator errors only
    TH3D* r = cloneTH3(num, name);
    r->Reset();
    for (int iz = 1; iz <= num->GetNbinsZ(); ++iz)
        for (int iy = 1; iy <= num->GetNbinsY(); ++iy)
            for (int ix = 1; ix <= num->GetNbinsX(); ++ix) {
                const double d = den->GetBinContent(ix, iy, iz);
                if (d <= 0) continue;
                r->SetBinContent(ix, iy, iz, num->GetBinContent(ix, iy, iz) / d);
                r->SetBinError  (ix, iy, iz, num->GetBinError  (ix, iy, iz) / d);
            }
    return r;
}

TH1D* ratio1D(const TH1* num, const TH1* den, const TString& name) {
    TH1D* r = (TH1D*) num->Clone(name);
    r->SetDirectory(0);
    r->Reset();
    for (int b = 1; b <= num->GetNbinsX(); ++b) {
        const double d = den->GetBinContent(b);
        if (d <= 0) continue;
        r->SetBinContent(b, num->GetBinContent(b) / d);
        r->SetBinError(b, num->GetBinError(b) / d);
    }
    return r;
}

TH1D* ptProjection(const TH3* h, const TString& name) {
    TH1D* p = h->ProjectionX(name, 1, h->GetNbinsY(), 1, h->GetNbinsZ());
    p->SetDirectory(0);
    return p;
}

// EEC-like projection in one pT bin: sum over the weight axis of (w bin center) * counts.
// The choice of w per bin cancels in unfolded/truth ratios.
TH1D* eecProjection(const TH3* h, int ipt, const TString& name) {
    const TAxis* ay = h->GetYaxis();
    const TAxis* az = h->GetZaxis();
    const TArrayD* xb = ay->GetXbins();
    TH1D* p = (xb->GetSize() > 0) ? new TH1D(name, name, ay->GetNbins(), xb->GetArray())
                                  : new TH1D(name, name, ay->GetNbins(), ay->GetXmin(), ay->GetXmax());
    p->SetDirectory(0);
    p->Sumw2();
    for (int iy = 1; iy <= ay->GetNbins(); ++iy) {
        double s = 0., e2 = 0.;
        for (int iz = 1; iz <= az->GetNbins(); ++iz) {
            const double w = az->GetBinCenter(iz);
            s  += w * h->GetBinContent(ipt, iy, iz);
            e2 += w * w * std::pow(h->GetBinError(ipt, iy, iz), 2);
        }
        p->SetBinContent(iy, s);
        p->SetBinError(iy, std::sqrt(e2));
    }
    return p;
}

// 3D ratio, pT-projection ratio and per-pT-bin EEC ratios; prints a compact summary
void addClosureRatios(const TH3* hUnf, const TH3* hRef, const TString& prefix, int it,
                      std::vector<TH1*>& out, const char* what = "unf/truth") {
    const TString sfx = TString::Format("_iter%d", it);
    out.push_back(ratio3D(hUnf, hRef, prefix + "_Ratio3D" + sfx));

    TH1D* pu = ptProjection(hUnf, prefix + "_ptU_tmp");
    TH1D* pr = ptProjection(hRef, prefix + "_ptR_tmp");
    TH1D* rpt = ratio1D(pu, pr, prefix + "_Ratio_pt" + sfx);
    out.push_back(rpt);
    printf("  %-34s iter %2d  %s pT:", prefix.Data(), it, what);
    for (int b = 1; b <= rpt->GetNbinsX(); ++b) printf(" %6.3f", rpt->GetBinContent(b));
    printf("\n  %-34s          max|EEC-1| per pT:", "");
    delete pu; delete pr;

    const TAxis* ax = hUnf->GetXaxis();
    for (int ipt = 1; ipt <= ax->GetNbins(); ++ipt) {
        const TString ptlab = TString::Format("_pt%g-%g", ax->GetBinLowEdge(ipt), ax->GetBinUpEdge(ipt));
        TH1D* eu = eecProjection(hUnf, ipt, prefix + "_eecU_tmp");
        TH1D* er = eecProjection(hRef, ipt, prefix + "_eecR_tmp");
        TH1D* r  = ratio1D(eu, er, prefix + "_Ratio_EEC" + ptlab + sfx);
        double maxdev = 0.;
        for (int b = 1; b <= r->GetNbinsX(); ++b)
            if (er->GetBinContent(b) > 0) maxdev = std::max(maxdev, std::abs(r->GetBinContent(b) - 1.));
        printf(" %6.3f", maxdev);
        out.push_back(r);
        delete eu; delete er;
    }
    printf("\n");
}

void closure_at_iter_3D(RooUnfoldResponse* resp, const TH3* hDet, const TH3* hRef, int it,
                        const TString& prefix, std::vector<TH1*>& out,
                        RooUnfold::ErrorTreatment err = kClosureErr) {
    TH3D* hUnf = unfold3D(resp, hDet, it, err, prefix + TString::Format("_Unfolded_iter%d", it));
    addClosureRatios(hUnf, hRef, prefix, it, out);
    out.push_back(hUnf);
}

//================================================================================================
// TRIVIAL CLOSURE TEST
// Unfold an MC det-level distribution with the ORIGINAL response, compare to the MC truth;
// refold the result and compare to the det-level input.
//   matched input -> reference = response->Htruth() (matched truth + misses)
//   all input     -> purity applied before, efficiency (Htruth/gen_all) after -> reference gen_all
//================================================================================================
std::vector<TH1*> trivial_closure_test_3D(RooUnfoldResponse* response, const TH3* rec, const TH3* ref,
                                          int n_iter, const TH3* efficiency, const TH3* purity,
                                          bool do_eff, bool do_purity, const TString& tag) {
    cout << "================ TRIVIAL CLOSURE TEST: " << tag << " ================" << endl;
    std::vector<TH1*> out;
    const TString prefix = "Trivial_" + tag;

    TH3D* rec_input = cloneTH3(rec, prefix + "_rec_input");
    TH3D* ref_input = cloneTH3(ref, prefix + "_ref");
    TH3D* rec_work  = cloneTH3(rec, prefix + "_rec_work");
    if (do_purity) rec_work->Multiply(purity);

    for (int it = 1; it <= n_iter; ++it) {
        TH3D* hUnf = unfold3D(response, rec_work, it, RooUnfold::kNoError,
                              prefix + TString::Format("_Unfolded_iter%d", it));
        if (do_eff) hUnf->Divide(efficiency);
        addClosureRatios(hUnf, ref_input, prefix, it, out);

        // Refold: undo efficiency, fold with the response, undo purity -> compare to det input
        TH3D* hFold = cloneTH3(hUnf, prefix + "_fold_tmp");
        if (do_eff) hFold->Multiply(efficiency);
        TH3D* hRefold = th3_refold(response, rec_input, hFold,
                                   prefix + TString::Format("_Refolded_iter%d", it));
        delete hFold;
        if (!hRefold) { out.push_back(hUnf); continue; }
        if (do_purity) hRefold->Divide(purity);

        TH1D* pf = ptProjection(hRefold, prefix + "_ptF_tmp");
        TH1D* pi = ptProjection(rec_input, prefix + "_ptI_tmp");
        TH1D* rf = ratio1D(pf, pi, prefix + TString::Format("_RefoldRatio_pt_iter%d", it));
        printf("  %-34s iter %2d  refold/input pT:", prefix.Data(), it);
        for (int b = 1; b <= rf->GetNbinsX(); ++b) printf(" %6.3f", rf->GetBinContent(b));
        printf("\n");
        delete pf; delete pi;

        out.insert(out.end(), {hUnf, hRefold, rf,
                               ratio3D(hRefold, rec_input, prefix + TString::Format("_RefoldRatio3D_iter%d", it))});
    }
    out.insert(out.end(), {rec_input, ref_input, rec_work});
    return out;
}

//================================================================================================
// STATISTICAL CLOSURE TEST
// 1. Det-level MC (in-range response) smeared with the data's relative stat. errors
// 2. Unfolded with the nominal closure response for i = 1..n_iter
// 3. Compared to the truth incl. misses
//================================================================================================
std::vector<TH1*> statistical_closure_test_3D(RooUnfoldResponse* resp_closure, const TH3* hDet,
                                              const TH3* hTruth, const TH3* h3_raw, int n_iter,
                                              const TString& obj) {
    cout << "================ STATISTICAL CLOSURE TEST: " << obj << " ================" << endl;
    std::vector<TH1*> out;
    const auto rel = getMeasuredErrors3D(h3_raw);
    TH3D* smeared = smearSpectrum3D(hDet, rel, "Statistical_" + obj + "_det_smeared");
    if (!smeared) return out;

    for (int it = 1; it <= n_iter; ++it)
        closure_at_iter_3D(resp_closure, smeared, hTruth, it, "Statistical_" + obj, out);

    out.push_back(smeared);
    return out;
}

//================================================================================================
// SHAPE CLOSURE TEST
// 1. RM and truth (incl. misses) reweighted by pt^p_pt RL^p_rl w^p_w at truth level
// 2. Scaled det-level smeared with the data's relative stat. errors
// 3. Unfolded with the NOMINAL closure response, compared to the SCALED truth
//================================================================================================
std::vector<TH1*> shape_closure_test_3D(RooUnfoldResponse* resp_closure, const TH2* H_inrange,
                                        const TH3* hTruth, const TH3* measTmpl, const TH3* h3_raw,
                                        int n_iter, const std::vector<PriorVariation>& variations,
                                        bool useBinCenter, const TString& obj) {
    cout << "================ SHAPE CLOSURE TEST: " << obj << " ================" << endl;
    std::vector<TH1*> out;
    const auto rel = getMeasuredErrors3D(h3_raw);

    for (const auto& pv : variations) {
        const TString prefix = "Shape_" + obj + "_" + pv.label();
        Scaled3D s = scaleResponse3D(H_inrange, hTruth, measTmpl, pv, useBinCenter, prefix);
        delete s.H; s.H = nullptr;      // only det/truth are needed here

        TH3D* smeared = smearSpectrum3D(s.hDet, rel, "Shape_" + prefix + "_det_scaled_smeared");
        if (!smeared) { delete s.hDet; delete s.hTruth; continue; }

        for (int it = 1; it <= n_iter; ++it)
            closure_at_iter_3D(resp_closure, smeared, s.hTruth, it, prefix, out);

        out.insert(out.end(), {s.hDet, s.hTruth, smeared});
    }
    return out;
}

//================================================================================================
// PRIOR CLOSURE TEST
// 1. Nominal det-level smeared with the data's relative stat. errors
// 2. RM and truth reweighted -> new response with a different prior (and consistent misses)
// 3. Smeared nominal det unfolded with the SCALED response, compared to the NOMINAL truth
//================================================================================================
std::vector<TH1*> prior_closure_test_3D(const TH2* H_inrange, const TH3* hDet, const TH3* hTruth,
                                        const TH3* h3_raw, int n_iter,
                                        const std::vector<PriorVariation>& variations,
                                        bool useBinCenter, const TString& obj) {
    cout << "================ PRIOR CLOSURE TEST: " << obj << " ================" << endl;
    std::vector<TH1*> out;
    const auto rel = getMeasuredErrors3D(h3_raw);
    TH3D* hDet_smeared = smearSpectrum3D(hDet, rel, "Prior_" + obj + "_det_smeared");
    if (!hDet_smeared) return out;
    out.push_back(hDet_smeared);

    for (const auto& pv : variations) {
        const TString prefix = "Prior_" + obj + "_" + pv.label();
        Scaled3D s = scaleResponse3D(H_inrange, hTruth, hDet, pv, useBinCenter, prefix);
        RooUnfoldResponse* resp_scaled = makeResponse3D(s.H, s.hDet, s.hTruth, prefix + "_response");
        delete s.H; s.H = nullptr;      // the response keeps its own copy
        delete s.hDet;

        for (int it = 1; it <= n_iter; ++it)
            closure_at_iter_3D(resp_scaled, hDet_smeared, hTruth, it, prefix, out);

        delete resp_scaled;
        out.push_back(s.hTruth);        // the prior used, for reference
    }
    return out;
}

void writeAndDelete(TDirectory* d, std::vector<TH1*>& v) {
    d->cd();
    for (auto* h : v) { h->Write(); delete h; }
    v.clear();
}



void unfolding_tests(const TString& objs_in = "AA", int rebinRL = 1,
                     int iter = 5, bool do_purity = true, 
                     TString outfile = "",   // empty -> built from objs_in
                     const TString& outdir = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/",
                     const TString& rm_file = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root",
                     const TString& data_file = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60320414/AnalysisResultsMerged_groomedbins.root",
                     const TString& unfolded_file = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded.root") { 

    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true);
    TH1::AddDirectory(false);   // every histogram below is owned by this macro

    // ======================= Parse objects =======================
    const std::vector<TString> allowed = {"AA", "AB", "BB", "rad"};
    std::vector<TString> objects;
    TObjArray* toks = objs_in.Tokenize(",");
    for (int i = 0; i < toks->GetEntries(); ++i) {
        TString t = ((TObjString*) toks->At(i))->GetString().Strip(TString::kBoth);
        if (t == "all") { objects = allowed; break; }
        if (std::find(allowed.begin(), allowed.end(), t) == allowed.end()) {
            std::cerr << "Error: unknown object '" << t << "' (allowed: AA, AB, BB, rad, all)" << std::endl;
            delete toks;
            return;
        }
        if (std::find(objects.begin(), objects.end(), t) == objects.end()) objects.push_back(t);
    }
    delete toks;
    if (objects.empty()) { std::cerr << "Error: no objects given" << std::endl; return; }

    if (outfile.IsNull()) {
        TString tag;
        for (const auto& o : objects) tag += (tag.IsNull() ? "" : "_") + o;
        outfile = outdir + "unfolded_tests_" + tag + ".root";
    }
    // If rebinning is applied, tag the output filename with the rebin factor
    // (inserted before the extension, e.g. unfolded_tests_AA.root -> unfolded_tests_AA_rebinRL2.root),
    // and read the unfolded results from the correspondingly tagged file
    // (unfolded.root -> unfolded_rebinRL2.root, as written by unfold.C with the same rebinRL)
    TString unfolded_file_use = unfolded_file;   // (param is const ref; retag locally)
    if (rebinRL > 1) {
        outfile = tagRebinFile(outfile, rebinRL);
        cout << "rebinRL=" << rebinRL << ": output file will be " << outfile << endl;
        unfolded_file_use = tagRebinFile(unfolded_file, rebinRL);
        cout << "rebinRL=" << rebinRL << ": unfolded results will be read from " << unfolded_file_use << endl;
    }

    cout << "Objects: " << objs_in << "  ->  output: " << outfile << endl;

    // Prior variations for shape and prior closure: {p_pt, p_rl, p_w}
    // Combined pt*weight variations (matching the det-level shape study):
    // pt^+0.15 * w^+0.1 and its mirror pt^-0.15 * w^-0.1
    const std::vector<PriorVariation> variations = {
        {+0.15, 0.0, +0.1}, {-0.15, 0.0, -0.1},
        // {+0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0},   // jet pT prior (as in 1D / pyjetty)
        // {0.0, +0.5, 0.0}, {0.0, -0.5, 0.0},   // RL prior
        // {0.0, 0.0, +0.1}, {0.0, 0.0, -0.1},   // weight prior
    };

    // ======================= Inputs =======================
    TFile* f_rm  = TFile::Open(rm_file);
    TFile* f_rawdata = TFile::Open(data_file);
    TFile* f_unf = TFile::Open(unfolded_file_use);
    if (!f_rm || !f_rawdata || !f_unf || f_rm->IsZombie() || f_rawdata->IsZombie() || f_unf->IsZombie()) {
        std::cerr << "Error: canno t open input files" << std::endl;
        return;
    }

    // ======================= Output =======================
    TFile* fout = TFile::Open(outfile, "RECREATE");
    if (!fout || fout->IsZombie()) { std::cerr << "Error: cannot create " << outfile << std::endl; return; }
    TDirectory* dir_refold  = fout->mkdir("refolding_closure");
    TDirectory* dir_trivial = fout->mkdir("trivial_closure");
    TDirectory* dir_stat    = fout->mkdir("statistical_closure");
    TDirectory* dir_shape   = fout->mkdir("shape_closure");
    TDirectory* dir_prior   = fout->mkdir("prior_closure");

    bool use_bin_center = false;
    
    for (const auto& obj : objects) {
        cout << "\n################ " << obj << " ################" << endl;

        RooUnfoldResponse* response = (RooUnfoldResponse*) f_rm->Get("groomed_" + obj + "_roounfold_response");
        if (!response) { std::cerr << "Error: response for " << obj << " not found" << std::endl; f_rm->ls(); return; }

        TH3D* h3_raw = (TH3D*) f_rawdata->Get(obj + "_sd0.1_raw");
        if (!h3_raw) { std::cerr << "Error: raw histogram not found" << std::endl; return; }
        h3_raw = cloneTH3(h3_raw, "h3_raw_" + obj);

        // ======================= Purity / efficiency =======================
        TH3D* h3_rec_num = (TH3D*) f_rm->Get("groomed_" + obj + "_reco_matched");
        TH3D* h3_rec_den = (TH3D*) f_rm->Get("groomed_" + obj + "_reco_all");
        TH3D* h3_gen_num = (TH3D*) f_rm->Get("groomed_" + obj + "_gen_matched");
        TH3D* h3_gen_den = (TH3D*) f_rm->Get("groomed_" + obj + "_gen_all");
        if (!h3_rec_num || !h3_rec_den || !h3_gen_num || !h3_gen_den) {
            std::cerr << "Error: reco_matched / reco_all / gen_matched / gen_all not found for " << obj << std::endl;
            return;
        }
        h3_rec_num->Sumw2(1);
        h3_rec_den->Sumw2(1);

        // ---- optional rebinning of the RL (Y) axis ----
        // Merge adjacent RL bins consistently in ALL inputs: the response
        // (measured, truth and flattened matrix), the raw data, and the
        // purity/efficiency inputs. The unfolded results are NOT rebinned here:
        // unfolded_file is auto-tagged _rebinRL<n> and expected from unfold.C
        // run with the same rebinRL, i.e. already in the rebinned binning.
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
            h_meas_r  = rebinY_TH3(tmpl_m, rebinRL, TString::Format("Hmeasured_%s_rebin%d", obj.Data(), rebinRL));
            h_truth_r = rebinY_TH3(tmpl_t, rebinRL, TString::Format("Htruth_%s_rebin%d", obj.Data(), rebinRL));
            Hmat_r    = rebinFlatResponse(response->Hresponse(),
                                          tmpl_m->GetNbinsX(), tmpl_m->GetNbinsY(), tmpl_m->GetNbinsZ(),
                                          tmpl_t->GetNbinsX(), tmpl_t->GetNbinsY(), tmpl_t->GetNbinsZ(),
                                          rebinRL, TString::Format("matrix_%s_rebin%d", obj.Data(), rebinRL));
            if (!h_meas_r || !h_truth_r || !Hmat_r) { std::cerr << "Error: RL rebinning failed" << std::endl; return; }
            response = new RooUnfoldResponse(h_meas_r, h_truth_r, Hmat_r,
                                             TString::Format("groomed_%s_roounfold_response_rebin%d", obj.Data(), rebinRL),
                                             TString::Format("groomed_%s response (RL rebin %d)", obj.Data(), rebinRL));
            resp_rebinned = true;
            cout << "RL rebin " << rebinRL << ": " << tmpl_m->GetNbinsY() << " -> " << tmpl_m->GetNbinsY() / rebinRL
                 << " RL bins (flat " << Hmat_r->GetNbinsX() << " x " << Hmat_r->GetNbinsY() << " matrix)" << endl;
        }

        // raw data and purity/efficiency inputs
        if (rebinRL > 1) {
            h3_raw      = rebinY_TH3(h3_raw,     rebinRL, TString::Format("%s_rebin%d", h3_raw->GetName(), rebinRL));
            h3_rec_num  = rebinY_TH3(h3_rec_num, rebinRL, TString::Format("groomed_%s_reco_matched_rebin%d", obj.Data(), rebinRL));
            h3_rec_den  = rebinY_TH3(h3_rec_den, rebinRL, TString::Format("groomed_%s_reco_all_rebin%d", obj.Data(), rebinRL));
            h3_gen_num  = rebinY_TH3(h3_gen_num, rebinRL, TString::Format("groomed_%s_gen_matched_rebin%d", obj.Data(), rebinRL));
            h3_gen_den  = rebinY_TH3(h3_gen_den, rebinRL, TString::Format("groomed_%s_gen_all_rebin%d", obj.Data(), rebinRL));
            if (!h3_raw || !h3_rec_num || !h3_rec_den || !h3_gen_num || !h3_gen_den) {
                std::cerr << "Error: RL rebinning of raw data / purity / efficiency inputs failed for "
                          << obj << std::endl;
                return;
            }
        }

        TH3D* purity = cloneTH3(h3_rec_num, "purity_" + obj);
        purity->Divide(h3_rec_num, h3_rec_den, 1., 1., "B");


        // ======================= Closure ingredients =======================
        TH3D* hMeasTmpl  = cloneTH3(response->Hmeasured(), "Hmeasured_" + obj);   // det binning
        TH3D* hTruthResp = cloneTH3(response->Htruth(),    "Htruth_" + obj);      // truth incl. misses
        if (!sameBinning3D(h3_raw, hMeasTmpl, "raw data vs response det axis") ||
            !sameBinning3D(h3_gen_den, hTruthResp, "gen_all vs response truth axis")) return;

        TH2* H_inrange   = inRangeResponse(response->Hresponse(), "H_inrange_" + obj);
        TH3D* hDetInRM   = detFromResponse(H_inrange, hMeasTmpl, "det_inRM_" + obj);
        diagnoseResponse(response, hMeasTmpl, hDetInRM, hTruthResp);
        RooUnfoldResponse* resp_closure = makeResponse3D(H_inrange, hDetInRM, hTruthResp, "resp_closure_" + obj);

        // Remaining (jet-level) efficiency: truth pairs of version-matched jets / all truth pairs
        TH3D* efficiency = cloneTH3(h3_gen_num, "efficiency_" + obj);
        efficiency->Divide(h3_gen_num, h3_gen_den, 1., 1., "B");

        wDiagonality(H_inrange, hMeasTmpl, hTruthResp);
        
        // ======================= TRIVIAL CLOSURE ! =======================
        struct TrivialCase { TString tag; const TH3* rec; const TH3* ref; bool eff, pur; };
        const std::vector<TrivialCase> cases = {
            {"Matched",       h3_rec_num, h3_gen_num, false, false},
            {"MatchedInRM",   hDetInRM,   hTruthResp, false, false},
            {"All",           h3_rec_den, h3_gen_den, true,  true },
            {"All_EffPurOff", h3_rec_den, h3_gen_den, false, false},
            {"All_OnlyEffOn", h3_rec_den, h3_gen_den, true,  false},
            {"All_OnlyPurOn", h3_rec_den, h3_gen_den, false, true },
        };
        std::vector<TH1*> trivial_hists;
        // for (const auto& c : cases) {
        //     auto v = trivial_closure_test_3D(response, c.rec, c.ref, iter, efficiency, purity,
        //                                      c.eff, c.pur, obj + "_" + c.tag);
        //     trivial_hists.insert(trivial_hists.end(), v.begin(), v.end());
        // }


        // ======================= REFOLD ! =======================
        // refolding_test takes std::string, so pass obj.Data()
        std::vector<TH3*> refolding_out_hists = refolding_test(obj.Data(), f_unf, response, h3_raw, iter, purity, do_purity);

        // ======================= STATISTICAL CLOSURE ! =======================
        // auto stat_hists = statistical_closure_test_3D(resp_closure, hDetInRM, hTruthResp, h3_raw, iter, obj);
        auto stat_hists = statistical_closure_test_3D(response, h3_rec_num, h3_gen_num, h3_raw, iter, obj);

        // ======================= SHAPE CLOSURE ! =======================
        // auto shape_hists = shape_closure_test_3D(resp_closure, H_inrange, hTruthResp, hMeasTmpl, h3_raw,
        //                                          iter, variations, use_bin_center, obj);
        auto shape_hists = shape_closure_test_3D(response, response->Hresponse(), hTruthResp, h3_rec_num, h3_raw,
                                                 iter, variations, use_bin_center, obj);

        // ======================= PRIOR CLOSURE ! =======================
        // auto prior_hists = prior_closure_test_3D(H_inrange, hDetInRM, hTruthResp, h3_raw,
        //                                          iter, variations, use_bin_center, obj);
        auto prior_hists = prior_closure_test_3D(H_inrange, h3_rec_num, hTruthResp, h3_raw,
                                                 iter, variations, use_bin_center, obj);

    
        //Write objects here
        cout << "writing closure test hists" << endl;
        fout->cd();
        h3_raw->Write();
        if (do_purity) purity->Write();

        writeAndDelete(dir_trivial, trivial_hists);
        dir_refold->cd();
        for (auto h : refolding_out_hists) { h->Write(); delete h; }
        writeAndDelete(dir_stat,  stat_hists);
        writeAndDelete(dir_shape, shape_hists);
        writeAndDelete(dir_prior, prior_hists);

        // per-object cleanup (matters when running several objects in one job)
        delete h3_raw; delete purity; delete efficiency;
        delete hMeasTmpl; delete hTruthResp; delete H_inrange; delete hDetInRM;
        delete resp_closure;
        if (resp_rebinned) {
            // cleanup of rebinned intermediates (the rebuilt response keeps its own
            // copies of the measured/truth/matrix histograms, so these can go)
            delete response;
            delete h_meas_r; delete h_truth_r; delete Hmat_r;
        }
        if (rebinRL > 1) {
            // rebinned purity/efficiency inputs are macro-owned (the originals
            // belong to the input file and stay with it)
            delete h3_rec_num; delete h3_rec_den; delete h3_gen_num; delete h3_gen_den;
        }
        

    }
    
    fout->Close();
    f_rm->Close(); f_rawdata->Close(); f_unf->Close();
    cout << "Wrote output to " << outfile << endl;
}
