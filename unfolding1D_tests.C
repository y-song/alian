// TODO: check the binnings. check the proper inputs are being fed in?
#include <iostream>
#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TMatrixD.h>
#include <TSystem.h>
#include <TString.h>
#include <TRandom3.h>
#include <TF1.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// Inspired from: https://github.com/ezradlesser/pyjetty/blob/09e41284e6bd7f289e457875d9e15224eaa34dc2/pyjetty/alice_analysis/analysis/user/substructure/roounfold_obs.py#L1233

// USAGE
// don't forget to run roounfold_preload_on first! -- prob not needed anymore
// root -q unfolding1D_tests.C



// Fold a truth-level histogram with the normalized response matrix M (nMeas x nTruth).
// Result has the binning of hTemplate (measured binning). No fakes are added.
// Note: errors assume uncorrelated truth bins (the unfolded covariance is ignored).
// This is a replacement for ApplyToTruth, because for whatever reason I can't get that to work.
TH1D* foldWithMatrix(const TMatrixD& M, const TH1* hTruthIn, const TH1* hTemplate, const TString& name) {
    TH1D* h = (TH1D*) hTemplate->Clone(name);
    h->SetDirectory(0);
    h->Reset();
    const int nMeas = M.GetNrows(), nTruth = M.GetNcols();
    for (int k = 0; k < nMeas; ++k) {
        double val = 0, err2 = 0;
        for (int j = 0; j < nTruth; ++j) {
            val  += M(k, j) * hTruthIn->GetBinContent(j + 1);
            err2 += M(k, j) * M(k, j) * std::pow(hTruthIn->GetBinError(j + 1), 2);
        }
        h->SetBinContent(k + 1, val);
        h->SetBinError(k + 1, std::sqrt(err2));
    }
    return h;
}


// h_mig_raw is 2D histogram with measured/reco in x axis and truth in y-axis
// h_meas_template and h_true_template are any histograms that can give the correct binning
// h_unfolded_truth is the unfolded result (that needs to be refolded)
TH1D* th1_manual_refold_from_raw_migration(TH2* h_mig_raw, TH1D* h_meas_template, TH1* h_true_template, TH1* h_unfolded_truth, const TString& name, bool include_reco_overflow = true) { //TH1D* h_fake_to_add, 

    const int nx_meas = h_meas_template->GetNbinsX(); //M.GetNrows();
    const int nx_true = h_true_template->GetNbinsX(); //M.GetNcols();

    // Column sums = matched truth per truth bin -> P(reco i | true j, matched)
    const int ix_m_lo = include_reco_overflow ? 0 : 1;
    const int ix_m_hi = include_reco_overflow ? nx_meas + 1 : nx_meas;
    // Get normalized response matrix
    std::vector<double> norm(nx_true + 2, 0.0);
    for (int ix_t = 1; ix_t <= nx_true; ++ix_t)
        for (int ix_m = ix_m_lo; ix_m <= ix_m_hi; ++ix_m)
            norm[ix_t] += h_mig_raw->GetBinContent(ix_m, ix_t);


    TH1D* h_refold = static_cast<TH1D*>(h_meas_template->Clone(name));
    h_refold->SetDirectory(0); //?
    h_refold->Reset();

    for (int ix_m = 1; ix_m <= nx_meas; ++ix_m) {

        double pred = 0.0;
        double pred_error_sq = 0.0;

        for (int ix_t = 1; ix_t <= nx_true; ++ix_t) {

            if (norm[ix_t] <= 0.0) continue;
            double mij = h_mig_raw->GetBinContent(ix_m, ix_t); 
            if (mij <= 0.0) continue;

            double u_j = h_unfolded_truth->GetBinContent(ix_t); 
            double u_j_err = h_unfolded_truth->GetBinError(ix_t);

            double transfer_coeff = mij / norm[ix_t];
            pred += transfer_coeff * u_j;
            pred_error_sq += std::pow(transfer_coeff * u_j_err, 2.);

        }

        h_refold->SetBinContent(ix_m, pred); 
        h_refold->SetBinError(ix_m, std::sqrt(pred_error_sq)); 
    }

    return h_refold;
}


// Error treatment used for all closure unfoldings (pyjetty's Hreco() defaults to diagonal errors;
// kCovariance matches what you use for the data unfolding)
static RooUnfold::ErrorTreatment kClosureErr = RooUnfold::kCovariance;

//------------------------------------------------------------------------------------------------
// Relative statistical errors of the measured data, per bin (analogue of getMeasuredErrors)
//------------------------------------------------------------------------------------------------
std::vector<double> getMeasuredErrors(const TH1* hData) {
    const int n = hData->GetNbinsX();
    std::vector<double> rel(n + 2, 0.);
    for (int i = 1; i <= n; ++i) {
        const double c = hData->GetBinContent(i), e = hData->GetBinError(i);
        rel[i] = (c > 0) ? e / c : 0.;
    }
    return rel;
}

//------------------------------------------------------------------------------------------------
// Smear a spectrum by the measured relative errors (analogue of smearSpectrum).
// Each bin: content -> Gaus(content, content*relErr), error -> content*relErr.
// Returns a new histogram (input is not modified).
//------------------------------------------------------------------------------------------------
TH1D* smearSpectrum(const TH1* h, const std::vector<double>& relErr, TRandom3& rng, const TString& name) {
    const int n = h->GetNbinsX();
    if ((int) relErr.size() != n + 2) {
        std::cerr << "smearSpectrum: binning mismatch (" << n << " bins vs " << relErr.size() - 2 << " errors)" << std::endl;
        return nullptr;
    }
    TH1D* hs = (TH1D*) h->Clone(name);
    hs->SetDirectory(0);
    for (int i = 1; i <= n; ++i) {
        const double mu = h->GetBinContent(i), sig = mu * relErr[i];
        double val = (sig > 0) ? rng.Gaus(mu, sig) : mu;
        if (val < 0) val = 0;
        hs->SetBinContent(i, val);
        hs->SetBinError(i, sig);
    }
    return hs;
}

//------------------------------------------------------------------------------------------------
// Project a response matrix (x = reco, y = truth) to det- and truth-level spectra.
// Truth: all reco incl. under/overflow. Det: reco for truth inside the truth range.
//------------------------------------------------------------------------------------------------
void projectResponse(const TH2* RM, TH1D*& hDet, TH1D*& hTruth, const TString& tag) {
    const int nr = RM->GetNbinsX(), nt = RM->GetNbinsY();
    hDet   = RM->ProjectionX("hMC_Det_"   + tag, 1, nt);
    hTruth = RM->ProjectionY("hMC_Truth_" + tag, 0, nr + 1);
    hDet->SetDirectory(0);
    hTruth->SetDirectory(0);
}

//------------------------------------------------------------------------------------------------
// Scale a response matrix along the truth axis by f(truth bin center)
//------------------------------------------------------------------------------------------------
TH2* scaleResponseTruth(const TH2* RM, TF1* f, const TString& name) {
    TH2* s = (TH2*) RM->Clone(name);
    s->SetDirectory(0);
    const int nx = RM->GetNbinsX(), ny = RM->GetNbinsY();
    for (int iy = 1; iy <= ny; ++iy) {
        const double w = f->Eval(RM->GetYaxis()->GetBinCenter(iy));
        for (int ix = 0; ix <= nx + 1; ++ix) {
            s->SetBinContent(ix, iy, RM->GetBinContent(ix, iy) * w);
            s->SetBinError  (ix, iy, RM->GetBinError(ix, iy)   * w);
        }
    }
    return s;
}

//------------------------------------------------------------------------------------------------
// Rebin a finely binned RM into the binning of a template RM (bin edges must align)
//------------------------------------------------------------------------------------------------
TH2* rebinResponseTo(const TH2* fine, const TH2* tmpl, const TString& name) {
    auto checkEdges = [](const TAxis* f, const TAxis* c, const char* ax) {
        for (int i = 1; i <= c->GetNbins() + 1; ++i) {
            const double e = c->GetBinLowEdge(i);
            const int b = f->FindFixBin(e);
            if (b >= 1 && b <= f->GetNbins() && std::fabs(f->GetBinLowEdge(b) - e) > 1e-6)
                printf("  rebinResponseTo: %s edge %g not aligned with fine binning\n", ax, e);
        }
    };
    checkEdges(fine->GetXaxis(), tmpl->GetXaxis(), "reco");
    checkEdges(fine->GetYaxis(), tmpl->GetYaxis(), "truth");

    TH2* c = (TH2*) tmpl->Clone(name);
    c->SetDirectory(0);
    c->Reset();
    for (int ix = 0; ix <= fine->GetNbinsX() + 1; ++ix)
        for (int iy = 0; iy <= fine->GetNbinsY() + 1; ++iy) {
            const int bx = c->GetXaxis()->FindFixBin(fine->GetXaxis()->GetBinCenter(ix));
            const int by = c->GetYaxis()->FindFixBin(fine->GetYaxis()->GetBinCenter(iy));
            const int b  = c->GetBin(bx, by);
            const double e_old = c->GetBinError(b), e_new = fine->GetBinError(ix, iy);
            c->SetBinContent(b, c->GetBinContent(b) + fine->GetBinContent(ix, iy));
            c->SetBinError(b, std::sqrt(e_old * e_old + e_new * e_new));
        }
    return c;
}

//------------------------------------------------------------------------------------------------
// Common piece: unfold hDet with the NOMINAL response at iteration i, compare to hTruth
//------------------------------------------------------------------------------------------------
void closure_at_iter(RooUnfoldResponse* response, TH1* hDet, TH1* hTruth, int i,
                     const TString& prefix, std::vector<TH1*>& out) {
    RooUnfoldBayes unf(response, hDet, i);
    unf.SetVerbose(0);
    TH1* hu = (TH1*) unf.Hunfold(kClosureErr);
    TH1* hUnf = (TH1*) hu->Clone(prefix + TString::Format("_Unfolded1Diter%d", i));
    hUnf->SetDirectory(0);
    delete hu;

    // Ratio: unfolded / truth, with unfolded errors only (truth is the reference)
    TH1* hRatio = (TH1*) hUnf->Clone(prefix + TString::Format("_RatioToTruth_iter%d", i));
    hRatio->SetDirectory(0);
    hRatio->Reset();
    printf("  %-28s iter %2d  unf/truth:", prefix.Data(), i);
    for (int j = 1; j <= hTruth->GetNbinsX(); ++j) {
        const double t = hTruth->GetBinContent(j);
        if (t > 0) {
            hRatio->SetBinContent(j, hUnf->GetBinContent(j) / t);
            hRatio->SetBinError(j, hUnf->GetBinError(j) / t);
        }
        printf(" %6.3f", hRatio->GetBinContent(j));
    }
    printf("\n");
    out.insert(out.end(), {hUnf, hRatio});
}

//------------------------------------------------------------------------------------------------
// Shape variation: scaled RM, its projections, smeared det-level
//------------------------------------------------------------------------------------------------
struct ShapeVariation {
    double  param;
    TString label;
    TH2*    RM_scaled;
    TH1D*   hMC_Truth;
    TH1D*   hMC_Det;          // unsmeared
    TH1D*   hMC_Det_smeared;
};

// RM_for_scaling: RM on which the scaling is applied (finely binned if available, else RM_2D)
// RM_template   : nominal RM binning (the one used for unfolding)
ShapeVariation make_shape_variation(TH2* RM_for_scaling, TH2* RM_template, TF1* f_shape, double param,
                                    const std::vector<double>& measuredErrors, TRandom3& rng) {
    ShapeVariation sv;
    sv.param = param;
    sv.label = TString::Format("shape%g", param);
    sv.label.ReplaceAll(".", "").ReplaceAll("-", "m");

    f_shape->SetParameter(0, param);
    TH2* scaled = scaleResponseTruth(RM_for_scaling, f_shape, "RM_scaled_tmp_" + sv.label);
    if (RM_for_scaling != RM_template) {
        sv.RM_scaled = rebinResponseTo(scaled, RM_template, "RM_scaled_" + sv.label);
        delete scaled;
    } else {
        sv.RM_scaled = scaled;
        sv.RM_scaled->SetName("RM_scaled_" + sv.label);
    }

    projectResponse(sv.RM_scaled, sv.hMC_Det, sv.hMC_Truth, sv.label);
    sv.hMC_Det_smeared = smearSpectrum(sv.hMC_Det, measuredErrors, rng, "hMC_Det_smeared_" + sv.label);
    return sv;
}



// REFOLDING TEST
// Feed in unfolded data distribution, "refold" (multiply) if with the RM
// To be compared: refolded distribuion to original raw (detector) data
// std::vector<TH1*> refolding_test(TFile * f_unf, TMatrixD M, TH1D * h1_raw, int iter, TH1D * efficiency, TH1D * purity, bool do_eff, bool do_purity, std::string addname="") {
std::vector<TH1*> refolding_test(TFile * f_unf, TH2* RM_2D, TH1D * h1_raw, int iter, TH1D * efficiency, TH1D * purity, bool do_eff, bool do_purity, std::string addname="") {

    std::vector<TH1*> out_hists;

    for (int i = 1; i <= iter; ++i) {
        TString unf_name = TString::Format("Baysian_Unfolded1Diter%d", i);
        TH1* h1_unf_in = (TH1*) f_unf->Get(unf_name);
        if (!h1_unf_in) { cout << "Skipping " << unf_name << " (not found)" << endl; continue; }
        if (h1_unf_in->GetNbinsX() != RM_2D->GetNbinsY()) { //M.GetNcols()) {
            std::cerr << "Error: " << unf_name << " bins != response truth bins, skipping" << std::endl;
            continue;
        }

        cout << "------------------------------------------" << endl;
        cout << "iter " << i << endl;

        // Truth level that M acts on
        TH1* h1_truth = (TH1*) h1_unf_in->Clone(TString::Format("%s", addname.c_str()) + unf_name); // + "_Clone"); //TString::Format("h1_jetpt_groomed_unfolded%d", i));
        h1_truth->SetDirectory(0);
        if (do_eff) h1_truth->Multiply(efficiency); // undo efficiency

        // Fold --> manual function b/c I can't get ApplyToTruth() to work :(
        // TH1D* h1_refold = foldWithMatrix(M, h1_truth, h1_raw, TString::Format("%sBaysian_Refolded1Diter%d", addname.c_str(), i));
        TH1D* h1_refold = th1_manual_refold_from_raw_migration(RM_2D, h1_raw, h1_truth, h1_truth, TString::Format("%sBaysian_Refolded1Diter%d", addname.c_str(), i));
        h1_refold->SetTitle(TString::Format("Refolded, iter %d", i));

        // Undo purity: add matching fakes back so it is comparable to raw
        if (do_purity) h1_refold->Divide(purity);

        cout << "unfolded = " << h1_unf_in->Integral()
             << ", refolded = " << h1_refold->Integral()
             << ", raw = " << h1_raw->Integral() << endl;
        cout << endl;

        out_hists.insert(out_hists.end(), {h1_truth, h1_refold});
    }

    return out_hists;

}

// TRIVIAL CLOSURE TEST
// Don't do any purity/efficiency corrections for this part.
// Feed in detector MC distribution, unfold it with the RM
// Then take unfolded MC and refold it with RM
// To be compared: unfolded (tr) MC with original truth MC AND refolded (det) MC with original det MC
// std::vector<TH1*> trivial_closure_test(TH1D * rec, TH1D *gen, RooUnfoldResponse* response1D, int iter, TMatrixD M, TH1D * efficiency, TH1D * purity, bool do_eff, bool do_purity, std::string match_str) {
std::vector<TH1*> trivial_closure_test(TH1D * rec, TH1D *gen, RooUnfoldResponse* response1D, int iter, TH2* RM_2D, TH1D * efficiency, TH1D * purity, bool do_eff, bool do_purity, std::string match_str) {

    cout << "================ TRIVIAL CLOSURE TEST ================" << endl;

    // Save all to vector to return
    std::vector<TH1*> out_hists_temp;
    std::vector<TH1*> out_hists;
    
    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Create temp output file
    const TString& temp_outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/trivial_unfolding_temp.root";
    TFile* fout_temp = new TFile(temp_outfile, "RECREATE");

    // Clone to avoid modifying the original reference histogram in-place
    TH1D* rec_input = (TH1D*) rec->Clone(TString::Format("rec_%s_input", match_str.c_str()));
    TH1D* gen_input = (TH1D*) gen->Clone(TString::Format("gen_%s_input", match_str.c_str()));
    rec_input->SetDirectory(0);
    gen_input->SetDirectory(0);

    TH1D* rec_work = (TH1D*) rec->Clone(TString::Format("rec_work_%s", match_str.c_str()));
    TH1D* gen_work = (TH1D*) gen->Clone(TString::Format("gen_work_%s", match_str.c_str()));
    rec_work->SetDirectory(0);
    gen_work->SetDirectory(0);
    if (do_purity) rec_work->Multiply(purity);

    for (int i = 1; i <= iter; ++i) {
        // Unfold the 3D histogram
        RooUnfoldBayes unfold1D(response1D, rec_work, i);
        TH1* hunf1D = (TH1*)unfold1D.Hunfold(errorTreatment);
        // TH1* hfold1D = response1D->ApplyToTruth (hunf1D, "");

        // Clone and name histograms
        TH1* htempUnf1D = (TH1*)hunf1D->Clone(TString::Format("Baysian_Unfolded1Diter%d", i));
        htempUnf1D->SetDirectory(0);

        // efficiency correction done here
        if (do_eff) htempUnf1D->Divide(efficiency);   // final, efficiency-corrected
        
        // Write histograms to output file
        fout_temp->cd();
        htempUnf1D->Write();

        // And rename for longer term save in closure tests file
        htempUnf1D->SetName(TString::Format("Trivial_Baysian_%s_Unfolded1Diter%d", match_str.c_str(), i));
        out_hists_temp.push_back(htempUnf1D);
    }
    fout_temp->Close(); // potentially might need to remove this
    // delete rec_work;

    

    // Now do all refoldings
    fout_temp = new TFile(temp_outfile, "READ");
    std::vector<TH1*> trivial_refolding_out_hists = refolding_test(fout_temp, RM_2D, rec_input, iter, efficiency, purity, do_eff, do_purity); //rec_input instead of rec_work so its not purity corrected (even though this doesn't matter, as it's just used or binning)
    for (auto h : trivial_refolding_out_hists) h->SetDirectory(0);
    

    for ( int i = 1; i <= iter; ++i) {
        TH1 * hunfolded = out_hists_temp[i-1];
        TH1 * hunfolded_repeat = trivial_refolding_out_hists[2*(i-1)]; //[i-1][0];
        TH1 * hrefolded = trivial_refolding_out_hists[2*(i-1)+1]; //[i-1][1];

        hrefolded->SetName(TString::Format("Trivial_Baysian_%s_Refolded1Diter%d", match_str.c_str(), i));
        
        out_hists.insert(out_hists.end(), {hunfolded, hrefolded});
    }

    // if (match_str == "Matched" || match_str == "All") {
    cout << "HISTOGRAM NAMES" << endl;
    cout << rec_input->GetName() << endl;
    cout << gen_input->GetName() << endl;
    cout << rec_work->GetName() << endl;
    cout << gen_work->GetName() << endl;
    out_hists.insert(out_hists.end(), {rec_input, gen_input, rec_work, gen_work});
    // }

    // Close and delete temp files
    fout_temp->Close();
    delete fout_temp;
    gSystem->Unlink(temp_outfile);

    return out_hists;

}



// STATISTICAL CLOSURE TEST
// 1. MC det-level spectrum (reco projection of RM) is smeared by the relative stat. errors of data
// 2. Smeared spectrum is unfolded with the nominal response for i = 1..iter
// 3. Compared to MC truth (truth projection of RM)
std::vector<TH1*> statistical_closure_test(RooUnfoldResponse* response, TH2* RM_2D, TH1D* h1_raw,
                                           int iter, UInt_t seed = 12345) {
    cout << "================ STATISTICAL CLOSURE TEST ================" << endl;
    std::vector<TH1*> out;
    TRandom3 rng(seed);

    TH1D *hMC_Det = nullptr, *hMC_Truth = nullptr;
    projectResponse(RM_2D, hMC_Det, hMC_Truth, "nominal");

    // Sanity check: are there misses/fakes in the response?
    for (int j = 1; j <= hMC_Truth->GetNbinsX(); ++j) {
        const double h = response->Htruth()->GetBinContent(j);
        if (h > 0 && std::fabs(hMC_Truth->GetBinContent(j) / h - 1) > 1e-3)
            printf("  WARNING: truth bin %d: RM projection / Htruth = %.4f\n", j, hMC_Truth->GetBinContent(j) / h);
    }
    for (int i = 1; i <= hMC_Det->GetNbinsX(); ++i) {
        const double h = response->Hmeasured()->GetBinContent(i);
        if (h > 0 && std::fabs(hMC_Det->GetBinContent(i) / h - 1) > 1e-3)
            printf("  WARNING: reco bin %d: RM projection / Hmeasured = %.4f\n", i, hMC_Det->GetBinContent(i) / h);
    }

    // Smear once, reuse for all iterations
    std::vector<double> measuredErrors = getMeasuredErrors(h1_raw);
    TH1D* hMC_Det_smeared = smearSpectrum(hMC_Det, measuredErrors, rng, "hMC_Det_smeared_nominal");
    if (!hMC_Det_smeared) return out;

    for (int i = 1; i <= iter; ++i)
        closure_at_iter(response, hMC_Det_smeared, hMC_Truth, i, "Statistical", out);

    out.insert(out.end(), {hMC_Det, hMC_Truth, hMC_Det_smeared});
    return out;
}

// SHAPE CLOSURE TEST
// 1. RM scaled along truth axis by f_shape (one variation per parameter), projected to det/truth
// 2. Scaled det-level spectrum smeared by the relative stat. errors of data
// 3. Unfolded with the NOMINAL response for i = 1..iter, compared to scaled truth
std::vector<TH1*> shape_closure_test(RooUnfoldResponse* response, TH2* RM_2D, TH1D* h1_raw, int iter,
                                     TF1* f_shape, const std::vector<double>& shape_params,
                                     TH2* RM_fine = nullptr, UInt_t seed = 54321) {
    cout << "================ SHAPE CLOSURE TEST ================" << endl;
    std::vector<TH1*> out;
    TRandom3 rng(seed);

    std::vector<double> measuredErrors = getMeasuredErrors(h1_raw);
    TH2* RM_scale_input = RM_fine ? RM_fine : RM_2D;

    for (double s : shape_params) {
        ShapeVariation sv = make_shape_variation(RM_scale_input, RM_2D, f_shape, s, measuredErrors, rng);
        if (!sv.hMC_Det_smeared) continue;

        for (int i = 1; i <= iter; ++i)
            closure_at_iter(response, sv.hMC_Det_smeared, sv.hMC_Truth, i, "Shape_" + sv.label, out);

        out.insert(out.end(), {(TH1*) sv.RM_scaled, sv.hMC_Det, sv.hMC_Truth, sv.hMC_Det_smeared});
    }
    return out;
}




void refold_convergence_check(RooUnfoldResponse* response, TH2* RM_2D, TH1D* h1_raw, TH1D* purity) {
    cout << "IN REFOLD CONVERGENCE CHECK NOW" << endl;
    // Bin edges must match, not just bin counts
    for (int i = 1; i <= h1_raw->GetNbinsX() + 1; ++i) {
        double a = h1_raw->GetXaxis()->GetBinLowEdge(i), b = RM_2D->GetXaxis()->GetBinLowEdge(i);
        if (std::fabs(a - b) > 1e-6) printf("EDGE MISMATCH at edge %d: data %g vs RM %g\n", i, a, b);
    }

    TH1D* d = (TH1D*) h1_raw->Clone("d_purcorr");
    d->SetDirectory(0);
    d->Multiply(purity);   // exactly what goes into the unfolding

    for (int it : {1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50}) {
        RooUnfoldBayes unf(response, d, it);
        TH1* u = unf.Hunfold(RooUnfold::kNoError);
        TH1D* rf = th1_manual_refold_from_raw_migration(RM_2D, d, response->Htruth(), u, TString::Format("rf_check_it%d", it));
        printf("iter %2d:", it);
        for (int i = 1; i <= d->GetNbinsX(); ++i)
            printf(" %6.3f", d->GetBinContent(i) > 0 ? rf->GetBinContent(i) / d->GetBinContent(i) : 0.);
        printf("\n");
        delete rf; delete u;
    }
    delete d;
    cout << "OUT OF REFOLD CONVERGENCE CHECK NOW" << endl;
}

void unfolding1D_tests(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root",
                       const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60320414/AnalysisResultsMerged_groomedbins.root", //"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60017705/AnalysisResultsMerged_groomedbins.root",
                       const TString& unfolded_file="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D.root",
                       const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D_tests.root",
                       const TString& resp_name="roounfold_response_1D_groomed_oldversion",
                       int iter=5, bool do_eff=true, bool do_purity=true) { 
    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true);
    // RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // ======================= Inputs =======================
    TFile* f_rm  = TFile::Open(rm_file);
    TFile* f_rawdata = TFile::Open(data_file);
    TFile* f_unf = TFile::Open(unfolded_file);
    // TFile* f_unf_noeffpur = TFile::Open("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D_noeffpur.root");
    if (!f_rm || !f_rawdata || !f_unf || f_rm->IsZombie() || f_rawdata->IsZombie() || f_unf->IsZombie()) {
        std::cerr << "Error: canno t open input files" << std::endl;
        return;
    }

    RooUnfoldResponse* response = (RooUnfoldResponse*) f_rm->Get(resp_name);
    
    if (!response) {
        std::cerr << "Error: " << resp_name << " not found. File contents:" << std::endl;
        f_rm->ls();
        return;
    }

    // Raw: prefer the copy written by unfold1D.C (exactly what was unfolded)
    TH1D* h1_raw = (TH1D*) f_rawdata->Get("groomed_sd0.1_jet_pt_raw1D");
    if (!h1_raw) { std::cerr << "Error: raw histogram not found" << std::endl; return; }
    h1_raw = (TH1D*) h1_raw->Clone("h1_raw");
    h1_raw->SetDirectory(0);

    // ======================= Response =======================
    TMatrixD M = response->Mresponse();
    TH2* RM_2D = (TH2*) response->Hresponse()->Clone("matrix1D_raw_working");
    RM_2D->SetDirectory(0);
    cout << "Mresponse: " << M.GetNrows() << " x " << M.GetNcols() << ", sum = " << M.Sum() << endl;
    if (M.GetNrows() != h1_raw->GetNbinsX()) {
        std::cerr << "Error: response measured bins (" << M.GetNrows()
                  << ") != raw bins (" << h1_raw->GetNbinsX() << ")" << std::endl;
        return;
    }

    // Get projections of the response matrix
    // Direct access to the unnormalized projections from RooUnfoldResponse
    TH1D* rec_fromRM = (TH1D*) response->Hmeasured()->Clone("rec_fromRM");
    TH1D* gen_fromRM = (TH1D*) response->Htruth()->Clone("gen_fromRM");

    // Unlink from current directory so they persist safely
    rec_fromRM->SetDirectory(0);
    gen_fromRM->SetDirectory(0);
    

    // ======================= Calculate Purity / efficiency (don't apply until later) =======================
    cout << "purity correction is running (backwards)" << endl;
    TH1D* rec_num = (TH1D*) f_rm->Get("jet_match_rec_pur_num_groomed");
    TH1D* rec_den = (TH1D*) f_rm->Get("jet_all_rec_pur_den_groomed");
    TH1D* gen_num = (TH1D*) f_rm->Get("jet_match_gen_eff_num_groomed");
    TH1D* gen_den = (TH1D*) f_rm->Get("jet_all_gen_eff_den_groomed");
    if (!rec_num || !rec_den || !gen_num || !gen_den) {
        std::cerr << "Error: purity/efficiency histograms not found in RM file" << std::endl;
        return;
    }
    
    // TH1D* rec_num_in = RM_2D->ProjectionX("rec_num_inTruthRange", 1, RM_2D->GetNbinsY()); // TEMPORARY FIX
    // TH1D* purity = (TH1D*) rec_num_in->Clone("purity"); // TEMPORARY FIX
    TH1D* purity = (TH1D*) rec_num->Clone("purity");
    purity->SetDirectory(0);
    purity->Divide(rec_num, rec_den, 1., 1., "B"); //h = (c1 * h1) / (c2 * h2), bin by bin, with binomial errors
    TH1D* efficiency = (TH1D*) gen_num->Clone("efficiency");
    efficiency->SetDirectory(0);
    efficiency->Divide(gen_num, gen_den, 1., 1., "B"); //h = (c1 * h1) / (c2 * h2), bin by bin, with binomial errors

    // refold_convergence_check(response, RM_2D, h1_raw, purity);



    // ======================= TRIVIAL CLOSURE ! =======================
    // matched
    std::vector<TH1*> trivial_closure_out_matched_hists = trivial_closure_test(rec_num, gen_num, response, iter, RM_2D, efficiency, purity, false, false, "Matched");

    // matched, projected from RMs
    std::vector<TH1*> trivial_closure_out_matched_fromrms_hists = trivial_closure_test(rec_fromRM, gen_fromRM, response, iter, RM_2D, efficiency, purity, false, false, "MatchedFromRMs");

    // unmatched/all - eff/pur both on
    std::vector<TH1*> trivial_closure_out_all_hists = trivial_closure_test(rec_den, gen_den, response, iter, RM_2D, efficiency, purity, true, true, "All");

    // check: unmatched/all - eff/pur both off
    std::vector<TH1*> trivial_closure_out_all_effpuroff_hists = trivial_closure_test(rec_den, gen_den, response, iter, RM_2D, efficiency, purity, false, false, "All_EffPurOff");

    // check: unmatched/all - just eff on
    std::vector<TH1*> trivial_closure_out_all_onlyeffon_hists = trivial_closure_test(rec_den, gen_den, response, iter, RM_2D, efficiency, purity, true, false, "All_OnlyEffOn");

    // check: unmatched/all - just purity on
    std::vector<TH1*> trivial_closure_out_all_onlypuron_hists = trivial_closure_test(rec_den, gen_den, response, iter, RM_2D, efficiency, purity, false, true, "All_OnlyPurOn");
    
    // ======================= REFOLD ! =======================
    std::vector<TH1*> refolding_out_hists = refolding_test(f_unf, RM_2D, h1_raw, iter, efficiency, purity, do_eff, do_purity);

    // // check: do the version with no eff/pur
    // std::vector<TH1*> refolding_out_hists_noeffpur = refolding_test(f_unf_noeffpur, RM_2D, h1_raw, iter, efficiency, purity, false, false, "NoEffPur_");

    
    // ======================= STATISTICAL CLOSURE ! =======================
    std::vector<TH1*> stat_closure_hists = statistical_closure_test(response, RM_2D, h1_raw, iter + 2);

    // ======================= SHAPE CLOSURE ! =======================
    // PLACEHOLDER: replace with pyjetty's shape variation function and parameters
    TF1 f_shape("f_shape", "pow(x, [0])", RM_2D->GetYaxis()->GetXmin(), RM_2D->GetYaxis()->GetXmax());
    std::vector<TH1*> shape_closure_hists = shape_closure_test(response, RM_2D, h1_raw, iter + 2,
                                                               &f_shape, {0.5, -0.5});



    // ======================= Output =======================
    TFile* fout = new TFile(outfile, "RECREATE");
    fout->cd();
    h1_raw->Write();
    response->Hresponse()->Write("matrix1D");
    TH2D hM(M);
    hM.SetName("matrix1D_normalized");
    hM.Write();

    if (do_purity) {
        purity->Write();
        efficiency->Write();
    }

    cout << "writing trivial closure test hists" << endl;
    TDirectory* dir_trivial = fout->mkdir("trivial_closure");
    dir_trivial->cd();
    for (auto h: trivial_closure_out_matched_hists) h->Write();
    for (auto h: trivial_closure_out_matched_fromrms_hists) h->Write();
    for (auto h: trivial_closure_out_all_hists) h->Write();
    for (auto h: trivial_closure_out_matched_hists) cout << h->GetName() << endl;
    for (auto h: trivial_closure_out_all_hists) cout << h->GetName() << endl;

    for (auto h: trivial_closure_out_all_effpuroff_hists) h->Write();
    for (auto h: trivial_closure_out_all_onlyeffon_hists) h->Write();
    for (auto h: trivial_closure_out_all_onlypuron_hists) h->Write();

    cout << "writing refolding closure test hists" << endl;
    TDirectory* dir_refold = fout->mkdir("refolding_closure");
    dir_refold->cd();
    for (auto h : refolding_out_hists) h->Write();
    // for (auto h : refolding_out_hists_noeffpur) h->Write();

    cout << "writing statistical closure test hists" << endl;
    TDirectory* dir_stat = fout->mkdir("statistical_closure");
    dir_stat->cd();
    for (auto h : stat_closure_hists) h->Write();

    cout << "writing shape closure test hists" << endl;
    TDirectory* dir_shape = fout->mkdir("shape_closure");
    dir_shape->cd();
    for (auto h : shape_closure_hists) h->Write();

    cout << "writing prior closure test hists" << endl;
    TDirectory* dir_prior = fout->mkdir("prior_closure");
    dir_prior->cd();
    // for (auto h : prior_closure_hists) h->Write();
    
    fout->Close();
    cout << "Wrote output to " << outfile << endl;
}
