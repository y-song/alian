#include <iostream>
#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TMatrixD.h>
#include <TSystem.h>
#include <TString.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// Inspired from: https://github.com/ezradlesser/pyjetty/blob/09e41284e6bd7f289e457875d9e15224eaa34dc2/pyjetty/alice_analysis/analysis/user/substructure/roounfold_obs.py#L1233

// USAGE
// don't forget to run roounfold_preload_on first! -- prob not needed anymore
// root -q unfolding_tests.C




// Flattened 1-based bin index, matching RooUnfold's convention for 3D histograms:
//   flat = (ix-1) + nx*(iy-1) + nx*ny*(iz-1)   (0-based)
// and the response TH2 bin = flat + 1.
static inline int flat_bin_3d(int ix, int iy, int iz, int nx, int ny) {
    return ix + nx * (iy - 1) + nx * ny * (iz - 1);
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

// static TH3D* th3_manual_refold(RooUnfoldResponse* resp, TH3* h3_raw, TH3* h3_unfolded_truth, TString name) {
//     TH2* h_mig   = resp->Hresponse();
//     return th3_manual_refold_from_raw_migration(h_mig, h3_raw, h3_unfolded_truth, h3_unfolded_truth, name);
// }



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

/*
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

void shape_closure_test(RooUnfoldResponse* response, TH2* RM_2D, TF1* w) {
    cout << "Doing closure test now" << endl;
    const int nr = RM_2D->GetNbinsX(), nt = RM_2D->GetNbinsY();
    TH1D* t_rw = (TH1D*) response->Htruth()->Clone("t_rw");  t_rw->SetDirectory(0);
    TH1D* r_rw = RM_2D->ProjectionX("r_rw", 1, nt);          r_rw->SetDirectory(0);
    r_rw->Reset();
    for (int j = 1; j <= nt; ++j) {
        double wj = w->Eval(RM_2D->GetYaxis()->GetBinCenter(j));
        t_rw->SetBinContent(j, t_rw->GetBinContent(j) * wj);
        for (int i = 1; i <= nr; ++i)
            r_rw->SetBinContent(i, r_rw->GetBinContent(i) + RM_2D->GetBinContent(i, j) * wj);
    }
    for (int it : {1, 3, 5, 10, 20, 30}) {
        RooUnfoldBayes unf(response, r_rw, it);
        unf.SetVerbose(0);
        TH1* u = unf.Hunfold(RooUnfold::kNoError);
        TH1D* rf = th1_manual_refold_from_raw_migration(RM_2D, r_rw, t_rw, u, TString::Format("rf_sc%d", it));
        printf("iter %2d  unf/true:", it);
        for (int j = 1; j <= nt; ++j) printf(" %6.3f", u->GetBinContent(j) / t_rw->GetBinContent(j));
        printf("   refold/input:");
        for (int i = 1; i <= nr; ++i) printf(" %6.3f", rf->GetBinContent(i) / r_rw->GetBinContent(i));
        printf("\n");
        delete u; delete rf;
    }
}
*/



void unfolding_tests(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root",
                       const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/60320414/AnalysisResultsMerged_groomedbins.root",
                       const TString& unfolded_file="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded.root",
                       const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded_tests.root",
                       int iter=5, bool do_purity=true) { 

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

    // ======================= Output =======================
    TFile* fout = new TFile(outfile, "RECREATE");
    TDirectory* dir_refold = fout->GetDirectory("refolding_closure");
    if (!dir_refold) dir_refold = fout->mkdir("refolding_closure");

    
    std::vector<std::string> objects = {"AA", "AB", "BB", "rad"};
    
    for (const auto& obj : objects) {
        TString resp_name = TString::Format("groomed_%s_roounfold_response", obj.c_str());
        RooUnfoldResponse* response = (RooUnfoldResponse*) f_rm->Get(resp_name);
        
        if (!response) {
            std::cerr << "Error: " << resp_name << " not found. File contents:" << std::endl;
            f_rm->ls();
            return;
        }
    

        // Raw: prefer the copy written by unfold1D.C (exactly what was unfolded)
        TH3D* h3_raw = (TH3D*) f_rawdata->Get(TString::Format("%s_sd0.1_raw", obj.c_str()));
        if (!h3_raw) { std::cerr << "Error: raw histogram not found" << std::endl; return; }
        h3_raw = (TH3D*) h3_raw->Clone(TString::Format("h3_raw_%s", obj.c_str()));
        h3_raw->SetDirectory(0);

        

        // ======================= Calculate Purity / efficiency (don't apply until later) =======================
        cout << "purity correction is running (backwards)" << endl;
        TH3D* h3_rec_num           = (TH3D*) f_rm->Get(TString::Format("groomed_%s_reco_matched", obj.c_str()));
        TH3D* h3_rec_den = (TH3D*) f_rm->Get(TString::Format("groomed_%s_reco_all", obj.c_str()));
        h3_rec_num->Sumw2(1);
        h3_rec_den->Sumw2(1);
        TH3D* purity = (TH3D*) h3_rec_num->Clone(TString::Format("purity_%s", obj.c_str()));
        purity->SetDirectory(0);
        // purity->Divide(h3_rec_den); // (reco_matched / reco_all)
        purity->Divide(h3_rec_num, h3_rec_den, 1., 1., "B"); //h = (c1 * h1) / (c2 * h2), bin by bin, with binomial errors


        /*
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
        */

        // ======================= REFOLD ! =======================
        std::vector<TH3*> refolding_out_hists = refolding_test(obj, f_unf, response, h3_raw, iter, purity, do_purity);

        // // check: do the version with no eff/pur
        // std::vector<TH1*> refolding_out_hists_noeffpur = refolding_test(f_unf_noeffpur, RM_2D, h1_raw, iter, efficiency, purity, false, false, "NoEffPur_");

        /*
        // ======================= SHAPE CLOSURE TEST ! =======================
        TF1 w("w", "pow(x/20., [0])", 10, 500); w.SetParameter(0, 0.2);
        shape_closure_test(response, RM_2D, &w);
        */

    
        //Write objects here
        fout->cd();
        h3_raw->Write();
        response->Hresponse()->Write(TString::Format("matrix3D_%s", obj.c_str()));
        // TH2D hM(M);
        // hM.SetName("matrix3D_normalized");
        // hM.Write();

        if (do_purity) {
            purity->Write();
        }

        // cout << "writing trivial closure test hists" << endl;
        // TDirectory* dir_trivial = fout->mkdir("trivial_closure");
        // dir_trivial->cd();
        // for (auto h: trivial_closure_out_matched_hists) h->Write();
        // for (auto h: trivial_closure_out_matched_fromrms_hists) h->Write();
        // for (auto h: trivial_closure_out_all_hists) h->Write();

        // for (auto h: trivial_closure_out_all_effpuroff_hists) h->Write();
        // for (auto h: trivial_closure_out_all_onlyeffon_hists) h->Write();
        // for (auto h: trivial_closure_out_all_onlypuron_hists) h->Write();

        cout << "writing refolding closure test hists" << endl;
        dir_refold->cd();
        for (auto h : refolding_out_hists) h->Write();
    }

    
    fout->Close();
    cout << "Wrote output to " << outfile << endl;
}
