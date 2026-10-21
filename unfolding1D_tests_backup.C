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
// don't forget to run roounfold_preload_on first!
// root -q unfolding1D_tests.C

bool sameBinning(const TAxis* a, const TAxis* b) {
    if (a->GetNbins() != b->GetNbins()) return false;
    for (int i = 1; i <= a->GetNbins() + 1; ++i)
        if (std::fabs(a->GetBinLowEdge(i) - b->GetBinLowEdge(i)) > 1e-6) return false;
    return true;
}

bool close(double a, double b, double tol = 1e-6) {
    return b != 0 && std::fabs(a - b) / std::fabs(b) < tol;
}

void unfolding1D_tests(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/58919321/response_groomed_forunfolding_merged.root",
                       const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/58416144/AnalysisResultsMerged_groomedbins.root",
                       const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D_tests.root",
                       const TString& resp_name="roounfold_response_1D_groomed_oldversion",
                       int iter=9, bool do_purity=true) {

    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true);
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // ======================= Inputs =======================
    TFile* f_data = new TFile(data_file);
    TFile* f_rm   = new TFile(rm_file);
    if (f_data->IsZombie() || f_rm->IsZombie()) { std::cerr << "Error: cannot open input files" << std::endl; return; }

    TH1D* h1_raw = (TH1D*) f_data->Get("groomed_sd0.1_jet_pt_raw1D");
    if (!h1_raw) { std::cerr << "Error: raw histogram not found" << std::endl; return; }
    h1_raw->SetDirectory(0);

    RooUnfoldResponse* response = (RooUnfoldResponse*) f_rm->Get(resp_name);
    if (!response) {
        std::cerr << "Error: " << resp_name << " not found. File contents:" << std::endl;
        f_rm->ls();
        return;
    }

    // ======================= Validate response =======================
    TH1* hMeas = response->Hmeasured();
    TH1* hTrue = response->Htruth();
    TH2* hResp = response->Hresponse();
    cout << "Response: " << resp_name << endl;
    cout << "Hresponse X [" << hResp->GetXaxis()->GetXmin() << ", " << hResp->GetXaxis()->GetXmax() << "], "
         << "Y [" << hResp->GetYaxis()->GetXmin() << ", " << hResp->GetYaxis()->GetXmax() << "]" << endl;
    cout << "Integrals in range / incl. flow:" << endl;
    cout << "  Hmeasured = " << hMeas->Integral() << " / " << hMeas->Integral(0, -1) << endl;
    cout << "  Htruth    = " << hTrue->Integral() << " / " << hTrue->Integral(0, -1) << endl;
    cout << "  Hresponse = " << hResp->Integral() << " / " << hResp->Integral(0, -1, 0, -1) << endl;

    if (!sameBinning(hResp->GetXaxis(), h1_raw->GetXaxis()) || !sameBinning(hResp->GetYaxis(), hTrue->GetXaxis())) {
        std::cerr << "Error: Hresponse binning doesn't match raw/truth binning -> this response can't be used" << std::endl;
        return;
    }
    if (hResp->Integral() <= 0) {
        std::cerr << "Error: Hresponse has no content in range -> this response can't be used" << std::endl;
        return;
    }

    TMatrixD M = response->Mresponse();
    int nMeas = M.GetNrows(), nTruth = M.GetNcols();
    cout << "Mresponse: " << nMeas << " x " << nTruth << ", sum = " << M.Sum() << endl;
    cout << "nMeas=" << response->GetNbinsMeasured()
     << " nTruth=" << response->GetNbinsTruth()
     << " overflow=" << response->UseOverflowStatus()
     << " hunf bins=" << hunf->GetNbinsX() << endl;

    // ======================= Kinematic purity (truncation fakes) =======================
    // det in range but truth out of range: subtracted by RooUnfoldBayes, must be added back after folding
    TH1D* hProjX = hResp->ProjectionX("hResp_projX_inrange", 1, hResp->GetNbinsY());
    hProjX->SetDirectory(0);
    TH1D* kinPurity = (TH1D*) hProjX->Clone("kinematic_purity");
    kinPurity->SetDirectory(0);
    kinPurity->Divide(hMeas);
    cout << "Kinematic purity per measured bin:";
    for (int k = 1; k <= kinPurity->GetNbinsX(); ++k) cout << " " << kinPurity->GetBinContent(k);
    cout << endl;

    // ======================= Matching purity / efficiency =======================
    TH1D* purity = nullptr;
    TH1D* efficiency = nullptr;
    if (do_purity) {
        TH1D* rec_num = (TH1D*) f_rm->Get("jet_match_rec_pur_num_groomed");
        TH1D* rec_den = (TH1D*) f_rm->Get("jet_all_rec_pur_den_groomed");
        TH1D* gen_num = (TH1D*) f_rm->Get("jet_match_gen_eff_num_groomed");
        TH1D* gen_den = (TH1D*) f_rm->Get("jet_all_gen_eff_den_groomed");
        if (!rec_num || !rec_den || !gen_num || !gen_den) {
            std::cerr << "Error: purity/efficiency histograms not found in RM file" << std::endl;
            return;
        }
        purity = (TH1D*) rec_num->Clone("purity");
        purity->SetDirectory(0);
        purity->Divide(rec_num, rec_den, 1., 1., "B");
        efficiency = (TH1D*) gen_num->Clone("efficiency");
        efficiency->SetDirectory(0);
        efficiency->Divide(gen_num, gen_den, 1., 1., "B");
    }

    // Raw corrected for matching fakes: this is what the matched-only response describes
    TH1D* h1_raw_corr = (TH1D*) h1_raw->Clone("h1_raw_puritycorrected");
    h1_raw_corr->SetDirectory(0);
    if (purity) h1_raw_corr->Multiply(purity);

    // Manual fold: M * truth (no fakes)
    auto manualFold = [&](TH1* hTruthIn, const TString& name) -> TH1D* {
        TH1D* h = (TH1D*) h1_raw->Clone(name);
        h->SetDirectory(0);
        h->Reset();
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
    };

    // ======================= Unfold + refold =======================
    std::vector<TH1*> out_hists;

    for (int i = 1; i <= iter; ++i) {
        cout << "------------------------------------------" << endl;
        cout << "iter " << i << endl;

        // Unfold (matched truth level; kinematic efficiency already corrected via Htruth normalization)
        RooUnfoldBayes unfold(response, h1_raw_corr, i);
        TH1* hunf = (TH1*) unfold.Hunfold(errorTreatment)->Clone(TString::Format("h1_jetpt_groomed_unfolded%d", i));
        hunf->SetDirectory(0);

        // Final physics result: correct for matching efficiency
        TH1* hfinal = (TH1*) hunf->Clone(TString::Format("h1_jetpt_groomed_unfolded_final%d", i));
        hfinal->SetDirectory(0);
        if (efficiency) hfinal->Divide(efficiency);

        // Manual folds without / with kinematic fakes
        TH1D* hmanual = manualFold(hunf, TString::Format("h1_jetpt_groomed_refolded_manual%d", i));
        TH1D* hmanualKin = (TH1D*) hmanual->Clone(TString::Format("h1_jetpt_groomed_refolded_manual_withfakes%d", i));
        hmanualKin->SetDirectory(0);
        hmanualKin->Divide(kinPurity);

        // ApplyToTruth
        cout << "Applying ApplyToTruth..." << endl;
        TH1* happ = response->ApplyToTruth(hunf);
        double iA = happ ? happ->Integral() : 0.;
        TH1* hfold = nullptr;
        if (happ && close(iA, hmanual->Integral())) {
            cout << "ApplyToTruth matches M*truth (no fakes) -> adding kinematic fakes back" << endl;
            hfold = (TH1*) happ->Clone();
            hfold->Divide(kinPurity);
        } else if (happ && close(iA, hmanualKin->Integral())) {
            cout << "ApplyToTruth already includes kinematic fakes" << endl;
            hfold = (TH1*) happ->Clone();
        } else {
            std::cerr << "ApplyToTruth (" << iA << ") disagrees with manual fold ("
                      << hmanual->Integral() << " / " << hmanualKin->Integral()
                      << " with fakes) -> using manual fold with fakes" << std::endl;
            hfold = (TH1*) hmanualKin->Clone();
        }
        hfold->SetName(TString::Format("h1_jetpt_groomed_refolded%d", i));
        hfold->SetTitle(TString::Format("Refolded, iter %d", i));
        hfold->SetDirectory(0);

        // Closure: refolded vs. purity-corrected raw (should be ~1)
        TH1* hratio = (TH1*) hfold->Clone(TString::Format("h1_jetpt_groomed_refolded_over_raw%d", i));
        hratio->SetTitle(TString::Format("Refolded / raw (purity-corrected), iter %d", i));
        hratio->SetDirectory(0);
        hratio->Divide(h1_raw_corr);

        cout << "unfolded = " << hunf->Integral()
             << ", final = " << hfinal->Integral()
             << ", refolded = " << hfold->Integral()
             << ", raw (purity-corr.) = " << h1_raw_corr->Integral() << endl;
        cout << "refolded/raw per bin:";
        for (int k = 1; k <= hratio->GetNbinsX(); ++k) cout << " " << hratio->GetBinContent(k);
        cout << endl;

        out_hists.insert(out_hists.end(), {hunf, hfinal, hfold, hmanual, hmanualKin, hratio});
    }

    // ======================= Output =======================
    TFile* fout = new TFile(outfile, "RECREATE");
    h1_raw->Write("h1_raw");
    h1_raw_corr->Write();
    hResp->Write("matrix1D");
    TH2D hM(M);
    hM.SetName("matrix1D_normalized");
    hM.Write();
    kinPurity->Write();
    response->Write("roounfold_response_1D_groomed_used");
    for (auto h : out_hists) h->Write();
    if (purity)     purity->Write();
    if (efficiency) efficiency->Write();
    fout->Close();
    cout << "Wrote output to " << outfile << endl;
}


// void unfolding1D_tests(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/58919321/response_groomed_forunfolding_merged.root",
//                        const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/58416144/AnalysisResultsMerged_groomedbins.root",
//                        const TString& unfolded_file="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D.root",
//                        const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded1D_tests.root",
//                        int iter=9, bool do_purity=false) {

//     gROOT->SetBatch(true);
//     TH1::SetDefaultSumw2(true);

//     // ---------------- Inputs ----------------
//     TFile* f_data = new TFile(data_file);
//     TFile* f      = new TFile(unfolded_file);
//     TFile* f_rm   = new TFile(rm_file);

//     TH1D* h1_raw = (TH1D*) f_data->Get("groomed_sd0.1_jet_pt_raw1D");
//     if (!h1_raw) { std::cerr << "Error: raw histogram not found" << std::endl; return; }
//     h1_raw->SetDirectory(0);
//     cout << "get histograms" << endl;

//     RooUnfoldResponse* response1D = (RooUnfoldResponse*) f_rm->Get("roounfold_response_1D_groomed");
//     if (!response1D) {
//         std::cerr << "Error: Could not find response object 'roounfold_response_1D_groomed' in " << rm_file << std::endl;
//         return;
//     }

//     // ---------------- Diagnostics ----------------
//     TH1* hMeas = response1D->Hmeasured();
//     TH1* hTrue = response1D->Htruth();
//     TH2* hResp = response1D->Hresponse();
//     cout << "Response: nMeasured = " << response1D->GetNbinsMeasured()
//          << ", nTruth = " << response1D->GetNbinsTruth() << endl;
//     cout << "h1_raw bins = " << h1_raw->GetNbinsX() << endl;
//     cout << "Hmeasured integral = " << (hMeas ? hMeas->Integral() : -1) << endl;
//     cout << "Htruth    integral = " << (hTrue ? hTrue->Integral() : -1) << endl;
//     cout << "Hresponse integral = " << (hResp ? hResp->Integral() : -1) << endl;
//     if (!hResp) { std::cerr << "Error: Hresponse is null" << std::endl; return; }

//     // ---------------- Normalized response matrix ----------------
//     // M(k,j) = P(measured bin k | truth bin j)
//     TMatrixD M = response1D->Mresponse();
//     cout << "Mresponse sum = " << M.Sum() << endl;
//     if (M.Sum() == 0) {
//         cout << "Mresponse is empty -> building normalized matrix from Hresponse/Htruth" << endl;
//         int nMeasH  = hResp->GetNbinsX();
//         int nTruthH = hResp->GetNbinsY();
//         M.ResizeTo(nMeasH, nTruthH);
//         for (int j = 0; j < nTruthH; ++j) {
//             double t = hTrue ? hTrue->GetBinContent(j + 1) : 0.;
//             if (t <= 0) t = hResp->Integral(0, -1, j + 1, j + 1);  // fallback: projection (ignores misses)
//             for (int k = 0; k < nMeasH; ++k)
//                 M(k, j) = (t > 0) ? hResp->GetBinContent(k + 1, j + 1) / t : 0.;
//         }
//     }
//     int nMeas  = M.GetNrows();
//     int nTruth = M.GetNcols();
//     cout << "M: " << nMeas << " x " << nTruth << ", sum = " << M.Sum() << endl;

//     // ---------------- Purity / efficiency ----------------
//     // Look in the RM file first, then the unfolded file
//     auto getHist = [&](const char* name) -> TH1D* {
//         TH1D* h = (TH1D*) f_rm->Get(name);
//         if (!h) h = (TH1D*) f->Get(name);
//         if (!h) std::cerr << "Error: could not find " << name << std::endl;
//         return h;
//     };

//     TH1D* purity = nullptr;
//     TH1D* efficiency = nullptr;
//     if (do_purity) {
//         cout << "Undoing purity and efficiency corrections for refolding test" << endl;
//         TH1D* h1_reco           = getHist("jet_match_rec_pur_num_groomed");
//         TH1D* h1_reco_unmatched = getHist("jet_all_rec_pur_den_groomed");
//         TH1D* h1_gen            = getHist("jet_match_gen_eff_num_groomed");
//         TH1D* h1_gen_unmatched  = getHist("jet_all_gen_eff_den_groomed");
//         if (!h1_reco || !h1_reco_unmatched || !h1_gen || !h1_gen_unmatched) return;

//         purity = (TH1D*) h1_reco->Clone("purity");
//         purity->SetDirectory(0);
//         purity->Divide(h1_reco, h1_reco_unmatched, 1., 1., "B");      // matched reco / all reco

//         efficiency = (TH1D*) h1_gen->Clone("efficiency");
//         efficiency->SetDirectory(0);
//         efficiency->Divide(h1_gen, h1_gen_unmatched, 1., 1., "B");    // matched gen / all gen

//         if (purity->GetNbinsX() != nMeas)      std::cerr << "Warning: purity binning != measured binning" << std::endl;
//         if (efficiency->GetNbinsX() != nTruth) std::cerr << "Warning: efficiency binning != truth binning" << std::endl;
//     }

//     // ---------------- Refolding ----------------
//     std::vector<TH1D*> unfolded_hists, folded_hists, ratio_hists;

//     for (int i = 1; i <= iter; ++i) {
//         cout << "------------------------------------------" << endl;
//         cout << "iter " << i << endl;
//         TString histName = TString::Format("Baysian_Unfolded1Diter%d", i);
//         TH1D* h1_unfolded_in = (TH1D*) f->Get(histName);
//         if (!h1_unfolded_in) {
//             std::cerr << "Error: Could not find histogram " << histName << "!" << std::endl;
//             continue;
//         }

//         // Work on a copy so the input isn't modified
//         TH1D* h1_unfolded = (TH1D*) h1_unfolded_in->Clone(TString::Format("h1_jetpt_groomed_unfolded%d", i));
//         h1_unfolded->SetDirectory(0);
//         cout << "h1_unfolded integral = " << h1_unfolded->Integral() << endl;

//         // Undo efficiency correction: truth level -> multiply back in
//         if (do_purity && efficiency) h1_unfolded->Multiply(efficiency);

//         // Fold: reco = M * truth
//         TVectorD vTruth(nTruth), vTruthErr2(nTruth);
//         for (int j = 0; j < nTruth; ++j) {
//             vTruth[j]     = h1_unfolded->GetBinContent(j + 1);
//             vTruthErr2[j] = std::pow(h1_unfolded->GetBinError(j + 1), 2);
//         }
//         TVectorD vReco = M * vTruth;

//         TH1D* h1_folded = (TH1D*) h1_raw->Clone(TString::Format("h1_jetpt_groomed_refolded%d", i));
//         h1_folded->SetDirectory(0);
//         h1_folded->Reset();
//         for (int k = 0; k < nMeas; ++k) {
//             double err2 = 0;
//             for (int j = 0; j < nTruth; ++j) err2 += M(k, j) * M(k, j) * vTruthErr2[j];
//             h1_folded->SetBinContent(k + 1, vReco[k]);
//             h1_folded->SetBinError(k + 1, std::sqrt(err2));
//         }

//         // Undo purity correction: reco level -> divide back out
//         if (do_purity && purity) h1_folded->Divide(purity);

//         // Refolded / raw ratio
//         TH1D* h1_ratio = (TH1D*) h1_folded->Clone(TString::Format("h1_jetpt_groomed_refolded_over_raw%d", i));
//         h1_ratio->SetDirectory(0);
//         h1_ratio->Divide(h1_raw);

//         cout << "h1_folded " << h1_folded->GetName()
//              << ", integral = " << h1_folded->Integral()
//              << " (raw integral = " << h1_raw->Integral() << ")" << endl;

//         unfolded_hists.push_back(h1_unfolded);
//         folded_hists.push_back(h1_folded);
//         ratio_hists.push_back(h1_ratio);
//     }

//     // ---------------- Output ----------------
//     TFile* fout = new TFile(outfile, "RECREATE");
//     fout->cd();

//     h1_raw->Write("h1_raw");
//     hResp->Write("matrix1D");
//     TH2D hM(M);                       // normalized matrix as a TH2D
//     hM.SetName("matrix1D_normalized");
//     hM.Write();

//     for (auto h : unfolded_hists) h->Write();
//     for (auto h : folded_hists)   h->Write();
//     for (auto h : ratio_hists)    h->Write();

//     if (do_purity) {
//         if (purity)     purity->Write();
//         if (efficiency) efficiency->Write();
//     }

//     fout->Close();
//     cout << "Wrote output to " << outfile << endl;
// }