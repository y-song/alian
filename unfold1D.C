#include <iostream>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>

// usage
// FULLSIM / DATA
// root -q "unfold1D.C(\"/global/cfs/cdirs/alice/kdevero/pp_fullsim_enc/26200088/merged.root\", \"/global/cfs/cdirs/alice/kdevero/pp_enc/25886413/merged.root\", \"unfolded_1D.root\", 5, false)"

// TEST
// root -q "unfold1D.C(\"./preunfold_rm1D_test.root\", \"./preunfold_data_test.root\", \"unfolded_1D_test.root\", 3, false)"

// TEST
// root -q "unfold1D.C(\"./output_mc/merged.root\", \"./output_data/AnalysisResults.root\", \"unfolded_1D_test.root\", 3, false)"

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

void unfold1D(const TString& rm_file="response_merged.root", const TString& data_file="preunfold_data.root",
            const TString& outfile="unfolded.root",
            int iter=9, bool do_purity=false) {
    // Set ROOT to batch mod
    gROOT->SetBatch(true);

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Open inputs
    // RM
    TFile* f = new TFile(rm_file);
    RooUnfoldResponse* response1D = (RooUnfoldResponse*) f->Get("response1D"); // response1D, reco1D_gen1D

    // DATA
    TFile* f_data = new TFile(data_file);
    TH1D* h1_raw = (TH1D*) f_data->Get("groomed_sd0.1_jet_pt_raw1D"); //raw1D
    // do not close these, if think its because the program reads directly from the files and does NOT copy it into memory

    // purity and efficiency correction
    TH1D* purity;
    TH1D* efficiency;
    if (do_purity)
    {
        cout<<"purity correction is running"<<endl;
        TH1D* h1_reco = (TH1D*) f->Get("reco1D");
        TH1D* h1_reco_unmatched = (TH1D*) f->Get("reco1D_unmatched");
        h1_reco->Sumw2(1);
        h1_reco_unmatched->Sumw2(1);
        purity = (TH1D*) h1_reco->Clone("purity");
        purity->Divide(h1_reco_unmatched); // (reco / reco_unmatched)
        // h1_raw->Multiply(purity);

        TH1D* h1_gen = (TH1D*) f->Get("gen1D");
        TH1D* h1_gen_unmatched = (TH1D*) f->Get("gen1D_unmatched");
        h1_gen->Sumw2(1);
        h1_gen_unmatched->Sumw2(1);
        efficiency = (TH1D*) h1_gen->Clone("efficiency");
        efficiency->Divide(h1_gen_unmatched);
        // h1_raw->Divide(efficiency);
    }

    // Create output file
    TFile* fout = new TFile(outfile, "RECREATE");

    for (int i = 1; i <= iter; ++i) {
        // Unfold the 3D histogram
        RooUnfoldBayes unfold1D(response1D, h1_raw, i);
        TH1* hunf1D = (TH1*)unfold1D.Hunfold(errorTreatment);
        // TH1* hfold1D = response1D->ApplyToTruth (hunf1D, "");

        // Clone and name histograms
        TH1* htempUnf1D = (TH1*)hunf1D->Clone(TString::Format("Baysian_Unfolded1Diter%d", i));
        // TH1* htempFold1D = (TH1*)hfold1D->Clone(TString::Format("Baysian_Folded1Diter%d", i));
        
        // Write histograms to output file
        htempUnf1D->Write();
        // htempFold1D->Write(); 
    }

    // Make TH2Ds out of response matricies, add to output file
    TH2D* matrix1D = (TH2D*) response1D->Hresponse();
    matrix1D->SetName("matrix1D");
    matrix1D->Write();

    if (do_purity)
    {
        purity->Write();
        efficiency->Write();
    }
    h1_raw->Write(); // note this raw data is now purity-corrected, use it to compare with folded result

    // Write output file
    fout->Write();
    fout->Close();
}