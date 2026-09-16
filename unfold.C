#include <iostream>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>
#include <string>
#include <vector>
// #include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldBayes.h>
// #include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldResponse.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/blianggi/mypyjetty/heppy/external/roounfold/roounfold-current/include/RooUnfoldResponse.h>

// usage

// FULLSIM / DATA
// root -q "unfold.C(\"/global/cfs/cdirs/alice/kdevero/pp_fullsim_enc/26200088/merged.root\", \"/global/cfs/cdirs/alice/kdevero/pp_enc/25886413/merged.root\", \"unfolded_fr.root\", 5, true)"

// root -q unfold.C

void unfold(const TString& rm_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/57911278/response_groomed_forunfolding_merged.root", 
            const TString& data_file="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/57879247/AnalysisResultsMerged_groomedbins.root",
            const TString& outfile="/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/unfolded.root", 
            int iter=9, bool do_purity=true) {
    
    
    // Set ROOT to batch mod
    gROOT->SetBatch(true);
    TH1::SetDefaultSumw2(true); // covers both 1D and 3D histogram since TH2 inherits from TH1

    // Open inputs
    TFile* f_data = new TFile(data_file);
    TFile* f = new TFile(rm_file);

    // Create output file ONCE, outside the loop
    TFile* fout = new TFile(outfile, "RECREATE");

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

        // purity correction
        TH3D* purity;
        if (do_purity)
        {
            TH3D* h3_reco           = (TH3D*) f->Get(TString::Format("groomed_%s_reco", o));
            TH3D* h3_reco_unmatched = (TH3D*) f->Get(TString::Format("groomed_%s_reco_unmatched", o));
            h3_reco->Sumw2(1);
            h3_reco_unmatched->Sumw2(1);
            purity = (TH3D*) h3_reco->Clone(TString::Format("purity_%s", o));
            purity->Divide(h3_reco_unmatched); // (reco / reco_unmatched)
            h3_raw->Multiply(purity);
        }
        

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

    }

    // Write output file
    fout->Write();
    fout->Close();
}