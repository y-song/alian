#include <iostream>
#include <TFile.h>
#include <TH1.h>
#include <TH3.h>
#include <TSystem.h>
#include <TString.h>
#include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldBayes.h>
#include </global/cfs/cdirs/alice/kdevero/RooUnfold/RooUnfold/build/RooUnfoldResponse.h>

// usage

// FULLSIM / DATA
// root -q "unfold.C(\"/global/cfs/cdirs/alice/kdevero/pp_fullsim_enc/26200088/merged.root\", \"/global/cfs/cdirs/alice/kdevero/pp_enc/25886413/merged.root\", \"unfolded_fr.root\", 5, true)"

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

void unfold(const TString& rm_file="preunfold.root", const TString& data_file="preunfold_data.root",
            const TString& outfile="unfolded.root", int iter=9, bool do_purity=true) {
    // Set ROOT to batch mod
    gROOT->SetBatch(true);

    // Error treatment for unfolding
    RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;

    // Open inputs
    // RM
    TFile* f = new TFile(rm_file);
    RooUnfoldResponse* response = (RooUnfoldResponse*) f->Get("response");

    // DATA
    TFile* f_data = new TFile(data_file);
    TH3D* h3_raw = (TH3D*) f_data->Get("raw");
    // do not close these, if think its because the program reads directly from the files and does NOT copy it into memory

    // purity correction
    TH3D* purity;
    if (do_purity)
    {
        TH3D* h3_reco = (TH3D*) f->Get("reco");
        TH3D* h3_reco_unmatched = (TH3D*) f->Get("reco_unmatched");
        h3_reco->Sumw2(1);
        h3_reco_unmatched->Sumw2(1);
        purity = (TH3D*) h3_reco->Clone("purity");
        purity->Divide(h3_reco_unmatched); // (reco / reco_unmatched)
        h3_raw->Multiply(purity);
    }
    

    // Create output file
    TFile* fout = new TFile(outfile, "RECREATE");

    for (int i = 1; i <= iter; ++i) {
        // Unfold the 3D histogram
        RooUnfoldBayes unfold(response, h3_raw, i);
        TH3* hunf = (TH3*)unfold.Hunfold(errorTreatment);
        // TH3* hfold = (TH3*)response->ApplyToTruth(hunf, "");

        cout<<"unfolded"<<endl;

        // Clone and name histograms
        TH3* htempUnf = (TH3*)hunf->Clone(TString::Format("Baysian_Unfoldediter%d", i));
        // TH3* htempFold = (TH3*)hfold->Clone(TString::Format("Baysian_Foldediter%d", i));

        // Write histograms to output file
        htempUnf->Write();
        // htempFold->Write();

        cout<<"written"<<endl;
    }

    // Make TH2Ds out of response matricies, add to output file
    TH2D* matrix = (TH2D*) response->Hresponse();
    matrix->SetName("matrix");
    matrix->Write();

    // write purity hist
    if (do_purity) purity->Write();
    h3_raw->Write(); // note this raw data is now purity-corrected, use it to compare with folded result

    // Write output file
    fout->Write();
    fout->Close();
}