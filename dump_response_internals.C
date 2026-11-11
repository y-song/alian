// dump_response_internals.C  — run with your usual RooUnfold-enabled root:
// root -l -b -q dump_response_internals.C
void dump_response_internals() {
    auto f = TFile::Open("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/rms/60403841/response_groomed_forunfolding_merged.root");
    auto out = TFile::Open("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/unfolding/response_internals.root", "RECREATE");
    for (auto* lab : {"AA", "AB", "BB", "rad"}) {
        auto* resp = (RooUnfoldResponse*) f->Get(TString::Format("groomed_%s_roounfold_response", lab));
        out->WriteObject(resp->Htruth(),    TString::Format("Htruth_%s", lab));
        out->WriteObject(resp->Hmeasured(), TString::Format("Hmeasured_%s", lab));
    }
    out->Close();
}