void test()
{
    gStyle->SetOptStat(0);
    
    TFile *f = new TFile("/rstorage/youqi/1934682/AnalysisResultsFinal.root", "READ");
    TH2D *h = (TH2D *)f->Get("track_phi_eta");
    h->Scale(1.0/h->Integral());
    h->SetMinimum(0);
    h->SetMaximum(0.0003);

    TCanvas c = new TCanvas();
    h->Draw("colz");
    c.SaveAs("output/test.pdf");   
}
