import os
import pandas as pd
import ROOT
import matplotlib.pyplot as plt

def main():
    # Configuration
    base_path = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/54598637"
    out_root_path = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/Data_AllJets.root"

    if os.path.exists(out_root_path):
        print(f"Loading histogram from {out_root_path}...")
        f_in = ROOT.TFile(out_root_path, "READ")
        h_pt = f_in.Get("h_pt")
        h_pt.SetDirectory(0)
        f_in.Close()
    else:
        # Histogram setup: 0-500 GeV in bins of 1 GeV
        # Range [0, 500) with 500 bins.
        # Bin 1: [0, 1), ..., Bin 500: [499, 500).
        # Overflow (Bin 501): [500, inf).
        h_pt = ROOT.TH1D("h_pt", "Jet p_{T} Spectrum;p_{T} [GeV/c];Counts", 500, 0, 500)

        print("Reading files...")
        count_files = 0
        for i in range(1, 171):
            file_path = os.path.join(base_path, str(i), "DataJetsForAnalysis.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    if 'jet_pt' in df.columns:
                        pts = df['jet_pt'].to_numpy()
                        for pt in pts:
                            h_pt.Fill(pt)
                    count_files += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Processed {count_files} files.")

        # Save the histogram to a ROOT file
        out_file = ROOT.TFile(out_root_path, "RECREATE")
        h_pt.Write()
        out_file.Close()
        print(f"\nHistogram saved to {out_root_path}")

    # --- Counts for specific bins ---
    # Ranges: (lo, hi)
    ranges = [
        (10, 20), (20, 40), (40, 60), (60, 80), (80, 100),
        (100, 120), (120, 150), (150, 200), (200, 500), (500, float('inf'))
    ]

    print("\nJet pT Counts:")
    print("-" * 30)

    for (lo, hi) in ranges:
        if hi == float('inf'):
            # Everything >= 500 goes into the overflow bin (index 501)
            count = h_pt.GetBinContent(501)
            print(f"{int(lo):3d}+  GeV: {int(count)}")
        else:
            # Bin index for lo is int(lo) + 1
            # Bin index for hi is int(hi)
            count = 0
            for b in range(int(lo) + 1, int(hi) + 1):
                if b <= 500:
                    count += h_pt.GetBinContent(b)
            print(f"{int(lo):3d}-{int(hi):3d} GeV: {int(count)}")

    # Save a plot of the spectrum
    canvas = ROOT.TCanvas("c1", "Jet pT Spectrum", 800, 600)
    canvas.SetLogy()

    h_pt.SetLineColor(ROOT.kBlack)
    h_pt.Draw("HIST")

    # Ensure the overflow is visible or just plot the main range
    canvas.SaveAs("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/data_checks/study_jetpt_data.pdf")
    print("\nSpectrum plot saved as study_jetpt_data.pdf")

    # --- Integrated yield above threshold ---
    # For each pT threshold (bin low edge), compute N(pT > threshold),
    # i.e. the sum of that bin and all bins above it, including overflow.
    n_bins = h_pt.GetNbinsX()
    overflow = h_pt.GetBinContent(n_bins + 1)

    h_integrated = h_pt.Clone("h_integrated")
    h_integrated.Reset()
    h_integrated.SetTitle(
        "Integrated Jet Yield;p_{T} threshold [GeV/c];N(p_{T} > threshold)"
    )

    for b in range(1, n_bins + 1):
        integral = h_pt.Integral(b, n_bins) + overflow
        h_integrated.SetBinContent(b, integral)

    canvas2 = ROOT.TCanvas("c2", "Integrated Yield", 800, 600)
    canvas2.SetLogy()
    canvas2.SetGrid()

    h_integrated.SetLineColor(ROOT.kBlack)
    h_integrated.Draw("HIST")

    text_box = ROOT.TPaveText(0.45, 0.67, 0.77, 0.8, "NDC")
    text_box.AddText("ALICE pp #sqrt{s} =5.36 TeV")
    text_box.AddText("LHC24 ppref pass 1, JE derived")
    text_box.AddText("anti-k_{T} R=0.4, |#eta_{jet}|<0.5")
    text_box.SetFillColor(0)          # White background
    text_box.SetBorderSize(0)         # Thin border line
    text_box.SetTextAlign(12)         # Center alignment for text (horizontal and vertical)
    text_box.Draw()

    canvas2.SaveAs("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/data_checks/study_jetpt_integrated.pdf")
    print("Integrated yield plot saved as study_jetpt_integrated.pdf")

    canvas2.SetLogx()
    h_integrated.GetXaxis().SetRangeUser(5, 500)
    canvas2.SaveAs("/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/data_checks/study_jetpt_integrated_logx.pdf")
    print("Integrated yield plot saved as study_jetpt_integrated_logx.pdf")


if __name__ == "__main__":
    main()
