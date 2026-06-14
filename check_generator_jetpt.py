import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

target_jet_pts = [50, 100, 200, 500]

slurm_base   = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/slurm-output"
parquet_base = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550"
herwig_base  = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/storage/herwig/1006458"
output_dir   = "/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/pythia_vs_herwig"

herwig_json  = "/global/cfs/cdirs/alice/blianggi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/jse/herwig_scale_factors.json"

# File index ranges for each pt bin in the Pythia slurm outputs
slurm_ranges = {
    50:  (1,   100),
    100: (101, 200),
    200: (201, 300),
    500: (301, 400),
}

BIN_WIDTH = 1.0  # GeV


def plot_jet_pt_counts():
    for i, target_jet_pt in enumerate(target_jet_pts):

        print("analyzing jet pt:", target_jet_pt, "GeV")

        pythia_path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/{target_jet_pt}gev/JetsForAnalysisCombined.parquet"
        herwig_path = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/storage/herwig/1006458/{target_jet_pt}gev/JetsForAnalysisCombined.parquet"
                    
        # 1. Load the Parquet files
        # Note: You may need to install 'fastparquet' or 'pyarrow'
        df_pythia = pd.read_parquet(pythia_path, columns=['jet_pt', 'jet_id'])
        df_herwig = pd.read_parquet(herwig_path, columns=['jet_pt', 'jet_id'])

        # 2. Setup the figure
        plt.figure(figsize=(10, 7))

        # Define common histogram settings
        # Based on your .describe(), p_T is around 100.
        # We use density=True to compare shapes if the file sizes differ.
        hist_settings = {
            'bins': target_jet_pt*2+1, 
            'range': (0, target_jet_pt*2+1), 
            'alpha': 0.6,
            # 'density': True, # This normalizes the histogram
            'histtype': 'step', # 'step' is standard for HEP plots
            'linewidth': 2
        }
        print("Plotting histograms with settings:", hist_settings)
        # 3. Plot Pythia (Blue)
        plt.hist(df_pythia['jet_pt'], color='blue', label='Pythia', **hist_settings)

        # 4. Plot Herwig (Red)
        plt.hist(df_herwig['jet_pt'], color='red', label='Herwig', **hist_settings)

        # 5. Labeling and Aesthetics
        plt.title('Jet $p_T$ Distribution Comparison', fontsize=16)
        plt.xlabel('Jet $p_T$ [GeV]', fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 6. Show/Save
        print("Saving histograms now")
        plt.tight_layout()
        # plt.show()
        output_name = f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/plots/pythia_vs_herwig/starting_all_jetpt_inclusive_jetpt{target_jet_pt}gev.pdf"
        plt.savefig(output_name, bbox_inches='tight')

        plt.close('all') # Essential: frees the memory used by the figure
        del df_pythia    # Delete the dataframe reference
        del df_herwig

# # --- Usage ---
plot_jet_pt_counts()
# # plot_jet_pt('path_to_pythia.parquet', 'path_to_herwig.parquet')





# ─────────────────────────────────────────────
# STEP 1: PARSE PYTHIA SLURM FILES
# ─────────────────────────────────────────────

def parse_slurm_file(filepath):
    """Extract sigmaGen (mb) and N_accepted from a single Pythia slurm output file."""
    sigma      = None
    n_accepted = None

    with open(filepath, 'r') as f:
        content = f.read()

    sum_match = re.search(
        r'\|\s+sum\s+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.e+\-]+)\s+([\d.e+\-]+)',
        content
    )
    if sum_match:
        n_accepted = int(sum_match.group(3))
        sigma      = float(sum_match.group(4))  # mb

    return sigma, n_accepted


def get_pythia_scale_factor(target_jet_pt):
    """
    Parse all slurm files for a given target_jet_pt bin.
    Returns scale_f in mb/event.
    """
    idx_start, idx_end = slurm_ranges[target_jet_pt]
    all_files = sorted(glob.glob(os.path.join(slurm_base, "slurm-51506550_*.out")))

    selected_files = [
        f for f in all_files
        if idx_start <= int(re.search(r'slurm-51506550_(\d+)\.out', f).group(1)) <= idx_end
    ]

    if len(selected_files) == 0:
        raise RuntimeError(f"No slurm files found for target_jet_pt={target_jet_pt} "
                           f"(expected indices {idx_start}–{idx_end})")

    sigma_values = []
    n_accepted_values = []
    n_failed = 0

    for filepath in selected_files:
        sigma, n_accepted = parse_slurm_file(filepath)
        if sigma is None or n_accepted is None:
            n_failed += 1
            continue
        sigma_values.append(sigma)
        n_accepted_values.append(n_accepted)

    sigma_arr  = np.array(sigma_values)
    n_arr      = np.array(n_accepted_values)

    sigma_mean = sigma_arr.mean()
    sigma_std  = sigma_arr.std()
    n_total    = int(n_arr.sum())
    scale_f    = sigma_mean / n_total  # mb/event

    print(f"\n  [Pythia] target_jet_pt = {target_jet_pt} GeV")
    print(f"    Files parsed:        {len(sigma_values)}  ({n_failed} failed)")
    print(f"    sigma_mean:          {sigma_mean:.4e} mb")
    print(f"    sigma_std:           {sigma_std:.4e} mb  "
          f"({'OK' if sigma_std/sigma_mean < 0.01 else 'LARGE — check logs!'})")
    print(f"    N_accepted (total):  {n_total}")
    print(f"    scale_f:             {scale_f:.4e} mb/event")

    return scale_f  # mb/event


# ─────────────────────────────────────────────
# STEP 2: LOAD HERWIG SCALE FACTORS FROM JSON
# ─────────────────────────────────────────────

def load_herwig_scale_factors(json_path):
    """
    Load pre-computed Herwig scale factors from JSON.
    Returns dict: { target_jet_pt (int): scale_f (nb/event) }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    scale_factors = {}
    print("\n  [Herwig] Scale factors loaded from JSON:")
    for pt_str, vals in data.items():
        pt = int(pt_str)
        scale_f = vals['scale_f_nb_per_event']  # nb/event
        scale_factors[pt] = scale_f
        print(f"    target_jet_pt = {pt:>4} GeV  |  sigma = {vals['sigma_nb']:.4e} nb  |  "
              f"N_total = {vals['n_total']}  |  scale_f = {scale_f:.4e} nb/event")

    return scale_factors


# ─────────────────────────────────────────────
# STEP 3: COMPUTE CROSS SECTION + PLOT
# ─────────────────────────────────────────────

print("=" * 60)
print("Parsing Pythia slurm files for scale factors...")
print("=" * 60)

# Load Herwig scale factors from JSON
herwig_scale_factors = load_herwig_scale_factors(herwig_json)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, target_jet_pt in enumerate(target_jet_pts):
    ax = axes[i]

    # --- Get scale factors ---
    pythia_scale_f = get_pythia_scale_factor(target_jet_pt)   # mb/event
    herwig_scale_f = herwig_scale_factors[target_jet_pt]       # nb/event

    # --- Load parquets ---
    pythia_path = os.path.join(parquet_base, f"{target_jet_pt}gev", "JetsForAnalysisCombined.parquet")
    herwig_path = os.path.join(herwig_base,  f"{target_jet_pt}gev", "JetsForAnalysisCombined.parquet")

    print(f"\n  Loading parquets for {target_jet_pt} GeV...")
    df_pythia = pd.read_parquet(pythia_path, columns=['jet_pt', 'jet_id'])
    df_herwig = pd.read_parquet(herwig_path, columns=['jet_pt', 'jet_id'])
    print(f"    Pythia jets: {len(df_pythia)},  Herwig jets: {len(df_herwig)}")

    # --- Define bins ---
    pt_max      = target_jet_pt * 2
    bins        = np.arange(0, pt_max + BIN_WIDTH, BIN_WIDTH)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --- Histogram jet pT ---
    pythia_counts, _ = np.histogram(df_pythia['jet_pt'], bins=bins)
    herwig_counts, _ = np.histogram(df_herwig['jet_pt'], bins=bins)

    # --- Convert to d sigma / d pT in pb/GeV ---
    # Pythia: scale_f [mb/event] * 1e9 [pb/mb] * counts / bin_width -> pb/GeV
    # Herwig: scale_f [nb/event] * 1e3 [pb/nb] * counts / bin_width -> pb/GeV
    pythia_xsec = pythia_scale_f * 1e9 * pythia_counts / BIN_WIDTH  # pb/GeV
    herwig_xsec = herwig_scale_f * 1e3 * herwig_counts / BIN_WIDTH  # pb/GeV

    # Mask empty bins for log scale
    pythia_mask = pythia_xsec > 0
    herwig_mask = herwig_xsec > 0

    # --- Plot ---
    ax.step(bin_centers[pythia_mask], pythia_xsec[pythia_mask],
            where='mid', color='blue', linewidth=2, label='Pythia')
    ax.step(bin_centers[herwig_mask], herwig_xsec[herwig_mask],
            where='mid', color='red',  linewidth=2, label='Herwig')

    ax.axvline(x=target_jet_pt, color='black', linestyle='--',
               linewidth=1.5, label=f'Target $p_T$ = {target_jet_pt} GeV')

    ax.set_yscale('log')
    ax.set_xlim(0, pt_max)
    ax.set_xlabel(r'Jet $p_T$ [GeV]', fontsize=13)
    ax.set_ylabel(r'$d\sigma/dp_T$ [pb/GeV]', fontsize=13)
    ax.set_title(
        rf'$\hat{{p}}_{{T,\mathrm{{min}}}}$ = {int(target_jet_pt * 0.8)} GeV  '
        rf'(target $p_T$ = {target_jet_pt} GeV)',
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    del df_pythia
    del df_herwig

plt.suptitle(r'Jet $p_T$ Cross Section: Pythia vs Herwig', fontsize=16, y=1.01)
plt.tight_layout()

output_path = os.path.join(output_dir, "jet_pt_xsec_pythia_vs_herwig.pdf")
plt.savefig(output_path, bbox_inches='tight')
plt.close('all')
print(f"\nSaved: {output_path}")