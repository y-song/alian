# Use this file to combine a number of parquet files into one merged parquet file
# Then filter out just the jets with jet pts being studied

import duckdb
import glob
import os

# User parameters!
# =============================================
generator = "herwig"  # "pythia" or "herwig" 
# =============================================

if generator == "pythia":
    jobid = "53423546"
    base_outputdir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/{jobid}"
elif generator == "herwig":
    # jobid = "1006458" #hiccup
    # base_outputdir = f"/rstorage/generators/herwig_alice/tree_gen/{jobid}" #hiccup
    jobid = "54380351" #perlmutter
    base_outputdir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/herwiggen/tree_gen/{jobid}" #perlmutter

jet_pts = [50, 100, 200, 500]

conn = duckdb.connect()
conn.execute("SET memory_limit='50GB'")  # adjust to what's available on your machine
conn.execute("SET threads=4")             # reduce from default (usually = num CPU cores)
conn.execute("SET preserve_insertion_order=false")

for i, jetpt in enumerate(jet_pts):
    print(f"Merging and filtering jetpt {jetpt}")

    outf_path_jetpt = f"{base_outputdir}/{jetpt}gev/"
    comb_output_path = f"{outf_path_jetpt}JetsForAnalysisCombined.parquet"
    filtered_output_path = f"{outf_path_jetpt}FilteredJetsForAnalysisCombined.parquet"

    # Check that input files exist before attempting to merge
    input_files = glob.glob(f"{outf_path_jetpt}/**/JetsForAnalysis.parquet", recursive=True)
    if not input_files:
        print(f"No JetsForAnalysis.parquet files found for jetpt={jetpt}, skipping...")
        continue

    print(f"Found {len(input_files)} input files for jetpt={jetpt}")

    # Step 1: Merge all individual parquet files into one combined file, if it doesn't already exist - this might go OOM
    if os.path.exists(comb_output_path):
        print(f"Combined file already exists, skipping merge: {comb_output_path}")
    else:
        print(f"Merging into {comb_output_path}...")
        conn.execute(f"""
            COPY (
                SELECT * FROM '{outf_path_jetpt}/**/JetsForAnalysis.parquet'
            ) TO '{comb_output_path}' (FORMAT 'PARQUET')
        """)
        result = conn.execute(f"SELECT COUNT(*) FROM '{comb_output_path}'").fetchone()
        print(f"Combined file created with {result[0]} rows")

    # Step 2: Filter the combined file by jet pt window
    print(f"Filtering into {filtered_output_path}...")
    conn.execute(f"""
        COPY (
            SELECT * FROM '{comb_output_path}'
            WHERE jet_pt >= {jetpt} AND jet_pt <= {jetpt * 1.2}
        ) TO '{filtered_output_path}' (FORMAT 'PARQUET')
    """)
    result = conn.execute(f"SELECT COUNT(*) FROM '{filtered_output_path}'").fetchone()
    print(f"Done! Filtered data saved to {filtered_output_path}")
    print(f"Number of rows kept: {result[0]}")

conn.close()