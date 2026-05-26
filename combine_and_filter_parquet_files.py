# Use this file to combine a number of parquet files into one merged parquet file
# Then filter out just the jets with jet pts being studied

import duckdb
import pyarrow.parquet as pq
import pyarrow as pa

# User parameters!
# =============================================
generator = "herwig"  # "pythia" or "herwig" 
# =============================================

if generator == "pythia":
    jobid = "51384740"
    base_outputdir = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/{jobid}"
elif generator == "herwig":
    jobid = "1006458"
    base_outputdir = f"/rstorage/generators/herwig_alice/tree_gen/{jobid}"

jet_pts = [ 50, 100, 200, 500]

for i, jetpt in enumerate(jet_pts):
    # The **/*.parquet pattern searches all subdirectories
    print("merging jetpt", jetpt)

    outf_path_jetpt = f"{base_outputdir}/{jetpt}gev/"
    comb_output_path = f"{outf_path_jetpt}JetsForAnalysisCombined.parquet"
    duckdb.query(f"COPY (SELECT * FROM '{outf_path_jetpt}/**/JetsForAnalysis.parquet') TO '{comb_output_path}' (FORMAT 'PARQUET')")


    print("Filtering jet pt", jetpt)

    # Define the filter
    # The format is a list of tuples: (column, operation, value)
    # Multiple tuples in a list act as an 'AND' operation
    filters = [
        ('jet_pt', '>=', jetpt),
        ('jet_pt', '<=', jetpt * 1.2)
    ]

    # Read the table with the filter applied
    # This only loads the rows that meet your criteria into RAM
    table = pq.read_table(comb_output_path, filters=filters)

    # Write the filtered result to a new Parquet file
    filtered_output_path = f"{base_outputdir}/{jetpt}gev/FilteredJetsForAnalysisCombined.parquet"
    pq.write_table(table, filtered_output_path)

    # Optional: Print the count to verify
    print(f"Done! Filtered data saved to {filtered_output_path}")
    print(f"Number of rows kept: {table.num_rows}")