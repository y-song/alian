# Use this file to examine what is in a parquet file!

import pandas as pd

# Load the file
# df = pd.read_parquet("testing/JetsForAnalysis.parquet")
df = pd.read_parquet("/software/users/blianggi/mypyjetty/analysis/testing/JetsForAnalysis.parquet")
# df = pd.read_parquet("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/51506550/100gev/FilteredJetsForAnalysisCombined.parquet")

# View the first 10 rows
print(df.head(10))

# View summary statistics (mean pt, min/max eta, etc.)
print(df.describe())

# Check how much RAM it's taking and column types
print(df.info())