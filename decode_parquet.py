# Use this file to examine what is in a parquet file!

import pandas as pd

# Load the file
# df = pd.read_parquet("testing/JetsForAnalysis.parquet")
# df = pd.read_parquet("/software/users/blianggi/mypyjetty/analysis/testing/JetsForAnalysis.parquet")
# df = pd.read_parquet("/global/cfs/cdirs/alice/blianggi/mypyjetty/analysis/testing/JetsForAnalysis.parquet")
df = pd.read_parquet("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/pythia_otf/53423546/100gev/FilteredJetsForAnalysisCombined.parquet")
# df = pd.read_parquet(f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/storage/herwig/1006458/100gev/FilteredJetsForAnalysisCombined.parquet")
# df = pd.read_parquet("/global/cfs/cdirs/alice/blianggi/rstorage/alice/generation/blianggi/herwiggen/tree_gen/54316088/100gev/1/JetsForAnalysis.parquet")
# df = pd.read_parquet("/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/blianggi/jse/data/55595752/69/DataJetsForAnalysis.parquet")
df = pd.read_parquet("/global/cfs/cdirs/alice/blianggi/mypyjetty/analysis/testing/jets_out.parquet")

# View the first 10 rows
print(df.head(50))

# View summary statistics (mean pt, min/max eta, etc.)
print(df.describe())

# Check how much RAM it's taking and column types
print(df.info())

print(df.tail(10))
# print(df["const_pt"].tail(2))
print(df["jet_pt"].iloc[-2])
print(df["jet_pt"].iloc[-1])
print(list(df["const_pt"].iloc[-2])) # untruncated version
print(list(df["const_pt"].iloc[-1]))

print("####")
print(df["const_pt"].iloc[-4]) # untruncated version
print(df["const_pt"].iloc[-3]) # untruncated version
print(list(df["const_label"].iloc[-4])) # untruncated version
print(list(df["const_label"].iloc[-3])) # untruncated version
print("nconst", df["nconst"].iloc[-4])
print("const_pt", list(df["const_pt"].iloc[-4]))
print("const_eta", list(df["const_eta"].iloc[-4]))
print("const_phi", list(df["const_phi"].iloc[-4]))
print("is_matched", df["is_matched"].iloc[-4])
print("match_index", df["match_index"].iloc[-4])
print("match_pt", df["match_pt"].iloc[-4])
print("match_dr", df["match_dr"].iloc[-4])


[ 2.11141443  3.15115809  0.94204134  1.08604252  2.04287624  0.73495799
 12.00445175]
[ 2.08764219  3.17547441  0.92008561  1.06224287  2.02708673 11.98383999
  0.73976988]
[np.int64(1665803), np.int64(1666120), np.int64(1665802), np.int64(1665954), np.int64(1665801), np.int64(1666201), np.int64(1666200)]
[np.int64(1665803), np.int64(1666120), np.int64(1665802), np.int64(1665954), np.int64(1665801), np.int64(1666200), np.int64(1666201)]
nconst 7
const_pt [np.float64(2.1114144325256348), np.float64(3.151158094406128), np.float64(0.9420413374900818), np.float64(1.0860425233840942), np.float64(2.0428762435913086), np.float64(0.7349579930305481), np.float64(12.004451751708984)]
const_eta [np.float64(0.3893775933543335), np.float64(-0.09895175485746767), np.float64(-0.10129255752694026), np.float64(0.12846644648095967), np.float64(0.38386609674234495), np.float64(0.13628182342028203), np.float64(0.23689555728081657)]
const_phi [np.float64(-2.6620493570910853), np.float64(-2.517160717641012), np.float64(-2.231929127370016), np.float64(-2.0717089811908167), np.float64(-2.3182762304889124), np.float64(-2.2632978598224085), np.float64(-2.319420401250021)]
is_matched True
match_index 0
match_pt 21.794659
match_dr 0.0005988517