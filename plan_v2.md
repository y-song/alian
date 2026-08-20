# Implementation Plan: Differential Efficiency, Purity, and Lund Plane

## Goal
Enhance `make_rms_etc.py` to provide differential efficiency and purity for jets and pairs, add splitting efficiency/purity, and plot the Lund plane for unmatched distributions.

## 1. Jet Efficiency & Purity vs $p_T$
- **Histograms**: 
  - `h_jet_eff_num` (Matched Jets vs $p_T^{part}$)
  - `h_jet_eff_den` (Total Particle Jets vs $p_T^{part}$)
  - `h_jet_pur_num` (Matched Jets vs $p_T^{det}$)
  - `h_jet_pur_den` (Total Detector Jets vs $p_T^{det}$)
- **Logic**:
  - Loop through all `jets` from parquet.
  - If `level == "part"`, fill `h_jet_eff_den`. If also matched, fill `h_jet_eff_num`.
  - If `level == "det"`, fill `h_jet_pur_den`. If also matched, fill `h_jet_pur_num`.

## 2. Pair Efficiency & Purity vs $R_L$
- **Histograms** (per label `lab` in `EEC_LABELS`):
  - `h_pair_eff_num_{lab}` (Matched Pairs vs $R_L^{part}$)
  - `h_pair_eff_den_{lab}` (Total Particle Pairs vs $R_L^{part}$)
  - `h_pair_pur_num_{lab}` (Matched Pairs vs $R_L^{det}$)
  - `h_pair_pur_den_{lab}` (Total Detector Pairs vs $R_L^{det}$)
- **Logic**:
  - Inside the main loop, for each pair in `det_pairs` and `part_pairs`, fill the respective denominator.
  - For each pair in `matched`, fill the respective numerators using the $R_L$ from the matched pair.

## 3. Splitting Efficiency & Purity (2D $\ln(k_T)$ vs $\ln(R/\Delta R)$)
- **Definitions**:
  - A "split" is the first SD-passing split found by `select_split_sd`.
  - Efficiency: $\frac{\text{Matched Splits}}{\text{Total Particle Splits}}$
  - Purity: $\frac{\text{Matched Splits}}{\text{Total Detector Splits}}$
  - X-axis: $\ln(k_T)$ where $k_T = \text{split.perp()}$
  - Y-axis: $\ln(R/\Delta R)$ where $R = \text{JET\_R}$ and $\Delta R = \text{split.dR()}$
- **Histograms**:
  - `h_split_eff_num`, `h_split_eff_den` (vs $\ln k_T, \ln(R/\Delta R)$)
  - `h_split_pur_num`, `h_split_pur_den` (vs $\ln k_T, \ln(R/\Delta R)$)
- **Logic**:
  - Since the main loop already processes matched jets and their SD splits (`det_d`, `part_d`), I will use these.
  - Particle split $\to$ `h_split_eff_den`. If matched $\to$ `h_split_eff_num`.
  - Detector split $\to$ `h_split_pur_den`. If matched $\to$ `h_split_pur_num`.

## 4. Lund Plane for Unmatched Distributions
- **Goal**: Plot $\ln(k_T)$ vs $\ln(R/\Delta R)$ for jets/splits that were NOT matched.
- **Histograms**:
  - `h_lund_unmatched_det`
  - `h_lund_unmatched_part`
- **Logic**:
  - When a jet is not matched (or the split is not matched), fill the corresponding histogram using the split's $k_T$ and $\Delta R$.
  - Note: For detector jets that are unmatched, they won't enter the current `if not bool(det.is_matched): continue` block. I need to move the split extraction logic to handle unmatched jets.

## Implementation Strategy
1.  **Binning**: Define bins for $\ln(k_T)$ and $\ln(R/\Delta R)$.
2.  **Histogram Setup**: Create all required ROOT histograms at the start of `main()`.
3.  **Loop Modification**:
    - Update the loop to process **all** detector jets (remove the `if not bool(det.is_matched): continue` guard) to capture unmatched detector distributions.
    - Ensure particle jets are also accounted for (perhaps a separate pass over `jets[jets.level == "part"]` to ensure all particle jets are counted for efficiency denominators).
4.  **Filling**: Integrate filling logic into the loop.
5.  **Saving**: Write all histograms to the output file.
