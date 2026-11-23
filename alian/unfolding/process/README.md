# Files!

GENERAL:
- decode_parquet.py: general file that summarizes parquet file (good way to check file was made correctly)
- check_generator_jetpt.py: checks the jet pt spectra from generated events, compares pythia+herwig
- determine_pthat_min.py: analyses the jet pt spectra for pt-hat min (generated from pythia_quark_gluon_jse_testing.py)
- extract_herwig_scalefactors.py: extracts herwig scale factors from job output files
- pythia_quark_gluon_jse_testing.py: generates pythia at a given pt-hat min, to test pt spectra

MONTE CARLO:
- combine_and_filter_parquet_files.py: takes existing parquet files made from slurm jobs, and merges them. also filters for the requested jet pTs
- herwig_make_jets.py: processes herwig ROOT gen-level TTree, finds the jets, and saves jets to parquet file
- plot_mc_curves_jse.py: plot combinations of the made JSE curves
- process_jets_jse.py: reads parquet file of jets, makes histograms of subjet EECs and C_{AB} [[could this be used for data too?]]
- pythia_quark_gluon_jse.py: generates pythia, finds jets, and saves jets to parquet file



order:
1. pythia_quark_gluon_jse.py OR herwig_make_jets.py
2. combine_and_filter_parquet_files.py
3. process_jets_jse.py
4. plot_mc_curves_jse.py
