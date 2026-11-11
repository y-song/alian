# alian

ALICE jet analysis built on heppyy (cppyy bindings to fastjet and ROOT).

## Where things live (all under `alian/`)
- `analysis/`: analysis scripts, grouped by study. Scripts like
  `analysis/test/embed_perpcone.py` explain their histograms in a top-of-file
  docstring; keep it up to date when you change what a script fills.
- `analysis/base/`: shared jet finding, selections, event and output helpers.
- `config/*.yaml`: analysis configs (`embed_perpcone.yaml`) and tree structures
  (`mc_tstruct.yaml`, `run3_tstruct.yaml`).
- `plot/*.C`: ROOT macros that read merged output. Each one has an
  "Example Usage" comment at the top.
- `submit*.sh`, `merge.sh`: SLURM submission and merging with `hadd`.
- `joblog.md`: log of SLURM job IDs and what each run changed.
- `sandbox/`: old or exploratory code. Don't edit it unless asked.

## Running
- Everything runs on the cluster (SLURM). Commands run from `alian/alian/`.
- Environment: `cd ~ && source setup.sh`. That script activates the yasp venv
  (`~/yasp/venvyasp`), sources `~/yasp/.venvstartup.sh`, and loads the
  `bundle/hepbase`, `heppyy` and `alian` modules. After changing C++ in `src/`,
  recompile with `./install_with_yasp.sh` from the repo root and reload
  `module load alian`.
- Smoke test before a full submission. Example for `embed_perpcone.py`, whose `-n`
  flag caps the number of event pairs:
  `python analysis/test/embed_perpcone.py -i1 /home/youqi/temp/combined1.root -i2 /home/youqi/temp/BerkeleyTree.root -c config/embed_perpcone.yaml -o output/embed_test.root -n 100`
- Full run: `sbatch submit_embed.sh`. Output goes to
  `/rstorage/youqi/<jobid>/<task>/out.root`, and the script copies the config
  and analysis code there for provenance.
- Merge: set `JOB_ID` and `N_TASKS` in `merge.sh`, which writes
  `/rstorage/youqi/<jobid>/AnalysisResultsFinal.root`.
- Plot: e.g. `root -l 'plot/plot_eec_perpcone.C("<jobid>", "40", "60", "OO")'`.

## Working conventions
- Claude Code is used both in the cloud (no heppyy, fastjet or ROOT available,
  so the code can't be run) and locally on the cluster (where it can).
  - In the cloud, say clearly that the code is untested, and end every PR
    description with the exact smoke-test command to run on the cluster.
  - On the cluster, run the smoke test before calling a change done. Never
    run full productions on a login node; submit them with `sbatch`.
- Keep changes small and focused, one physics change per commit. Commit
  messages start with the file name, e.g. `embed_perpcone.py: ...`.
- When a change affects production output, propose a `joblog.md` entry
  (`<jobid>: <date>. <what changed>`). Leave the job ID blank for the user to
  fill in after submitting.
- Histogram names are read by the plot macros. When renaming or removing a
  histogram, update or flag every macro in `plot/` that uses it.
- Don't hard-code new absolute paths in analysis code. Take them from the
  command-line arguments or the config.

## C/C++ (plot macros, `src/`)
- Use Allman brace style (opening brace on its own line) with four-space indentation.
- Don't column-align consecutive declarations or assignments.
