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
- Full run (the user submits it): `sbatch submit_embed.sh`. Output goes to
  `/rstorage/youqi/<jobid>/<task>/out.root`, and the script copies the config
  and analysis code there for provenance.
- Merge: set `JOB_ID` and `N_TASKS` in `merge.sh`, which writes
  `/rstorage/youqi/<jobid>/AnalysisResultsFinal.root`.
- Plot: e.g. `root -l 'plot/plot_eec_perpcone.C("<jobid>", "40", "60", "OO")'`.

## Working conventions
- The user runs all analysis code. Don't run analysis scripts, smoke tests or
  ROOT macros, and never submit, cancel or modify SLURM jobs (`sbatch`,
  `scancel`, ...), even when working on the cluster where they would work.
- Static checks are the exception: after changing Python, run
  `scripts/check_python.sh` (from anywhere in the repo). It only byte-compiles
  and runs pyflakes on files changed relative to `main`, and reports only
  warnings that are new compared with `main`. Fix what it reports before handing
  over. Don't clean up pre-existing warnings unless asked. If pyflakes is
  missing, install it in the cloud with `python3 -m pip install pyflakes`; on the
  cluster, don't install into the yasp venv. Tell the user instead.
- Code changes are still untested when handed over. Say so, and end every
  change (and every PR description) with the exact smoke-test command for the
  user to run, plus what to look for in the output.
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
