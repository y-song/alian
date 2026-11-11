---
name: prepare-run
description: Pre-submission check for an alian SLURM production run - verify the submit script, config, file lists and merge.sh agree with the change, draft the joblog.md entry, and list the exact smoke-test, sbatch, merge and plot commands for the user to run. Use when the user is about to submit a job or asks whether a change is ready to run.
---

# Prepare a production run

Paths are relative to `alian/alian/`. This skill **prepares** a run; it never
runs anything. Don't run the analysis, ROOT, `sbatch`, `merge.sh` or any other
SLURM command, even on the cluster. The user runs all of them.

## 1. Find what will run

- Which submit script? `grep -n 'ANALYSIS_CODE=\|CONFIG=' submit*.sh`.
  Embedding runs use `submit_embed.sh`; data runs `submit_data.sh`; JEWEL
  `submit.sh`.
- What changed since the last production run? `git log --oneline` and
  `git diff <last run commit>` for the analysis script, the config and
  `analysis/base/`. The last run's job ID is the newest line in the relevant
  `joblog.md` section. On the cluster, its exact config and code are in
  `/rstorage/youqi/<last jobid>/`, so `diff` against those copies.

## 2. Check the submit script

For the script found above, check and report each item:

- `ANALYSIS_CODE` and `CONFIG` point to the files that were changed.
- The provenance block copies the submit script by its own name
  (`cp submit_embed.sh ...` inside `submit_embed.sh`).
- `FILE_LIST` / `FILE_LIST_PP` / `FILE_LIST_OO` are the intended samples. Task
  `i` reads line `i+1`, so every list needs at least `N+1` lines for
  `#SBATCH --array=0-N`, and for embedding the two lists are paired line by
  line. The `list/` files are not in git: on the cluster, check with
  `wc -l list/<file>`; otherwise ask the user for the line counts.
- `--time` and `--mem-per-cpu` still look sufficient if the change adds work
  per jet (e.g. `perp_recluster_closure: true` adds a clustering per matched
  jet). Flag it; don't change resources without asking.
- `--output`/`--error` and `OUTPUT_DIR` are the usual locations
  (`/home/youqi/temp/%A_%a.*`, `/rstorage/youqi/$SLURM_ARRAY_JOB_ID`).

## 3. Check the config

- Every switch the change depends on is set the way the run needs (cuts,
  `z_cut`, `perpcone_R`, `ghost_area`, `cent_min/max`, closure flags).
- Cuts in the config agree with what the code and joblog entry claim
  (pT and eta cuts, R, centrality range).
- If histograms were added or renamed, run the checker from the
  `add-histogram` skill:
  ```bash
  python3 ../.claude/skills/add-histogram/check_hists.py <analysis script> <config>
  ```
- Run `scripts/check_python.sh` if any Python changed.

## 4. Check merge.sh

- `N_TASKS` must equal the array size: `--array=0-N` → `N_TASKS=N+1`. Update
  the trailing comment (`# array 0-N in submit_<x>.sh`) to match.
- `JOB_ID` is filled in only after submission. Leave it, and tell the user to
  set it.
- `FILE_DIR` / `OUTPUT_DIR` use `/rstorage/youqi/$JOB_ID`, not the
  commented-out `~/temp` alternatives, unless the user wants those.

Fix mismatches in `merge.sh` or the submit script only if the user agrees, as
a separate commit (`merge.sh: N_TASKS = 33 for submit_embed.sh array 0-32`).

## 5. Draft the joblog entry

Add nothing to `joblog.md` yourself unless asked. Propose a line for the right
section (e.g. `# pp MC + OO MB data embedding`), in the existing format:

```
: <MM/DD/YY>. <what changed physics-wise, and the key settings>
```

Use today's date as MM/DD/YY, e.g. `11/11/26`. The job ID stays blank for the user to fill in.
Name the physics change and settings, not the code change ("zcut applied to
rhoA-subtracted subjets", not "edited select_soft_drop").

## 6. Hand over

Report what you checked as a short list: OK, fixed, or needs the user's
decision. State that the code is untested. Suggest committing and pushing
before submitting, so the job ID maps to a commit. Then give the commands
in order, each in its own block. Example for `submit_embed.sh`:

1. Smoke test, and what to look for (finishes, no `KeyError`, new histograms
   present with entries):
   ```bash
   python analysis/test/embed_perpcone.py -i1 /home/youqi/temp/combined1.root -i2 /home/youqi/temp/BerkeleyTree.root -c config/embed_perpcone.yaml -o output/embed_test.root -n 100
   ```
2. Submit:
   ```bash
   sbatch submit_embed.sh
   ```
3. After it finishes, set `JOB_ID=<jobid>` in `merge.sh`, then merge. Any
   `WARNING: missing or empty` line names a task to rerun:
   ```bash
   bash merge.sh
   ```
4. Plot, with the macro that matches the change:
   ```bash
   root -l 'plot/plot_eec_perpcone.C("<jobid>", "40", "60", "OO")'
   ```
