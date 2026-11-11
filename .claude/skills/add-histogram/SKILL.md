---
name: add-histogram
description: Checklist for adding, renaming or removing a histogram in an alian analysis script (e.g. analysis/test/embed_perpcone.py) - booking it in the config YAML, filling it, updating the docstring and plot macros, and handing over a smoke test. Use whenever a change makes a script fill a new or different histogram.
---

# Add a histogram to an analysis script

Paths below are relative to `alian/alian/`. Follow every step; the point of this
skill is that nothing gets skipped.

## How booking and writing work here

- Histograms are **booked in the config**, not in Python. `Output.load(cfg)`
  (`analysis/base/output.py`) creates one ROOT histogram per entry of
  `output.histograms`, and `Output.save()` writes every booked histogram. So a
  booked histogram is always written, and a histogram created by hand in Python
  (`TH2F(...)`) is **not** written unless `save()` writes it explicitly.
- Entry format: `tag: [type, "root_name", "title;x title;y title", bin_x, bin_y]`
  - `type`: `1`/`2`/`3` for TH1F/TH2F/TH3F (`'2D'` etc. for doubles).
  - Keep `tag` equal to `root_name`. Code fills by tag, plot macros `Get()` by
    ROOT name.
  - Axes name binnings from `output.bins`. Reuse an existing one (`jet_pT`, `RL`,
    `area`, ...) when it fits; add a new binning only when none does.
- A fill of a tag missing from the config raises `KeyError` mid-run, often
  hours into a SLURM job. A booked tag that is never filled gives an empty
  histogram that a macro may silently plot.

## Checklist

1. **Book** the histogram in the config the script runs with (for
   `embed_perpcone.py`: `config/embed_perpcone.yaml`). Put it in the matching
   `# --- section ---` block. If another config is used with the same script
   (check `CONFIG=` in the `submit*.sh` files and the docstring's example
   commands), book it there too.
2. **Fill** it in the script with the event weight, like its neighbours:
   `self.hists['name'].Fill(x, y, weight)`. Check:
   - axis order matches the bin list in the config (x first);
   - which pT goes on the axis (ungroomed vs groomed, raw vs rho-subtracted)
     matches the title and the related histograms;
   - it is filled under the intended selection (all jets, matched, SD-passed)
     and with the right track/particle set (ghosts excluded where needed;
     pp `user_index >= 0`, OO `< 0`).
   If the fill goes through a helper that takes a name or prefix, follow the
   name all the way to `self.hists[...]`.
3. **Run the static checks** (they never run the analysis):
   ```bash
   python3 ../.claude/skills/add-histogram/check_hists.py analysis/test/embed_perpcone.py config/embed_perpcone.yaml
   scripts/check_python.sh
   ```
   `check_hists.py` must report no `ERROR`, and no new `WARNING` for your
   histogram. If your name appears only under `NOTE` or in the f-string list,
   confirm by reading the code that the f-string really produces it. Fix what
   `check_python.sh` reports.
4. **Update the top-of-file docstring** of the script: what the histogram is,
   when it is filled, and how it relates to the others (e.g. what it should
   sum to or be compared with).
5. **Plot macros** (`plot/*.C`):
   - `grep -n '<related histogram names>' plot/*.C` to find the macro that
     draws the related histograms. If the new histogram belongs there, add it.
     Follow the C++ style in CLAUDE.md: Allman braces, four spaces, no column
     alignment.
   - When renaming or removing a histogram, grep `plot/` for the old name and
     update every macro, or list the ones you did not update for the user.
   - If no macro fits and the user wants one, say so; don't write a new macro
     unasked.
6. **Commit** one physics change per commit, message starting with the file
   name, e.g. `embed_perpcone.py: add combined_matched_area_a/b`. Push only to
   the branch you were given.
7. **Hand over.** Say the change is untested. If it changes production output,
   propose a `joblog.md` line (`: <MM/DD/YY>. <what changed>`, job ID left blank).
   End with the smoke test for the user to run, e.g.:
   ```bash
   python analysis/test/embed_perpcone.py -i1 /home/youqi/temp/combined1.root -i2 /home/youqi/temp/BerkeleyTree.root -c config/embed_perpcone.yaml -o output/embed_test.root -n 100
   ```
   ```bash
   root -l -b -q -e 'TFile f("output/embed_test.root"); auto h = (TH1 *)f.Get("<name>"); if (h) { h->Print(); } else { cout << "MISSING" << endl; }'
   ```
   and what to look for: the run finishes without `KeyError`, the histogram
   exists, it has a non-zero number of entries (for a rarely filled one,
   say how many events it may take), and the axis ranges look sensible.

Do not run the smoke test or ROOT yourself; the user runs all analysis code.
