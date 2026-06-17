import re
import glob
import numpy as np
import os
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

jobid = "54380351"
# herwig_base   = "/rstorage/generators/herwig_alice/hepmc/1006458"
herwig_base   = f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/generation/blianggi/herwiggen/hepmc/{jobid}"
output_json   = f"herwig_scale_factors_{jobid}.json"  # will be read by the main plotting script

target_jet_pts = [50, 100, 200, 500]

# Subdirectory name for each pt bin (use folder names like '50gev', etc)
pt_to_subdir = {
    50:  '50gev',
    100: '100gev',
    200: '200gev',
    500: '500gev',
}

N_DIRS = 750  # directories per pt bin


# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────

def parse_herwig_file(filepath):
    """
    Extract cross section (nb) and N_generated from a single Herwig .out file.
    Targets the 'Total (from generated events)' line, e.g.:
    Total (from generated events):             5000         5000      2.65(2)e+03
    """
    sigma      = None
    n_generated = None

    with open(filepath, 'r') as f:
        content = f.read()

    # Match the "Total (from generated events)" line
    # Format: Total (from generated events):   N_gen   N_attempts   sigma(nb)
    # Sigma may look like: 2.65(2)e+03 or 2.650e+03
    match = re.search(
        r'Total \(from generated events\):\s+(\d+)\s+(\d+)\s+([\d.]+(?:\(\d+\))?[eE][+\-]\d+)',
        content
    )
    if match:
        n_generated = int(match.group(1))
        # Strip uncertainty notation e.g. 2.65(2)e+03 -> 2.65e+03
        sigma_str = re.sub(r'\(\d+\)', '', match.group(3))
        sigma = float(sigma_str)  # nb

    return sigma, n_generated


def get_herwig_scale_factor(target_jet_pt):
    """
    Parse all .out files for a given target_jet_pt bin and return
    the combined scale factor sigma / N_total  [nb/event].
    """
    subdir = pt_to_subdir[target_jet_pt]
    sigma_values    = []
    n_gen_values    = []
    n_failed        = 0

    for job_idx in range(1, N_DIRS + 1):
        pattern = os.path.join(herwig_base, str(subdir), str(job_idx), "*.out")
        files   = glob.glob(pattern)

        if len(files) == 0:
            print(f"    [WARNING] No .out file found: {pattern}")
            n_failed += 1
            continue

        # Take the first .out file found in each job directory
        filepath = files[0]
        sigma, n_generated = parse_herwig_file(filepath)

        if sigma is None or n_generated is None:
            print(f"    [WARNING] Could not parse: {filepath}")
            n_failed += 1
            continue

        sigma_values.append(sigma)
        n_gen_values.append(n_generated)

    sigma_arr  = np.array(sigma_values)
    n_arr      = np.array(n_gen_values)

    sigma_mean = sigma_arr.mean()
    sigma_std  = sigma_arr.std()
    n_total    = int(n_arr.sum())
    scale_f = sigma_mean / n_total   # nb/event

    print(f"\n  target_jet_pt = {target_jet_pt} GeV")
    print(f"    Directories parsed:  {len(sigma_values)}  ({n_failed} failed/missing)")
    print(f"    sigma_mean:          {sigma_mean:.4e} nb")
    print(f"    sigma_std:           {sigma_std:.4e} nb  "
          f"({'OK' if sigma_std/sigma_mean < 0.01 else 'LARGE — check logs!'})")
    print(f"    N_generated (total): {n_total}")
    print(f"    scale_f:             {scale_f:.4e} nb/event")

    return {
        'sigma_nb':  sigma_mean,
        'sigma_std': sigma_std,
        'n_total':   n_total,
        'scale_f_nb_per_event': scale_f,
        'n_failed':  n_failed,
    }


# ─────────────────────────────────────────────
# RUN + SAVE
# ─────────────────────────────────────────────

print("=" * 60)
print("Parsing Herwig output files for scale factors...")
print("=" * 60)

results = {}
for target_jet_pt in target_jet_pts:
    results[str(target_jet_pt)] = get_herwig_scale_factor(target_jet_pt)

# Save to JSON so the main plotting script can read it
with open(output_json, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n{'='*60}")
print(f"Saved scale factors to: {output_json}")
print(f"{'='*60}")

# Pretty print summary table
print(f"\n{'target_jet_pt':>15} {'sigma (nb)':>14} {'N_total':>12} {'scale_f (nb/ev)':>18}")
print("-" * 65)
for pt, vals in results.items():
    print(f"{pt+' GeV':>15} {vals['sigma_nb']:>14.4e} {vals['n_total']:>12} {vals['scale_f_nb_per_event']:>18.4e}")