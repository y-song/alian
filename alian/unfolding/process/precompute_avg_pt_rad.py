#!/usr/bin/env python3
"""
Precompute average radiator pT (avg_pt_rad) for the JSE analysis.

Does a single pass over all jets for each input file, selects the split
according to each cut config, and records the mean radiator pT.

Saves results to a JSON file keyed by source range (e.g. "50-60", "50-55")
and flattened config tag (e.g. "inclusive_sd0.1") that the main analysis
script can read.

Run as:
    python precompute_avg_pt_rad.py pythia
    python precompute_avg_pt_rad.py herwig
"""

import pandas as pd
import fastjet as fj
import fjcontrib
import argparse
import os
import json


class AvgPtRadCalculator:
    def __init__(self, gen):
        self.gen = gen
        self.target_jet_pts = [50, 100, 200, 500]
        self.partontypes = ["inclusive", "quark", "gluon"]

        # High edge of the wide "main" bin for each target jet pt
        self.main_hi = {50: 60, 100: 120, 200: 240, 500: 600}

        self.jet_R = 0.4
        self.jet_def_ca = fj.JetDefinition(fj.cambridge_algorithm, 1.0)

        self.cut_configs = [("sd", 0.1), ("maxkt", None)]

        self.trk_thrd = 1

        self.out_path = (
            f"/global/cfs/cdirs/alice/blianggi/mypyjetty/storage/jse/rootfiles/"
            f"avg_pt_rad_{self.gen}.json"
        )

    def get_parton_type(self, pid: int) -> str:
        if abs(pid) in {1, 2, 3, 4, 5, 6}:
            return "quark"
        elif pid == 21:
            return "gluon"
        elif pid == 22:
            return "photon"
        else:
            return "unknown"

    def format_cut_tag(self, cut_mode, cut_value):
        if cut_mode == "sd":
            return f"sd{str(cut_value)}"
        if cut_mode == "maxkt":
            return "maxkt"
        return f"{cut_mode}{str(cut_value)}"

    def select_split(self, lund_plane_elements, cut_mode, cut_value):
        if cut_mode == "sd":
            for d in lund_plane_elements:
                if d.z() > cut_value:
                    return d
            return None
        if cut_mode == "maxkt":
            filtered = [d for d in lund_plane_elements if d.kt() > 1.0]
            return max(filtered, key=lambda d: d.kt(), default=None) if filtered else None
        return None

    def get_sources(self, target_jetpt):
        """Return {source_range_label: path} for a given target jet pt bin."""
        if self.gen == "pythia":
            base = (
                f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
                f"AnalysisResults/blianggi/jse/pythia_otf/55555648/{target_jetpt}gev"
            )
        elif self.gen == "herwig":
            base = (
                f"/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/"
                f"generation/blianggi/herwiggen/tree_gen/55293842/{target_jetpt}gev"
            )
        else:
            raise ValueError(f"Unknown gen: {self.gen}")

        main_hi = self.main_hi[target_jetpt]
        narrow_hi = int(round(target_jetpt * 1.1))  # e.g. 50 -> 55, 100 -> 110

        sources = {
            f"{target_jetpt}-{main_hi}": (
                f"{base}/FilteredJetsForAnalysisCombined.parquet"
            ),
            f"{target_jetpt}-{narrow_hi}": (
                f"{base}/FilteredJetsForAnalysisCombined_{target_jetpt}_{narrow_hi}.parquet"
            ),
        }
        return sources

    def process_file(self, path):
        """Single pass over one parquet file. Returns dict:
        results["<partontype>_<cut_tag>"] = {'avg_pt_rad': float, 'n_jets': int}
        """
        if not os.path.exists(path):
            print(f"  WARNING: file not found, skipping: {path}")
            return None

        df = pd.read_parquet(path)
        grouped_jets = df.groupby(['event_id', 'jet_id'])

        # accumulators: sums[cut_tag][partontype] = [sum_rad_pt, n]
        sums = {}
        for cut_mode, cut_value in self.cut_configs:
            cut_tag = self.format_cut_tag(cut_mode, cut_value)
            sums[cut_tag] = {pt: [0.0, 0] for pt in self.partontypes}

        for (event_idx, jet_id), jet_constituents in grouped_jets:
            jet_parton_pid = jet_constituents['parton_pid'].values[0]
            jet_true_type = self.get_parton_type(jet_parton_pid)

            pj_particles = [
                fj.PseudoJet(row.c_px, row.c_py, row.c_pz, row.c_e)
                for row in jet_constituents.itertuples()
            ]
            cs = fj.ClusterSequence(pj_particles, self.jet_def_ca)
            jets = fj.sorted_by_pt(cs.inclusive_jets())
            if not jets:
                continue
            jet = jets[0]

            lund_gen = fjcontrib.LundGenerator(self.jet_def_ca)
            lund_plane_elements = lund_gen.result(jet)

            # For each cut config, the selected split (and thus radiator) differs
            for cut_mode, cut_value in self.cut_configs:
                cut_tag = self.format_cut_tag(cut_mode, cut_value)
                selected_d = self.select_split(lund_plane_elements, cut_mode, cut_value)
                if selected_d is None:
                    continue
                parent_radiator = selected_d.pair()
                subjets = parent_radiator.pieces()
                if len(subjets) != 2:
                    continue
                rad_pt = parent_radiator.perp()

                # Fill for every partontype this jet qualifies for
                for pt in self.partontypes:
                    if pt == "inclusive" or jet_true_type == pt:
                        sums[cut_tag][pt][0] += rad_pt
                        sums[cut_tag][pt][1] += 1

        # build flattened results: "<partontype>_<cut_tag>"
        results = {}
        for cut_tag, per_parton in sums.items():
            for pt, (s, n) in per_parton.items():
                avg = s / n if n > 0 else None
                results[f"{pt}_{cut_tag}"] = {"avg_pt_rad": avg, "n_jets": n}
        return results

    def run(self):
        all_results = {"gen": self.gen, "sources": {}}

        for target_jetpt in self.target_jet_pts:
            sources = self.get_sources(target_jetpt)
            for source_label, path in sources.items():
                print(f"Processing {source_label}: {path}")
                res = self.process_file(path)
                if res is None:
                    continue
                all_results["sources"][source_label] = res
                # print a short summary
                for config_tag, info in res.items():
                    avg = info["avg_pt_rad"]
                    avg_str = f"{avg:.3f}" if avg is not None else "n/a"
                    print(f"    {config_tag:22s}: avg_pt_rad = {avg_str}  (n={info['n_jets']})")

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote avg_pt_rad values to {self.out_path}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Precompute avg radiator pT for JSE analysis")
    parser.add_argument("gen", choices=["pythia", "herwig"], help="Generator to process")
    args = parser.parse_args()

    calc = AvgPtRadCalculator(args.gen)
    calc.run()


if __name__ == "__main__":
    main()


'''
### Output Structure Example:
{
  "gen": "pythia",
  "sources": {
    "50-60": {
      "inclusive_sd0.1": {"avg_pt_rad": 42.1, "n_jets": 12345},
      "quark_sd0.1":     {"avg_pt_rad": 44.0, "n_jets": 6000},
      "gluon_sd0.1":     {"avg_pt_rad": 40.5, "n_jets": 6345},
      "inclusive_maxkt": {"avg_pt_rad": 38.2, "n_jets": 12000},
      "quark_maxkt":     {"avg_pt_rad": 39.9, "n_jets": 5800},
      "gluon_maxkt":     {"avg_pt_rad": 36.8, "n_jets": 6200}
    },
    "50-55":  { ... },
    "100-120": { ... },
    "100-110": { ... },
    ...
  }
}
'''