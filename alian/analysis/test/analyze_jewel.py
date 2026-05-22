#!/usr/bin/env python3
""" Example usage: python analysis/test/analyze_jewel.py -i /rstorage/lbergmann/jewel/jewel_50GeV_vacuum_1.root -c config/jewel.yaml
"""

import argparse
import itertools
import numpy as np
from alian.analysis.base import AnalysisBaseFlat, add_default_args_flat, delta_R
import heppyy

fj = heppyy.load_cppyy('fastjet')

class Analysis(AnalysisBaseFlat):
    _defaults = {
        'pt_min_eec': 1.0
    }
    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.clusters contains selected clusters (i.e. after selection cuts)
        # self.jets contains selected jets

        [self.hists['track_pT'].Fill(t.pt()) for t in self.tracks]
        [self.hists['jet_pT'].Fill(j.pt()) for j in self.jets]
        [self.hists['jet_pT_coarse'].Fill(j.pt()) for j in self.jets]
        [self.do_eec(j) for j in self.jets]

    def do_eec(self, jet):
        tracks = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(tracks, 2):
            ew = p1.pt() * p2.pt() / jet.pt() / jet.pt()
            angle = delta_R(p1, p2)
            self.hists["eec"].Fill(jet.pt(), angle, ew)

    def finalize(self):
        self.hists['track_pT'].Scale(1, "width")
        self.hists['jet_pT_coarse'].Scale(1, "width")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on ROOT file using YAML configuration.")
    parser = add_default_args_flat(parser)

    args = parser.parse_args()

    ana = Analysis(args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev)
    ana.run()