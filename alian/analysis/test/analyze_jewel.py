#!/usr/bin/env python3
""" Example usage: 
python analysis/test/analyze_jewel.py -i /rstorage/lbergmann/jewel/pp_100GeV.root -c config/jewel.yaml -o output/pp_100GeV.root
"""

import argparse
import itertools
import numpy as np
from alian.analysis.base import AnalysisBaseFlat, add_default_args_flat, delta_R
import heppyy

fj = heppyy.load_cppyy('fastjet')

class AnalyzeJewel(AnalysisBaseFlat):

    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.jets contains selected jets (i.e. after selection cuts)

        [self.hists['track_pT'].Fill(t.pt()) for t in self.tracks]
        [self.hists['jet_pT'].Fill(j.pt()) for j in self.jets]
        [self.hists['jet_eta'].Fill(j.eta()) for j in self.jets]
        for j in self.jets:
            if (j.pt() < self.pt_min_jet):
                break
            self.do_eec(j, "eec")
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(lund_seq, z_cut=self.z_cut) # class is LundDeclustering
            if l is None:
                self.hists['rg'].Fill(j.pt(), -0.99)
                self.hists['zg'].Fill(j.pt(), -0.99)
                continue
            subjet_a = l.harder()
            subjet_b = l.softer()
            sd_pt = subjet_a.pt() + subjet_b.pt() # approximate
            self.do_eec(j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=j.pt()) # jet passes SD, but this includes stuff removed by SD
            self.do_eec(subjet_a, "eec_aa", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec(subjet_b, "eec_bb", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec_cross(subjet_a, subjet_b, "eec_ab", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.hists['rg'].Fill(j.pt(), delta_R(subjet_a, subjet_b))
            self.hists['zg'].Fill(j.pt(), subjet_b.pt() / sd_pt)

    def do_eec(self, jet, hist_name, ew_denom=None, jet_pt_bin=None):
        if ew_denom is None:
            ew_denom = jet.pt()
        if jet_pt_bin is None:
            jet_pt_bin = jet.pt()
        tracks = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(tracks, 2):
            ew = p1.pt() * p2.pt() / ew_denom / ew_denom
            rl = delta_R(p1, p2)
            self.hists[hist_name].Fill(jet_pt_bin, rl, ew)

    def do_eec_cross(self, subjet_a, subjet_b, hist_name, ew_denom, jet_pt_bin):
        tracks_a = self.eec_trk_selector(subjet_a.constituents())
        tracks_b = self.eec_trk_selector(subjet_b.constituents())
        for p1 in tracks_a:
            for p2 in tracks_b:
                ew = p1.pt() * p2.pt() / ew_denom / ew_denom
                rl = delta_R(p1, p2)
                self.hists[hist_name].Fill(jet_pt_bin, rl, ew)

    def select_soft_drop(self, lund_seq, z_cut=0.1):
        """
        Return the first primary Lund splitting satisfying z > z_cut (soft-drop
        condition with beta=0), or None if no splitting passes.
        """
        for l in lund_seq:
            if l.z() > z_cut:
                return l
        return None

    def finalize(self):
        self.hists['track_pT'].Scale(1, "width")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on ROOT file using YAML configuration.")
    parser = add_default_args_flat(parser)

    args = parser.parse_args()

    ana = AnalyzeJewel(args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev)
    ana.run()