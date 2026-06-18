#!/usr/bin/env python3
""" Example usage: 
python analysis/test/analyze.py -i /rstorage/alice/run3/data/LHC25ae/BerkeleyTrees/69/BerkeleyTree.root -c config/test.yaml -o output/test.root
"""

import argparse
import itertools
import numpy as np
from alian.analysis.base import AnalysisBase, add_default_args, delta_R
import heppyy

fj = heppyy.load_cppyy('fastjet')

class Analyze(AnalysisBase):
    _defaults = {
        'pt_min_eec': 1.0,
        'pt_min_jet': 20.0,
        'z_cut': 0.1,
    }

    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()
        # self.sd = fj.contrib.SoftDrop(0, self.z_cut)

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.jets contains selected jets (i.e. after selection cuts)

        # self.hists['event'].Fill(0.5)
        self.hists['cent_mult'].Fill(self.event.multiplicity, self.event.centrality)
        self.hists['cent_ntrack'].Fill(len(self.tracks), self.event.centrality)
        self.hists['cent_rho'].Fill(self.rho, self.event.centrality)
        self.hists['sigma_rho'].Fill(self.rho, self.sigma)
        [self.hists['track_pT'].Fill(t.pt()) for t in self.tracks]
        [self.hists['jet_pT'].Fill(j.pt()) for j in self.jets]
        [self.hists['jet_eta'].Fill(j.eta()) for j in self.jets]
        [self.hists['jet_rhoA_pT'].Fill(j.pt(), j.area()*self.rho) for j in self.jets]
        [self.hists['jet_A_pT'].Fill(j.pt(), j.area()) for j in self.jets]
        [self.hists['jet_pT_sub_pT'].Fill(j.pt(), j.pt()-j.area()*self.rho) for j in self.jets]
        for j in self.jets:
            pt_sub = j.pt() - j.area()*self.rho
            if (pt_sub < self.pt_min_jet):
                break
            self.do_eec(j, "eec")
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(lund_seq, z_cut=self.z_cut) # class is LundDeclustering
            if l is None:
                self.hists['rg'].Fill(pt_sub, -0.99)
                self.hists['zg'].Fill(pt_sub, -0.99)
                continue
            subjet_a = l.harder()
            subjet_b = l.softer()
            sd_pt = subjet_a.pt() + subjet_b.pt() # approximate
            # sd_j = self.sd(j)
            self.do_eec(j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=pt_sub) # jet passes SD, but this includes stuff removed by SD
            # self.do_eec(sd_j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=pt_sub) # jet passes SD, and this only includes stuff passes by SD
            self.do_eec(subjet_a, "eec_aa", ew_denom=sd_pt, jet_pt_bin=pt_sub)
            self.do_eec(subjet_b, "eec_bb", ew_denom=sd_pt, jet_pt_bin=pt_sub)
            self.do_eec_cross(subjet_a, subjet_b, "eec_ab", ew_denom=sd_pt, jet_pt_bin=pt_sub)
            self.hists['rg'].Fill(pt_sub, delta_R(subjet_a, subjet_b))
            # self.hists['rg'].Fill(pt_sub, sd_j.structure_of[fj.contrib.SoftDrop]().delta_R())
            self.hists['rg_log'].Fill(pt_sub, delta_R(subjet_a, subjet_b))
            self.hists['zg'].Fill(pt_sub, subjet_b.pt() / sd_pt)
            self.do_eec_noew(subjet_a, "eec_aa_noew", jet_pt_bin=pt_sub)
            self.do_eec_noew(subjet_b, "eec_bb_noew", jet_pt_bin=pt_sub)
            self.do_eec_cross_noew(subjet_a, subjet_b, "eec_ab_noew", jet_pt_bin=pt_sub)

    def do_eec(self, jet, hist_name, ew_denom=None, jet_pt_bin=None):
        if ew_denom is None:
            ew_denom = jet.pt() - jet.area()*self.rho
        if jet_pt_bin is None:
            jet_pt_bin = jet.pt() - jet.area()*self.rho
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

    def do_eec_noew(self, jet, hist_name, jet_pt_bin=None):
        if jet_pt_bin is None:
            jet_pt_bin = jet.pt() - jet.area()*self.rho
        tracks = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(tracks, 2):
            rl = delta_R(p1, p2)
            self.hists[hist_name].Fill(jet_pt_bin, rl)

    def do_eec_cross_noew(self, subjet_a, subjet_b, hist_name, jet_pt_bin):
        tracks_a = self.eec_trk_selector(subjet_a.constituents())
        tracks_b = self.eec_trk_selector(subjet_b.constituents())
        for p1 in tracks_a:
            for p2 in tracks_b:
                rl = delta_R(p1, p2)
                self.hists[hist_name].Fill(jet_pt_bin, rl)

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
    parser = add_default_args(parser)

    args = parser.parse_args()

    ana = Analyze(args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev, args.lhc_run)
    ana.run()