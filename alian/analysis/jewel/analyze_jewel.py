#!/usr/bin/env python3
"""Example usage:
python analysis/jewel/analyze_jewel.py -i /rstorage/lbergmann/jewel/temp_scan/Ti_590MeV/pThat_90GeV/jewel_90GeV_medium_recoilON_0.root -n 10 -c config/jewel.yaml -o output/test.root"""

import argparse
import itertools
import numpy as np
from alian.analysis.base import AnalysisBaseFlat, add_default_args_flat, delta_R
import heppyy
from cppyy.gbl import std

fj = heppyy.load_cppyy("fastjet")


class AnalyzeJewel(AnalysisBaseFlat):

    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()
        # self.sd = fj.contrib.SoftDrop(0, self.z_cut)

        # Set default parton mask to no mask if not in config
        if not hasattr(self, 'parton_mask'):
            self.parton_mask = "pid < 999"

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.jets contains selected jets (i.e. after selection cuts)

        self.hists["event"].Fill(0.5)
        [self.hists["track_pT"].Fill(t.pt()) for t in self.tracks]

        if not self.require_matched_jets:
            self.matched_jets = self.jets
        else:
            self.matched_jets = self.match_jets_partons(self.jets, self.partons, self.parton_mask, matching_distance=0.24)
        [self.hists["jet_pT"].Fill(j.pt()) for j in self.matched_jets]
        [self.hists["jet_eta"].Fill(j.eta()) for j in self.matched_jets]

        for j in self.matched_jets:    
            if j.pt() < self.pt_min_jet:
                break
            self.do_eec(j, "eec")
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(
                lund_seq, z_cut=self.z_cut
            )  # class is LundDeclustering
            if l is None:
                self.hists["rg"].Fill(j.pt(), -0.99)
                self.hists["zg"].Fill(j.pt(), -0.99)
                self.hists["sdjet_pT_jet_pT"].Fill(j.pt(), 0)
                continue
            subjet_a = l.harder()
            subjet_b = l.softer()
            sd_pt = subjet_a.pt() + subjet_b.pt()  # approximate
            self.hists["sdjet_pT_jet_pT"].Fill(j.pt(), sd_pt)
            # sd_j = self.sd(j)
            self.do_eec(
                j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=j.pt()
            )  # jet passes SD, but this includes stuff removed by SD
            # self.do_eec(sd_j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=j.pt()) # jet passes SD, and this only includes stuff passes by SD
            self.do_eec(subjet_a, "eec_aa", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec(subjet_b, "eec_bb", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec_cross(
                subjet_a, subjet_b, "eec_ab", ew_denom=sd_pt, jet_pt_bin=j.pt()
            )
            self.hists["rg"].Fill(j.pt(), delta_R(subjet_a, subjet_b))
            # self.hists['rg'].Fill(j.pt(), sd_j.structure_of[fj.contrib.SoftDrop]().delta_R())
            self.hists["rg_log"].Fill(j.pt(), delta_R(subjet_a, subjet_b))
            self.hists["zg"].Fill(j.pt(), subjet_b.pt() / sd_pt)
            self.do_eec_noew(subjet_a, "eec_aa_noew", jet_pt_bin=j.pt())
            self.do_eec_noew(subjet_b, "eec_bb_noew", jet_pt_bin=j.pt())
            self.do_eec_cross_noew(subjet_a, subjet_b, "eec_ab_noew", jet_pt_bin=j.pt())

    def finalize(self):
        self.hists["track_pT"].Scale(1, "width")

    def match_jets_partons(self, jets, partons, parton_mask_str, matching_distance=0.24):
        """Match jets to partons based on deltaR proximity.

        Args:
            jets: vector of PseudoJets
            partons: vector of PseudoJets
            parton_mask_str: string expression to filter partons, e.g., "pid < 21" or "pid == 21"
                            where 'pid' refers to parton.user_index()
            matching_distance: deltaR threshold for matching
        """
        matched_jets_count = 0
        # Filter partons using the mask string
        selected_partons = [p for p in partons if eval(parton_mask_str, {"pid": p.user_index()})]
        if len(selected_partons) == 0:
            return std.vector[fj.PseudoJet]()
        matched_partons = [False for i in range(len(selected_partons))]
        matched_jets = std.vector[fj.PseudoJet]()
        for j in jets:
            for ip, p in enumerate(selected_partons):
                if matched_jets_count == len(selected_partons):
                    return fj.sorted_by_pt(matched_jets)
                if matched_partons[ip] == True:
                    continue
                if delta_R(p, j) < matching_distance:
                    matched_jets_count += 1
                    matched_partons[ip] = True
                    # j.set_user_index(p.user_index())
                    matched_jets.push_back(j)
                    self.hists["parton_pT_jet_pT"].Fill(j.perp(), p.perp())

        return fj.sorted_by_pt(matched_jets)
    
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

    def do_eec_noew(self, jet, hist_name, jet_pt_bin=None):
        if jet_pt_bin is None:
            jet_pt_bin = jet.pt()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis on ROOT file using YAML configuration."
    )
    parser = add_default_args_flat(parser)

    args = parser.parse_args()

    ana = AnalyzeJewel(
        args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev
    )
    ana.run()