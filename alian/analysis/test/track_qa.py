#!/usr/bin/env python3
""" Example usage: 
python analysis/test/track_qa.py -i /rstorage/alice/run3/data/LHC25ae/BerkeleyTrees/69/BerkeleyTree.root -c config/test.yaml -o output/test.root
"""

import argparse
import itertools
import numpy as np
from alian.analysis.base import AnalysisBase, add_default_args, delta_R
import heppyy
from ROOT import TH1F, TH2F

fj = heppyy.load_cppyy('fastjet')

class TrackQA(AnalysisBase):
    _defaults = {
        'pt_min_eec': 1.0,
        'pt_min_jet': 20.0,
        'n_rho_grid_diag_events': 10,
    } # gets overriden by yaml values

    def init_analysis(self, analysis_cfg: dict):
        config = self._defaults | analysis_cfg
        for setting, value in config.items():
            setattr(self, setting, value)
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()
        # self.sd = fj.contrib.SoftDrop(0, self.z_cut)
        # diagnostic: per-grid-cell background density distribution, one histogram
        # per event, to check whether the median in bge.rho() is taken from a
        # roughly Gaussian distribution of cell densities
        for i in range(self.n_rho_grid_diag_events):
            self.hists[f'rho_grid_evt{i}'] = TH1F(
                f'rho_grid_evt{i}',
                f'Background density per grid cell, event {i};#rho_{{grid}} (GeV);N(grid cells)',
                100, 0, 100,
            )
            # event display: particle (#eta, #varphi) positions, z-axis (bin content) is summed pT;
            # title is filled in with N(tracks) and centrality once the event is analyzed
            h_disp = TH2F(
                f'event_display_evt{i}',
                f'Event {i};#eta;#phi',
                50, -1.0, 1.0,
                63, 0, 2 * np.pi,
            )
            h_disp.SetStats(False)
            h_disp.GetZaxis().SetTitle("#it{p}_{T} (GeV)")
            h_disp.GetZaxis().SetRangeUser(0, 3)
            self.hists[f'event_display_evt{i}'] = h_disp

    def analyze_event(self):
        # Analyzes this event that has passed the selection criteria
        # self.event contains the selected event
        # self.tracks contains selected tracks (i.e. after selection cuts)
        # self.jets contains selected jets (i.e. after selection cuts)

        if self.event_counter < self.n_rho_grid_diag_events:
            for rho_cell in self.get_rho_per_cell():
                self.hists[f'rho_grid_evt{self.event_counter}'].Fill(rho_cell)
            h_disp = self.hists[f'event_display_evt{self.event_counter}']
            h_disp.SetTitle(
                f"Event {self.event_counter}: centrality = {self.event.centrality:.1f}%, "
                f"N_{{tracks}} = {len(self.tracks)};#eta;#varphi"
            )
            [h_disp.Fill(t.eta(), t.phi(), t.pt()) for t in self.tracks]
            print("Event:", self.event_counter, ", rho =", self.rho, ", sigma =", self.sigma)

        self.hists['event'].Fill(0.5)
        self.hists['cent_mult'].Fill(self.event.multiplicity, self.event.centrality)
        self.hists['cent_ntrack'].Fill(len(self.tracks), self.event.centrality)
        self.hists['cent_rho'].Fill(self.rho, self.event.centrality)
        self.hists['sigma_rho'].Fill(self.rho, self.sigma)
        [self.hists['track_pT'].Fill(t.pt()) for t in self.tracks]
        if len(self.jets) == 0:
            return
        self.hists['event'].Fill(1.5)
        [self.hists['jet_pT'].Fill(j.pt()) for j in self.jets]
        [self.hists['jet_eta'].Fill(j.eta()) for j in self.jets]
        [self.hists['jet_rhoA_pT'].Fill(j.pt(), j.area()*self.rho) for j in self.jets]
        [self.hists['jet_A_pT'].Fill(j.pt(), j.area()) for j in self.jets]
        [self.hists['jet_pT_sub_pT'].Fill(j.pt(), j.pt()-j.area()*self.rho) for j in self.jets]
        has_acceptable_jet = False
        for j in self.jets:
            pt_sub = j.pt() - j.area()*self.rho
            if (pt_sub < self.pt_min_jet):
                break
            has_acceptable_jet = True
            [self.hists['track_in_jet_pT'].Fill(t.pt()) for t in j.constituents()]
        if has_acceptable_jet == False:
            return
        self.hists['event'].Fill(2.5)
        [self.hists['track_in_jet_event_pT'].Fill(t.pt()) for t in self.tracks]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on ROOT file using YAML configuration.")
    parser = add_default_args(parser)

    args = parser.parse_args()

    ana = TrackQA(args.input_file, args.output_file, args.config_file, args.tree_struct, args.nev, args.lhc_run)
    ana.run()