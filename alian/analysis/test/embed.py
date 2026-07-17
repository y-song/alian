#!/usr/bin/env python3
"""Embedding analysis: compare grid-median vs jet-median rho estimators.

For each event pair (one from each file):
  1. Cluster tracks from a pp file → pp jets
  2. Extract pp jet constituents; combine with tracks from a OO file
  3. Estimate rho from the combined event with both estimators:
       - GridMedianBackgroundEstimator  (bge_type: grid)
       - JetMedianBackgroundEstimator   (bge_type: jet)
  4. Cluster the combined event with jet areas → combined jets
  5. For each pp jet, find the nearest combined jet (delta-R < max_match_dR)
  6. Apply area subtraction: pt_sub = pt_combined - rho * A
  7. Fill delta_pT = pt_sub - pt_pp histograms for each rho estimator.

Example usage:
python analysis/test/embed.py -i1 /rstorage/alice/run3/data/LHC24_ppref/BerkeleyTrees/69/BerkeleyTree.root -i2 /rstorage/alice/run3/data/LHC25ae/BerkeleyTrees/69/BerkeleyTree.root -c config/embed.yaml -o output/embed.root
"""

import argparse
from time import perf_counter
import numpy as np

import heppyy
fj = heppyy.load_cppyy('fastjet')
from cppyy.gbl import std

from ROOT import TFile

from alian.analysis.base import (
    AnalysisSelector, JetFinder,
    set_up_logger, delta_R,
)
from alian.analysis.base.output import Output
from alian.analysis.base.event import Event, get_selected_tracks
from alian.analysis.base.utils import read_yaml, is_slurm
from alian.io.data_io import DataInput


class EmbeddingAnalysis:
    """Embed pp jet constituents from a pp file into OO events from an OO file.

    Tests whether the grid-median or jet-median rho estimator better recovers
    the pp jet pT when area subtraction (pt_combined - rho*A) is applied.

    nev limits the number of event *pairs* processed.  Events that fail their
    respective event selectors are skipped without consuming a slot from the
    other source.
    """

    _defaults = {
        'pt_min_pp_jet': 10.0,
        'max_match_dR': 0.4,
    }

    def __init__(self, pp_file, oo_file, output_file, cfg_file,
                 pp_tree_struct, oo_tree_struct,
                 nev=-1, lhc_run=3):
        self.pp_file = pp_file
        self.oo_file = oo_file
        self.output_file = output_file
        self.cfg_file = cfg_file
        self.pp_tree_struct = pp_tree_struct
        self.oo_tree_struct = oo_tree_struct
        self.nev = nev
        self.lhc_run = lhc_run

    def run(self):
        self.start_time = perf_counter()
        self.logger = set_up_logger(__name__)
        self.logger.info("Starting embedding analysis!")
        self.init()
        self.analyze_events()
        self.finalize()
        self.save()
        self.note_time("Embedding analysis complete")

    def init(self):
        self.cfg = read_yaml(self.cfg_file)

        for k, v in (self._defaults | self.cfg.get('analysis', {})).items():
            setattr(self, k, v)

        # --- signal source (file 1) ---
        pp_cfg = self.cfg.get('pp', {})
        self.pp_source = DataInput(
            self.pp_file,
            lhc_run=self.lhc_run,
            yaml_file=self.pp_tree_struct,
            n_events=-1,
        )
        self.pp_selector = AnalysisSelector.load(pp_cfg)
        pp_jf_opts = {**JetFinder._defaults, **pp_cfg.get('jet_finder', {})}
        self.pp_jet_finder = JetFinder(**pp_jf_opts)

        # --- background source (file 2) ---
        oo_cfg = self.cfg.get('oo', {})
        self.oo_source = DataInput(
            self.oo_file,
            lhc_run=self.lhc_run,
            yaml_file=self.oo_tree_struct,
            n_events=-1,
        )
        self.oo_selector = AnalysisSelector.load(oo_cfg)

        # --- rho estimators ---
        bge_cfg = self.cfg.get('bkg_estimator', {})
        max_eta  = bge_cfg.get('max_eta', 0.9)

        grid_size = bge_cfg.get('bge_rho_grid_size', 0.5)
        self.bge_grid = fj.GridMedianBackgroundEstimator(max_eta, grid_size)

        bge_jet_R = bge_cfg.get('bge_jet_R', 0.2)
        sel_not = getattr(fj, "operator!")
        bge_jet_selector = (
            fj.SelectorAbsEtaMax(max_eta - bge_jet_R)
            * sel_not(fj.SelectorNHardest(2))
        )
        bge_jet_def  = fj.JetDefinition(fj.kt_algorithm, bge_jet_R)
        bge_area_def = fj.AreaDefinition(
            fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(max_eta)
        )
        self.bge_jet = fj.JetMedianBackgroundEstimator(
            bge_jet_selector, bge_jet_def, bge_area_def
        )

        # --- embedded jet finder (needs jet areas for area subtraction) ---
        combined_jf_opts = {**JetFinder._defaults, **oo_cfg.get('jet_finder', {})}
        self.combined_jet_finder = JetFinder(**combined_jf_opts)

        self.area_cut = 0.56*np.pi*pp_jf_opts['R']*pp_jf_opts['R']
        self.output = Output.load(self.cfg)
        self.hists = self.output.hists
        self.logger.info("Embedding analysis initialized.")

    # -----------------------------------------------------------------------
    # event loop
    # -----------------------------------------------------------------------

    def _selected_events(self, source, selector, disable_bar=False):
        """Yield raw event structs from source that pass the event selector."""
        for ev in source.next_event(disable_bar=disable_bar):
            event = Event(ev)
            if selector.event and not selector.event.selects(event):
                continue
            yield ev

    def analyze_events(self):
        self.logger.info("Analyzing embedded events...")
        slurm_check = is_slurm()
        n_pairs = 0
        for pp_ev, oo_ev in zip(
            self._selected_events(self.pp_source, self.pp_selector, slurm_check),
            self._selected_events(self.oo_source, self.oo_selector, slurm_check),
        ):
            self.analyze_event_pair(pp_ev, oo_ev)
            n_pairs += 1
            if self.nev > 0 and n_pairs >= self.nev:
                break
        self.logger.info(f"Processed {n_pairs} event pairs.")
        self.note_time("Events analyzed")

    # -----------------------------------------------------------------------
    # per-pair analysis
    # -----------------------------------------------------------------------

    def analyze_event_pair(self, pp_ev, oo_ev):
        # --- truth jets from pp files ---
        pp_tracks = get_selected_tracks(pp_ev, self.pp_selector.track)
        pp_jets = self.pp_jet_finder.find_jets(pp_tracks, use_area=False)
        pp_jets = [j for j in pp_jets if j.pt() >= self.pt_min_pp_jet]
        if not pp_jets:
            return

        # --- combined event: signal jet constituents + background tracks ---
        oo_tracks = get_selected_tracks(oo_ev, self.oo_selector.track)
        combined = self._combined_event(pp_jets, oo_tracks)

        # --- rho from both estimators on the combined event ---
        self.bge_grid.set_particles(combined)
        rho_grid = self.bge_grid.rho()
        sigma_grid = self.bge_grid.sigma()
        self.hists['area_grid'].Fill(self.bge_grid.mean_area())

        self.bge_jet.set_particles(combined)
        rho_jet = self.bge_jet.rho()
        sigma_jet = self.bge_jet.sigma()
        [self.hists['area_jet_median'].Fill(j.area()) for j in self.bge_jet.jets_used()]

        # --- cluster combined event with jet areas ---
        combined_jets = self.combined_jet_finder.find_jets(combined, use_area=True)
        combined_jets = [j for j in combined_jets if j.pt() >= self.pt_min_combined_jet]

        oo_centrality = oo_ev.data['centrality']
        self.hists['cent_rho_grid'].Fill(rho_grid, oo_centrality)
        self.hists['cent_rho_jet'].Fill(rho_jet, oo_centrality)

        # --- fill histograms per pp jet ---
        for pp_j in pp_jets:    

            match = self._find_match(pp_j, combined_jets)
            self.hists['combined_pT_pp_jet_pT'].Fill(pp_j.pt(), match.pt() if match is not None else 0.0)
            self.hists['combined_A_pp_jet_pT'].Fill(pp_j.pt(), match.area() if match is not None else -0.1)

            if match is None:
                pt_sub_grid = 0
                pt_sub_jet = 0
            else:
                pt_sub_grid = match.pt() - rho_grid * match.area()
                pt_sub_jet  = match.pt() - rho_jet  * match.area()
                perpcone_rho = self._find_perpcone_rho(match, combined, coneR=self.perpcone_R)

            self.hists['delta_pT_grid_pp_jet_pT'].Fill(pp_j.pt(), pt_sub_grid - pp_j.pt())
            self.hists['delta_pT_jet_pp_jet_pT'].Fill(pp_j.pt(), pt_sub_jet  - pp_j.pt())
            self.hists['sub_pT_grid_pp_jet_pT'].Fill(pp_j.pt(), pt_sub_grid)
            self.hists['sub_pT_jet_pp_jet_pT'].Fill(pp_j.pt(), pt_sub_jet)

            if pt_sub_grid > 10.0 and match.area() > self.area_cut:
                self.hists['delta_pT_grid_pp_jet_pT_matched'].Fill(pp_j.pt(), pt_sub_grid - pp_j.pt())
                self.hists['sub_pT_grid_pp_jet_pT_matched'].Fill(pp_j.pt(), pt_sub_grid)
                self.hists['rho_grid_pp_jet_pT_matched'].Fill(pp_j.pt(), rho_grid)
                self.hists['sigma_grid_pp_jet_pT_matched'].Fill(pp_j.pt(), sigma_grid)
                self.hists['combined_A_pp_jet_pT_grid_matched'].Fill(pp_j.pt(), match.area())
                self.hists['combined_perpcone_rho_pp_jet_pT_grid_matched'].Fill(pp_j.pt(), perpcone_rho)
                self.hists['delta_rho_grid_pp_jet_pT_grid_matched'].Fill(pp_j.pt(), rho_grid-perpcone_rho)
            if pt_sub_jet > 10.0 and match.area() > self.area_cut:
                self.hists['delta_pT_jet_pp_jet_pT_matched'].Fill(pp_j.pt(), pt_sub_jet - pp_j.pt())
                self.hists['sub_pT_jet_pp_jet_pT_matched'].Fill(pp_j.pt(), pt_sub_jet)
                self.hists['sigma_jet_pp_jet_pT_matched'].Fill(pp_j.pt(), sigma_jet)
                self.hists['rho_jet_pp_jet_pT_matched'].Fill(pp_j.pt(), rho_jet)
                self.hists['combined_A_pp_jet_pT_jet_matched'].Fill(pp_j.pt(), match.area())
                self.hists['combined_perpcone_rho_pp_jet_pT_jet_matched'].Fill(pp_j.pt(), perpcone_rho)
                self.hists['delta_rho_jet_pp_jet_pT_jet_matched'].Fill(pp_j.pt(), rho_jet-perpcone_rho)

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------

    def _combined_event(self, pp_jets, oo_tracks):
        """Merge signal jet constituents with background tracks into one vector."""
        combined = list(oo_tracks)
        for j in pp_jets:
            combined.extend(j.constituents())
        return std.vector[fj.PseudoJet](combined)

    def _find_match(self, ref_jet, jets):
        """Return the closest jet to ref_jet within max_match_dR, or None."""
        best, best_dR = None, self.max_match_dR
        for j in jets:
            dR = delta_R(ref_jet, j)
            if dR < best_dR:
                best, best_dR = j, dR
        return best
    
    def _find_perpcone_rho(self, ref_jet, combined, coneR=0.4):
        perpcone1 = fj.PseudoJet()
        perpcone1.reset_PtYPhiM(ref_jet.perp(), ref_jet.rapidity(), ref_jet.phi() + np.pi/2, ref_jet.m())
        perpcone2 = fj.PseudoJet()
        perpcone2.reset_PtYPhiM(ref_jet.perp(), ref_jet.rapidity(), ref_jet.phi() - np.pi/2, ref_jet.m())
        perpcone_pt = 0
        for part in combined:
          if perpcone1.delta_R(part) <= coneR or perpcone2.delta_R(part) <= coneR:
            perpcone_pt += part.perp()
        return perpcone_pt / (2*np.pi*coneR*coneR)

    # -----------------------------------------------------------------------
    # output / timing
    # -----------------------------------------------------------------------

    def finalize(self):
        pass

    def save(self):
        self.logger.info(f"Saving output to: {self.output_file}")
        with TFile(self.output_file, "RECREATE") as f:
            self.output.save(f)
        self.logger.info("Output saved.")

    def note_time(self, msg):
        self.logger.info(
            f"{msg}: ------- {self.fmt_time(perf_counter() - self.start_time)} -------",
            stacklevel=2,
        )

    def fmt_time(self, seconds):
        m, s = divmod(round(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}h {m:02d}m {s:02d}s"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Embed signal jet constituents into background events and compare "
            "grid-median vs jet-median rho estimation."
        )
    )
    parser.add_argument(
        '-i1', '--pp-file', type=str, required=True,
        help="pp input file (Run3 tree).",
    )
    parser.add_argument(
        '-i2', '--oo-file', type=str, required=True,
        help="OO input file (Run3 tree).",
    )
    parser.add_argument(
        '-o', '--output-file', type=str, default="embed.root",
        help="Output ROOT file.",
    )
    parser.add_argument(
        '-c', '--config-file', type=str, required=True,
        help="YAML configuration file.",
    )
    parser.add_argument(
        '-t1', '--pp-tree-struct', type=str, default=None,
        help="YAML describing the pp tree structure (default: auto-detected).",
    )
    parser.add_argument(
        '-t2', '--oo-tree-struct', type=str, default=None,
        help="YAML describing the OO tree structure (default: auto-detected).",
    )
    parser.add_argument(
        '-n', '--nev', type=int, default=-1,
        help="Number of event pairs to process (-1 for all).",
    )
    parser.add_argument(
        '--lhc-run', type=int, default=3,
        help="LHC run number (used for both files).",
    )
    args = parser.parse_args()

    ana = EmbeddingAnalysis(
        pp_file=args.pp_file,
        oo_file=args.oo_file,
        output_file=args.output_file,
        cfg_file=args.config_file,
        pp_tree_struct=args.pp_tree_struct,
        oo_tree_struct=args.oo_tree_struct,
        nev=args.nev,
        lhc_run=args.lhc_run,
    )
    ana.run()
