#!/usr/bin/env python3
"""
Example usage:
python analysis/test/embed.py -i1 /rstorage/alice/run3/mc_central/LHC26b6/BerkeleyTree_564356_0.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed.yaml -o output/embed.root
python analysis/test/embed.py -i1 /rstorage/youqi/MC_pp_anchored_OO_hadd50/combined1.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed.yaml -o output/embed_test.root
"""

import argparse
from time import perf_counter
import numpy as np

import heppyy
fj = heppyy.load_cppyy('fastjet')
from cppyy.gbl import std

from ROOT import (
    TFile, TH2F, TCanvas, TGraph, TEllipse, TLegend,
    kBlack, kRed, kBlue, kGreen,
)
from alian.analysis.base import (
    AnalysisSelector, JetFinder,
    set_up_logger, delta_R,
)
from alian.analysis.base.output import Output
from alian.analysis.base.event import Event, get_selected_tracks
from alian.analysis.base.utils import read_yaml, is_slurm
from alian.io.data_io import DataInput


class EmbeddingAnalysis:
    """Embed pp events from a pp file into OO events from an OO file.

    nev limits the number of event *pairs* processed.  Events that fail their
    respective event selectors are skipped without consuming a slot from the
    other source.
    """

    _defaults = {
        'pt_min_pp_jet': 10.0,
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
        self.pTHat_min = pp_cfg.get('pTHat_min', 5.0)
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
        max_eta = bge_cfg.get('max_eta', 0.9)

        bge_R = bge_cfg.get('bge_R', 0.4)
        sel_not = getattr(fj, "operator!")
        bge_selector = (
            fj.SelectorAbsEtaMax(max_eta - bge_R)
            * sel_not(fj.SelectorNHardest(2))
        )
        bge_def  = fj.JetDefinition(fj.kt_algorithm, bge_R)
        bge_area_def = fj.AreaDefinition(
            fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(max_eta)
        )
        self.bge = fj.JetMedianBackgroundEstimator(
            bge_selector, bge_def, bge_area_def
        )
        print("bge:\n", self.bge.description())

        # --- embedded jet finder (needs jet areas for area subtraction) ---
        combined_jf_opts = {**JetFinder._defaults, **oo_cfg.get('jet_finder', {})}
        self.combined_jet_finder = JetFinder(**combined_jf_opts)

        self.R = pp_jf_opts['R']
        self.area_cut = 0.56*np.pi*self.R*self.R
        self.output = Output.load(self.cfg)
        self.hists = self.output.hists
        # --- event displays ---
        self.n_event_displays = 30
        self.event_display_index = 0
        self.event_displays = []
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
            if event.pTHat is not None and event.pTHat < self.pTHat_min:
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
    # per-event-pair analysis
    # -----------------------------------------------------------------------

    def analyze_event_pair(self, pp_ev, oo_ev):
        # --- pp tracks and pp jets ---
        pp_tracks = get_selected_tracks(pp_ev, self.pp_selector.track)
        pp_jets = self.pp_jet_finder.find_jets(pp_tracks, use_area=False)
        pp_jets = [j for j in pp_jets if j.pt() >= self.pt_min_pp_jet]
        weight = pp_ev.data['weight']
        [self.hists['pp_pT'].Fill(j.pt(), weight) for j in pp_jets]

        # --- combined event: pp tracks + oo tracks ---
        oo_tracks = get_selected_tracks(oo_ev, self.oo_selector.track, index_offset=-9999)
        combined = self._combined_event(pp_tracks, oo_tracks)

        # --- rho from bge on the combined event ---
        self.bge.set_particles(combined)
        rho = self.bge.rho()
        sigma = self.bge.sigma()
        self.hists['cent_rho'].Fill(rho, oo_ev.data['centrality'])

        # --- combined jets: cluster combined event with jet areas ---
        combined_jets_preselect = self.combined_jet_finder.find_jets(combined, use_area=True)
        [self.hists['combined_pT_preselect'].Fill(j.pt(), weight) for j in combined_jets_preselect]
        [self.hists['sub_pT_preselect'].Fill(j.pt() - rho*j.area(), weight) for j in combined_jets_preselect]
        combined_jets = [j for j in combined_jets_preselect if j.pt() - rho*j.area() >= self.pt_min_combined_jet]
        combined_jets.sort(key=lambda j: j.pt() - rho*j.area(), reverse=True)
        [self.hists['combined_pT'].Fill(j.pt(), weight) for j in combined_jets]
        [self.hists['sub_pT'].Fill(j.pt() - rho*j.area(), weight) for j in combined_jets]
        # ---  match pp jets to combined jets --- 
        combined_jet_matched_indices = [-1 for x in range(0, len(combined_jets))]
        for pp_ijet in range(0, len(pp_jets)):
            pp_jet = pp_jets[pp_ijet]
            combined_jet_matched = []
            combined_ijet_matched = -1
            for combined_ijet in range(0, len(combined_jets)):
                combined_jet = combined_jets[combined_ijet]
                if (combined_jet_matched_indices[combined_ijet] != -1): # combined jet already has a match
                    continue
                if (self.mc_fraction(pp_jet, combined_jet) > self.mc_fraction_threshold) and (self.is_geo_matched(combined_jet, pp_jet)):
                    combined_jet_matched.append(combined_jet)
                    combined_ijet_matched = combined_ijet
            if (len(combined_jet_matched) == 1): # pp jet has a unique combined jet match
                combined_jet_matched_indices[combined_ijet_matched] = pp_ijet
       
        self._make_event_display(
            pp_tracks=pp_tracks,
            oo_tracks=oo_tracks,
            pp_jets=pp_jets,
            combined_jets=combined_jets,
            combined_jet_matched_indices=combined_jet_matched_indices,
            rho=rho,
            centrality=oo_ev.data['centrality'],
        )

        # --- fill histograms per combined jet ---
        for combined_ijet in range(0, len(combined_jets)):
            j = combined_jets[combined_ijet]
            pp_ijet = combined_jet_matched_indices[combined_ijet] # -1 if unmatched to pp

            self.hists['combined_pT_sub_pT'].Fill(j.pt()-j.area()*rho, j.pt(), weight)
            self.hists['combined_area_sub_pT'].Fill(j.pt()-j.area()*rho, j.area(), weight)

            if (pp_ijet != -1):
                pp_j = pp_jets[pp_ijet]

                self.hists['pp_pT_matched'].Fill(pp_j.pt(), weight)

                self.hists['combined_pT_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.pt(), weight)
                self.hists['combined_area_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.area(), weight)

                self.hists['sub_pT_pp_pT_matched'].Fill(pp_j.pt(), j.pt()-j.area()*rho, weight)
                self.hists['residual_pp_pT_matched'].Fill(pp_j.pt(), j.pt()-j.area()*rho-pp_j.pt(), weight)
                self.hists['residual_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.pt()-j.area()*rho-pp_j.pt(), weight)

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    #---------------------------------------------------------------
    # Compare two jets and store matching candidates in user_info
    #---------------------------------------------------------------
    def is_geo_matched(self, jet1, jet2):
        deltaR = jet1.delta_R(jet2)
      
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < 0.6 * self.R:
            return True
        else:
            return False
    
    #---------------------------------------------------------------
    # Return pt-fraction of tracks in jet_pp that are contained in jet_combined
    #---------------------------------------------------------------
    def mc_fraction(self, jet_pp, jet_combined):
        pt_total = jet_pp.pt()
        pt_contained = 0.
        for track in jet_combined.constituents():
          if track.user_index() >= 0:
            pt_contained += track.pt()           
        return pt_contained/pt_total
    
    def _combined_event(self, pp_tracks, oo_tracks):
        """Merge pp tracks with background tracks."""
        combined = list(oo_tracks)
        combined.extend(pp_tracks)
        return std.vector[fj.PseudoJet](combined)
    
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
    
    def _make_event_display(
        self,
        pp_tracks,
        oo_tracks,
        pp_jets,
        combined_jets,
        combined_jet_matched_indices,
        rho,
        centrality,
    ):
        if self.event_display_index >= self.n_event_displays:
            return

        i_event = self.event_display_index

        # -----------------------------------------------------------
        # Base eta-phi histogram.
        #
        # Bin content = summed track pT in that eta-phi cell.
        # Both pp and OO tracks contribute to the z-axis.
        # -----------------------------------------------------------
        h_disp = TH2F(
            f'event_display_evt{i_event}',
            (
                f'Event {i_event}, centrality = {centrality:.1f}%, '
                f'#rho = {rho:.2f} GeV;'
                f'#eta;#varphi;#Sigma #it{{p}}_{{T}} (GeV)'
            ),
            50, -1.0, 1.0,
            63, 0.0, 2.0 * np.pi,
        )

        h_disp.SetStats(False)

        for track in oo_tracks:
            h_disp.Fill(track.eta(), track.phi(), track.pt())

        for track in pp_tracks:
            h_disp.Fill(track.eta(), track.phi(), track.pt())

        # -----------------------------------------------------------
        # TGraphs identify track origin.
        #
        # Marker color/style identifies pp vs OO.
        # Marker size does NOT encode pT; pT is encoded by the
        # TH2 z-axis.
        # -----------------------------------------------------------
        g_oo = TGraph(len(oo_tracks))
        g_oo.SetName(f'event_display_oo_tracks_evt{i_event}')
        g_oo.SetMarkerStyle(24)
        g_oo.SetMarkerSize(1.2)
        g_oo.SetMarkerColor(kBlack)

        for i_track, track in enumerate(oo_tracks):
            g_oo.SetPoint(
                i_track,
                track.eta(),
                track.phi(),
            )

        g_pp = TGraph(len(pp_tracks))
        g_pp.SetName(f'event_display_pp_tracks_evt{i_event}')
        g_pp.SetMarkerStyle(20)
        g_pp.SetMarkerSize(1.2)
        g_pp.SetMarkerColor(kGreen+2)

        for i_track, track in enumerate(pp_tracks):
            g_pp.SetPoint(
                i_track,
                track.eta(),
                track.phi(),
            )

        # -----------------------------------------------------------
        # Canvas
        # -----------------------------------------------------------
        c = TCanvas(
            f'event_display_canvas_evt{i_event}',
            f'Event display {i_event}',
            1000,
            800,
        )

        c.SetRightMargin(0.14)

        h_disp.Draw("COLZ")
        g_oo.Draw("P SAME")
        g_pp.Draw("P SAME")

        jet_circles = []

        print("Event:", i_event)
        # -----------------------------------------------------------
        # pp jets:
        # green solid circles
        # -----------------------------------------------------------
        for jet in pp_jets:
            print("pp jet:", jet.perp())
            circle = TEllipse(
                jet.eta(),
                jet.phi(),
                self.R,
                self.R,
            )

            circle.SetFillStyle(0)
            circle.SetLineColor(kGreen + 2)
            circle.SetLineStyle(1)
            circle.SetLineWidth(2)
            circle.Draw("SAME")

            jet_circles.append(circle)

            # Handle phi periodicity if the jet is near phi = 0.
            if jet.phi() < self.R:
                circle_copy = TEllipse(
                    jet.eta(),
                    jet.phi() + 2.0 * np.pi,
                    self.R,
                    self.R,
                )

                circle_copy.SetFillStyle(0)
                circle_copy.SetLineColor(kGreen + 2)
                circle_copy.SetLineStyle(1)
                circle_copy.SetLineWidth(2)
                circle_copy.Draw("SAME")

                jet_circles.append(circle_copy)

            # Handle phi periodicity if the jet is near phi = 2pi.
            if jet.phi() > 2.0 * np.pi - self.R:
                circle_copy = TEllipse(
                    jet.eta(),
                    jet.phi() - 2.0 * np.pi,
                    self.R,
                    self.R,
                )

                circle_copy.SetFillStyle(0)
                circle_copy.SetLineColor(kGreen + 2)
                circle_copy.SetLineStyle(1)
                circle_copy.SetLineWidth(2)
                circle_copy.Draw("SAME")

                jet_circles.append(circle_copy)

        # -----------------------------------------------------------
        # Combined jets:
        #
        # matched   -> black solid
        # unmatched -> black dashed
        #
        # All jets here have already passed 
        # j.pt() - rho * j.area() selection.
        # -----------------------------------------------------------
        for combined_ijet in range(0, len(combined_jets)):
            jet = combined_jets[combined_ijet]
            print("combined jet:", jet.perp(), jet.perp()-rho*jet.area())

            if combined_jet_matched_indices[combined_ijet] != -1:
                line_style = 1
            else:
                line_style = 2

            circle = TEllipse(
                jet.eta(),
                jet.phi(),
                self.R,
                self.R,
            )

            circle.SetFillStyle(0)
            circle.SetLineColor(kBlack)
            circle.SetLineStyle(line_style)
            circle.SetLineWidth(2)
            circle.Draw("SAME")

            jet_circles.append(circle)

            if jet.phi() < self.R:
                circle_copy = TEllipse(
                    jet.eta(),
                    jet.phi() + 2.0 * np.pi,
                    self.R,
                    self.R,
                )

                circle_copy.SetFillStyle(0)
                circle_copy.SetLineColor(kBlack)
                circle_copy.SetLineStyle(line_style)
                circle_copy.SetLineWidth(2)
                circle_copy.Draw("SAME")

                jet_circles.append(circle_copy)

            if jet.phi() > 2.0 * np.pi - self.R:
                circle_copy = TEllipse(
                    jet.eta(),
                    jet.phi() - 2.0 * np.pi,
                    self.R,
                    self.R,
                )

                circle_copy.SetFillStyle(0)
                circle_copy.SetLineColor(kBlack)
                circle_copy.SetLineStyle(line_style)
                circle_copy.SetLineWidth(2)
                circle_copy.Draw("SAME")

                jet_circles.append(circle_copy)

        # -----------------------------------------------------------
        # Legend
        # -----------------------------------------------------------
        legend = TLegend(0.15, 0.72, 0.48, 0.89)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)

        # legend.AddEntry(g_oo, "OO tracks", "p")
        # legend.AddEntry(g_pp, "pp tracks", "p")

        pp_jet_legend = TEllipse()
        # pp_jet_legend.SetFillStyle(0)
        # pp_jet_legend.SetLineColor(kGreen + 2)
        # pp_jet_legend.SetLineStyle(1)
        # pp_jet_legend.SetLineWidth(3)

        combined_matched_legend = TEllipse()
        # combined_matched_legend.SetFillStyle(0)
        # combined_matched_legend.SetLineColor(kBlack)
        # combined_matched_legend.SetLineStyle(1)
        # combined_matched_legend.SetLineWidth(3)

        combined_unmatched_legend = TEllipse()
        # combined_unmatched_legend.SetFillStyle(0)
        # combined_unmatched_legend.SetLineColor(kBlack)
        # combined_unmatched_legend.SetLineStyle(2)
        # combined_unmatched_legend.SetLineWidth(3)

        # legend.AddEntry(pp_jet_legend, "pp jets", "l")
        # legend.AddEntry(
        #     combined_matched_legend,
        #     "Combined jets, matched",
        #     "l",
        # )
        # legend.AddEntry(
        #     combined_unmatched_legend,
        #     "Combined jets, unmatched",
        #     "l",
        # )

        # legend.Draw()

        c.Update()

        # Keep Python references alive until the ROOT file is written.
        self.event_displays.append(
            {
                'canvas': c,
                'hist': h_disp,
                'oo_tracks': g_oo,
                'pp_tracks': g_pp,
                'jet_circles': jet_circles,
                'legend': legend,
                'pp_jet_legend': pp_jet_legend,
                'combined_matched_legend': combined_matched_legend,
                'combined_unmatched_legend': combined_unmatched_legend,
            }
        )

        self.event_display_index += 1

    # -----------------------------------------------------------------------
    # output / timing
    # -----------------------------------------------------------------------

    def finalize(self):
        pass

    def save(self):
        self.logger.info(f"Saving output to: {self.output_file}")

        with TFile(self.output_file, "RECREATE") as f:
            self.output.save(f)

            f.mkdir("event_displays")
            f.cd("event_displays")

            for display in self.event_displays:
                display['canvas'].Write()

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
        '-t1', '--pp-tree-struct', type=str, default="config/mc_tstruct.yaml",
        help="YAML describing the pp tree structure (default: mc_tstruct.yaml).",
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
