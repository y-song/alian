#!/usr/bin/env python3
"""
Example usage:
python analysis/test/embed.py -i1 /rstorage/alice/run3/mc_central/LHC26b6/BerkeleyTree_564356_0.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed.yaml -o output/embed.root
python analysis/test/embed.py -i1 /rstorage/youqi/MC_pp_anchored_OO_hadd50/combined1.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed.yaml -o output/embed_test.root
"""

import argparse
import itertools
from time import perf_counter
import numpy as np

import heppyy
fj = heppyy.load_cppyy('fastjet')
alian = heppyy.load_cppyy("alian")
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
from alian.analysis.base.event import Event, get_selected_tracks, get_selected_particles
from alian.analysis.base.utils import read_yaml, is_slurm
from alian.io.data_io import DataInput

# source code locations
# /home/youqi/yasp/.workdir/bundle/hepbase/fastjet/3.4.2/fastjet-3.5.1/include/fastjet
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/RecursiveTools/SoftDrop.cc
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/RecursiveTools/include/fastjet/contrib/SoftDrop.hh
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/LundPlane/LundGenerator.cc
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/LundPlane/include/fastjet/contrib/LundGenerator.hh

class EmbeddingAnalysis:
    """Embed pp events from a pp file into OO events from an OO file.

    nev limits the number of event *pairs* processed.  Events that fail their
    respective event selectors are skipped without consuming a slot from the
    other source.
    """

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

        for k, v in self.cfg.get('analysis', {}).items():
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
        bge_def = fj.JetDefinition(fj.kt_algorithm, bge_R)
        bge_area_def = fj.AreaDefinition(
            fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(max_eta)
        )
        self.bge = fj.JetMedianBackgroundEstimator(
            bge_selector, bge_def, bge_area_def
        )
        print("bge:\n", self.bge.description())

        # --- jet finders ---
        pp_jf_opts = {**JetFinder._defaults, **pp_cfg.get('jet_finder', {})}
        self.pp_jet_finder = JetFinder(**pp_jf_opts)
        combined_jf_opts = {**JetFinder._defaults, **oo_cfg.get('jet_finder', {})}
        self.combined_jet_finder = JetFinder(**combined_jf_opts)
        self.R = pp_jf_opts['R']
        self.area_cut = 0.56*np.pi*self.R*self.R

        # --- substructure tools ---
        self.eec_trk_selector = fj.SelectorPtMin(self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()
        # self.sd = fj.contrib.SoftDrop(0, self.z_cut)

        # --- event displays ---
        self.n_event_displays = 30
        self.event_display_index = 0
        self.event_displays = []
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

        # --- pp particles, tracks, particle jets and det jets ---
        pp_particles = get_selected_particles(pp_ev, self.pp_selector.track)
        pp_tracks = get_selected_tracks(pp_ev, self.pp_selector.track)
        pp_particle_jets = self.pp_jet_finder.find_jets(pp_particles, use_area=False)
        pp_particle_jets = [j for j in pp_particle_jets if j.pt() >= self.pt_min_pp_jet]
        pp_jets_preselect = self.pp_jet_finder.find_jets(pp_tracks, use_area=False)
        pp_jets_preselect = [j for j in pp_jets_preselect if j.pt() >= self.pt_min_pp_jet]
        weight = pp_ev.data['weight']
        pTHat = pp_ev.data['ptHat']
        if (len(pp_particle_jets) > 0 and pp_particle_jets[0].pt() > 6 * pTHat):
            return
        if (len(pp_jets_preselect) > 0 and pp_jets_preselect[0].pt() > 4 * pTHat):
            return
        pp_jets = self.reject_high_track_pt_jets(pp_jets_preselect)

        # --- match pp det jets to pp partice jets ---
        pp_jet_matched_indices = self.match_pp_particle_to_pp_det(pp_particle_jets, pp_jets)
        pp_jets = [j for index, j in enumerate(pp_jets) if pp_jet_matched_indices[index] >= 0] 

        # !!! from now on, only look at pp jets with a truth match !!!
        pp_jet_tracks = []
        for j in pp_jets:
            pp_jet_tracks.extend(j.constituents())        
        
        [self.hists['pp_pT'].Fill(j.pt(), weight) for j in pp_jets]
        [self.hists['pp_n'].Fill(len(j.constituents()), weight) for j in pp_jets]
        for j in pp_jets:
            self.do_eec(j, "pp_eec", ew_denom=j.pt(), jet_pt_bin=j.pt(), event_weight=weight)
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(lund_seq, z_cut=self.z_cut) # class is LundDeclustering
            if l is None:
                continue
            gj = l.pair()
            if gj.pt() < self.pt_min_pp_jet:
                continue
            self.hists['pp_pTg'].Fill(gj.pt(), weight)
            subjet_a = l.harder()
            subjet_b = l.softer()
            self.do_eec(j, "pp_sdjet_eec", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight) # jet passes SD, but this includes stuff removed by SD
            # self.do_eec(gj, "sdjet_eec", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight) # jet passes SD, and this only includes stuff passes by SD
            self.do_eec(subjet_a, "pp_eec_aa", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
            self.do_eec(subjet_b, "pp_eec_bb", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
            self.do_eec_cross(subjet_a, subjet_b, "pp_eec_ab", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
            self.hists['pp_rg'].Fill(gj.pt(), delta_R(subjet_a, subjet_b), weight)
        
        # --- combined event: pp jet tracks + oo tracks ---
        oo_tracks = get_selected_tracks(oo_ev, self.oo_selector.track, index_offset=-9999)
        combined = self._combined_event(pp_jet_tracks, oo_tracks)

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
        self.hists['combined_njet_pp_njet'].Fill(len(pp_jets), len(combined_jets), weight)

        # Match the retained pp det jets to combined jets.
        # This starts a fresh matching pass, clearing the truth-match JetInfo.
        combined_jet_matched_indices = self.match_pp_det_to_combined(pp_jets, combined_jets)

        # --- fill cut flow histogram ---
        for combined_jet in combined_jets:
            self.hists['combined_sub_cutflow'].Fill(0.5, weight) # all combined jets
            combined_jet_info = combined_jet.user_info[alian.JetInfo]()
            if combined_jet_info.matching_candidates.size() > 0:
                self.hists['candidate_dR'].Fill(combined_jet_info.closest_jet_deltaR, weight)
            deltaR_matches = []
            for pp_jet in pp_jets:
                if self.is_geo_matched(combined_jet, pp_jet):
                    deltaR_matches.append(pp_jet)  
            if len(deltaR_matches) > 0:
                self.hists['combined_sub_cutflow'].Fill(1.5, weight) # combined jets with pp jets within delta R
                for pp_jet in deltaR_matches:
                    if self.mc_fraction(pp_jet, combined_jet) > self.mc_fraction_threshold:
                        self.hists['combined_sub_cutflow'].Fill(2.5, weight) # combined jets with pp jets within delta R and mc fraction
                        break
        for pp_ijet in combined_jet_matched_indices:
            if pp_ijet >= 0:
                self.hists['combined_sub_cutflow'].Fill(3.5, weight) # mutually unique matches

        # --- fill histograms per combined jet ---
        for combined_ijet in range(0, len(combined_jets)):

            j = combined_jets[combined_ijet]
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(lund_seq, z_cut=self.z_cut) # class is LundDeclustering
            if l is None:
                continue
            gj = l.pair()
            gj_pt_sub = gj.pt()-gj.area()*rho
            if gj_pt_sub < self.pt_min_combined_jet:
                continue
            # keep going if combined passes grooming
            
            pp_ijet = combined_jet_matched_indices[combined_ijet] # -1 if unmatched to pp
            combined_constituents = [t for t in j.constituents() if not t.is_pure_ghost()]
            combined_g_constituents = [t for t in gj.constituents() if not t.is_pure_ghost()]
            
            self.hists['combined_pTg'].Fill(gj.pt(), weight)
            self.hists['combined_pTg_sub_pTg'].Fill(gj_pt_sub, gj.pt(), weight)
            self.hists['combined_pT_sub_pT'].Fill(j.pt()-j.area()*rho, j.pt(), weight)
            self.hists['combined_area_sub_pT'].Fill(j.pt()-j.area()*rho, j.area(), weight)

            if (pp_ijet != -1): # mateched to pp

                pp_j = pp_jets[pp_ijet]
                pp_lund_seq = self.lund_gen.result(pp_j)
                pp_l = self.select_soft_drop(pp_lund_seq, z_cut=self.z_cut) # class is LundDeclustering
                if pp_l is None:
                    continue
                pp_gj = pp_l.pair()
                if pp_gj.pt() < self.pt_min_pp_jet:
                    continue
                # keep going if pp passes grooming

                self.hists['pp_pT_matched'].Fill(pp_j.pt(), weight)
                self.hists['pp_pTg_matched'].Fill(pp_gj.pt(), weight)
                self.hists['pp_pTg_pp_pT_matched'].Fill(pp_j.pt(), pp_gj.pt(), weight)

                self.hists['combined_pTg_matched'].Fill(gj_pt_sub, weight)
                self.hists['combined_pT_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.pt(), weight)
                self.hists['combined_area_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.area(), weight)
                self.hists['combined_pTg_sub_pTg_matched'].Fill(gj_pt_sub, gj.pt(), weight)

                self.hists['sub_pT_pp_pT_matched'].Fill(pp_j.pt(), j.pt()-j.area()*rho, weight)
                self.hists['sub_pTg_pp_pTg_matched'].Fill(pp_gj.pt(), gj_pt_sub, weight)
                self.hists['sub_n_pp_n_matched'].Fill(len(pp_j.constituents()), len(combined_constituents), weight)
                self.hists['sub_ng_pp_ng_matched'].Fill(len(pp_gj.constituents()), len(combined_g_constituents), weight)
                self.hists['residual_pp_pT_matched'].Fill(pp_j.pt(), j.pt()-j.area()*rho-pp_j.pt(), weight)
                self.hists['residual_sub_pT_matched'].Fill(j.pt()-j.area()*rho, j.pt()-j.area()*rho-pp_j.pt(), weight)
                self.hists['residual_pp_pTg_matched'].Fill(pp_gj.pt(), gj_pt_sub-pp_gj.pt(), weight)
                self.hists['residual_sub_pTg_matched'].Fill(gj_pt_sub, gj_pt_sub-pp_gj.pt(), weight)

                self.do_eec(pp_j, "pp_matched_eec", ew_denom=pp_j.pt(), jet_pt_bin=pp_j.pt(), event_weight=weight)
                pp_subjet_a = pp_l.harder()
                pp_subjet_b = pp_l.softer()
                self.do_eec(pp_j, "pp_matched_sdjet_eec", ew_denom=pp_gj.pt(), jet_pt_bin=pp_gj.pt(), event_weight=weight) # jet passes SD, but this includes stuff removed by SD
                # self.do_eec(pp_gj, "sdjet_eec", ew_denom=pp_gj.pt(), jet_pt_bin=pp_gj.pt(), event_weight=weight) # jet passes SD, and this only includes stuff passes by SD
                self.do_eec(pp_subjet_a, "pp_matched_eec_aa", ew_denom=pp_gj.pt(), jet_pt_bin=pp_gj.pt(), event_weight=weight)
                self.do_eec(pp_subjet_b, "pp_matched_eec_bb", ew_denom=pp_gj.pt(), jet_pt_bin=pp_gj.pt(), event_weight=weight)
                self.do_eec_cross(pp_subjet_a, pp_subjet_b, "pp_matched_eec_ab", ew_denom=pp_gj.pt(), jet_pt_bin=pp_gj.pt(), event_weight=weight)
                self.hists['pp_matched_rg'].Fill(pp_gj.pt(), delta_R(pp_subjet_a, pp_subjet_b), weight)
                
                self.do_eec(j, "combined_matched_eec", ew_denom=j.pt()-j.area()*rho, jet_pt_bin=j.pt()-j.area()*rho, event_weight=weight)
                subjet_a = l.harder()
                subjet_b = l.softer()
                self.do_eec(j, "combined_matched_sdjet_eec", ew_denom=gj_pt_sub, jet_pt_bin=gj_pt_sub, event_weight=weight) # jet passes SD, but this includes stuff removed by SD
                # self.do_eec(gj, "sdjet_eec", ew_denom=gj_pt_sub, jet_pt_bin=gj_pt_sub, event_weight=weight) # jet passes SD, and this only includes stuff passes by SD
                self.do_eec(subjet_a, "combined_matched_eec_aa", ew_denom=gj_pt_sub, jet_pt_bin=gj_pt_sub, event_weight=weight)
                self.do_eec(subjet_b, "combined_matched_eec_bb", ew_denom=gj_pt_sub, jet_pt_bin=gj_pt_sub, event_weight=weight)
                self.do_eec_cross(subjet_a, subjet_b, "combined_matched_eec_ab", ew_denom=gj_pt_sub, jet_pt_bin=gj_pt_sub, event_weight=weight)
                self.hists['combined_matched_rg'].Fill(gj_pt_sub, delta_R(subjet_a, subjet_b), weight)

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    def reject_high_track_pt_jets(self, jets_preselect):
        jets = []
        for j in jets_preselect:
            highpttrack = False
            if (j.pt() < 100):
                jets.append(j)
                continue
            for c in j.constituents(): # only check jets with jet pT >= 100 GeV
                if c.pt() > 100:
                    highpttrack = True
                    break
            if not highpttrack:
                jets.append(j)
        return jets

    def match_pp_particle_to_pp_det(self, particle_jets, det_jets):
        """Return the particle-jet index for each det jet, or -1."""
        return self.match_jets(particle_jets, det_jets)

    def match_pp_det_to_combined(self, det_jets, combined_jets):
        """Return the det-jet index for each combined jet, or -1."""
        return self.match_jets(det_jets, combined_jets, require_mc_fraction=True)

    def match_jets(self, source_jets, target_jets, require_mc_fraction=False):
        """Find mutually unique matches; return one source index per target jet.

        Every pass replaces JetInfo on both collections. Previous matching
        metadata is discarded; returned index lists remain valid for the
        collection ordering used in that pass.

        All candidates must pass the geometrical cut. Embedding candidates
        must additionally pass mc_fraction(source, target) > threshold.
        """
        for jets in (source_jets, target_jets):
            for jet in jets:
                jet.set_user_info(alian.JetInfo())

        for source_index, source_jet in enumerate(source_jets):
            for target_index, target_jet in enumerate(target_jets):
                if not self.is_geo_matched(source_jet, target_jet):
                    continue
                if require_mc_fraction:
                    if not (self.mc_fraction(source_jet, target_jet) > self.mc_fraction_threshold):
                        continue
                deltaR = source_jet.delta_R(target_jet)
                self.set_jet_info(source_jet, target_index, deltaR)
                self.set_jet_info(target_jet, source_index, deltaR)

        matched_indices = [-1 for _ in target_jets]
        for source_index, source_jet in enumerate(source_jets):
            if not self.is_match_unique(source_jet, target_jets):
                continue
            source_info = source_jet.user_info[alian.JetInfo]()
            target_index = source_info.closest_jet_index
            target_info = target_jets[target_index].user_info[alian.JetInfo]()
            source_info.match_index = target_index
            target_info.match_index = source_index
            matched_indices[target_index] = source_index

        return matched_indices

    #---------------------------------------------------------------
    # Set 'jet_match' as a matching candidate in user_info of 'jet'
    #
    # Adapted directly from:
    # pyjetty/alice_analysis/process/base/process_base.py
    #---------------------------------------------------------------
    def set_jet_info(self, jet, jet_match_index, deltaR):
        if jet.has_user_info[alian.JetInfo]():
            jet_user_info = jet.user_info[alian.JetInfo]()
        else:
            jet_user_info = alian.JetInfo()
            jet.set_user_info(jet_user_info)
        jet_user_info.matching_candidates.push_back(jet_match_index)
        if deltaR < jet_user_info.closest_jet_deltaR:
            jet_user_info.closest_jet_index = jet_match_index
            jet_user_info.closest_jet_deltaR = deltaR
    #---------------------------------------------------------------
    # Return whether a jet has a unique match
    #---------------------------------------------------------------
    def is_match_unique(self, jet, matched_jets):
        if jet.has_user_info[alian.JetInfo]():
            jet_info = jet.user_info[alian.JetInfo]()
            if jet_info.matching_candidates.size() == 1:
                jet_match = matched_jets[jet_info.closest_jet_index]
                if jet_match.has_user_info[alian.JetInfo]():
                    jet_match_info = jet_match.user_info[alian.JetInfo]()
                    if jet_match_info.matching_candidates.size() == 1:
                        return True
        return False

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

    def do_eec(self, jet, hist_name, ew_denom=None, jet_pt_bin=None, event_weight=1.0):
        if ew_denom is None:
            ew_denom = jet.pt()
        if jet_pt_bin is None:
            jet_pt_bin = jet.pt()
        tracks = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(tracks, 2):
            ew = p1.pt() * p2.pt() / ew_denom / ew_denom
            rl = delta_R(p1, p2)
            self.hists[hist_name].Fill(jet_pt_bin, rl, ew*event_weight)

    def do_eec_cross(self, subjet_a, subjet_b, hist_name, ew_denom, jet_pt_bin, event_weight=1.0):
        tracks_a = self.eec_trk_selector(subjet_a.constituents())
        tracks_b = self.eec_trk_selector(subjet_b.constituents())
        for p1 in tracks_a:
            for p2 in tracks_b:
                ew = p1.pt() * p2.pt() / ew_denom / ew_denom
                rl = delta_R(p1, p2)
                self.hists[hist_name].Fill(jet_pt_bin, rl, ew*event_weight)

    def select_soft_drop(self, lund_seq, z_cut=0.1):
        """
        Return the first primary Lund splitting satisfying z > z_cut (soft-drop
        condition with beta=0), or None if no splitting passes.
        """
        for l in lund_seq:
            if l.z() > z_cut:
                return l
        return None
        
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
        '-t2', '--oo-tree-struct', type=str, default="config/run3_tstruct.yaml",
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
