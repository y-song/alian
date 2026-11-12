#!/usr/bin/env python3
"""
Modified based on embed_perpcone.py: the background is removed from the combined
event with constituent subtraction (CS) instead of area subtraction. The pp
detector-level jet tracks are embedded in an OO event; rho is estimated on the
combined event (bkg_estimator config block), the combined event is subtracted
with the ConstituentSubtractor (constituent_subtractor block, sharing that rho
estimator) and the subtracted event is clustered with explicit ghosts. Combined
jets are therefore already subtracted: j.pt() is the subtracted pT, there is no
rho * area term (get_subjets is called with rho = 0), and
analysis jet_finder pT_min / eta_max of the oo block act on the subtracted jets.
Histograms with the "sub_" prefix hold subtracted quantities:
  sub_pT                              subtracted combined jet pT
  sub_pTg_matched                     subtracted groomed pT of matched combined jets
  sub_pT_pp_pT_matched, sub_pTg_pp_pTg_matched   subtracted vs pp pT (pTg)
  residual_sub_pT[g]_matched          subtracted - pp pT vs subtracted pT (pTg)
  residual_pp_pTg_matched             subtracted - pp pTg vs pp pTg
The raw (unsubtracted) combined_pT* and combined_area_* histograms of
embed_perpcone.py are not filled. The "closest ghost" and perp-cone terms below are
unchanged, except that the perp cones are taken from the subtracted event.

EEC weights use the pre-CS pp pT for embedded signal particles (user_index >= 0, see
_true_pt), so CS does not rescale the signal; OO particles keep their CS-subtracted
pT. The EEC track pT cut (pt_min_eec) is applied to that same pT. Jet pT and
jet / subjet membership still come from the subtracted event.

Matching to pp jets, the true-pT lookup and the sigsig/sigbkg/bkgbkg split below rely on user_index
being kept by the subtractor (pp: >= 0, OO: < 0).

Plus perpendicular-cone background estimates for the
matched combined subjet EECs (combined_matched_eec_aa/_ab/_bb) and inclusive EEC
(combined_matched_sdjet_eec, for jets that pass SD).

For each matched combined jet passing SD, two perp cones of radius perpcone_R
are taken from the subtracted combined event at phi +/- pi/2 of the ungroomed jet axis and
rotated under the jet. Each rotated perp-cone particle is assigned to subjet A
or B according to the closest ghost of the (ungroomed) jet; particles whose
closest ghost was groomed away are dropped.

Filled per subjet pair type xy in (aa, ab, bb), with P1 (P2) the perp cone at
phi + pi/2 (phi - pi/2):
  combined_matched_eec_xy_jetperp     jet-P1 cross pairs
  combined_matched_eec_xy_perpperp    P1-P1 pairs
  combined_matched_eec_xy_perp1perp2  P1-P2 cross pairs
The jet-jet term is combined_matched_eec_xy itself. Pair orientations follow
combined_matched_eec_xy (both orders for aa/bb, one A-B orientation for ab), so
  signal_xy ~ eec_xy - (jetperp_xy - perp1perp2_xy) - perpperp_xy

For closure, combined_matched_eec_xy is also split by the truth origin of the
two tracks (pp: user_index >= 0, OO: user_index < 0):
  combined_matched_eec_xy_sigsig      pp-pp pairs   (compare to signal_xy)
  combined_matched_eec_xy_sigbkg      pp-OO pairs   (compare to jetperp_xy - perp1perp2_xy)
  combined_matched_eec_xy_bkgbkg      OO-OO pairs   (compare to perpperp_xy)
with sigsig + sigbkg + bkgbkg = combined_matched_eec_xy.

To check that the ghost density (oo.jet_finder.ghost_area) is sufficient for the
A/B assignment, the following are also filled (see _fill_ghost_diagnostics):
  combined_matched_perp_ghost_dR_{a,b,none}  dR of perp particle to closest ghost
  combined_matched_perp_assign[_rg]          perp particle category (A, B, neither)
  combined_matched_area_{a,b}                area of subjet A / B
Rerun with decreasing ghost_area: the assignment fractions and perp-cone
EEC terms should converge. dR to the closest ghost scales as sqrt(ghost_area) by
construction and does not converge.

With analysis.perp_recluster_closure: true, the perp-cone particles are also
assigned by reclustering the jet's constituents with scaled copies of them (see
_recluster_assign), filling the same assignment histograms for comparison:
  combined_matched_perp_assign[_rg]_recl     perp particle category (A, B, neither)

Example usage:
python analysis/test/embed_cs.py -i1 /rstorage/alice/run3/mc_central/LHC26b6/BerkeleyTree_564356_0.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed_cs.yaml -o output/embed_cs.root
python analysis/test/embed_cs.py -i1 /rstorage/youqi/MC_pp_anchored_OO_hadd50/combined1.root -i2 /rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees/1/BerkeleyTree.root -c config/embed_cs.yaml -o output/embed_test.root
python analysis/test/embed_cs.py -i1 /home/youqi/temp/combined1.root -i2 /home/youqi/temp/BerkeleyTree.root -c config/embed_cs.yaml -o output/embed_test.root
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
from alian.analysis.base.bkg_estimator import BkgEstimator
from alian.analysis.base.output import Output
from alian.analysis.base.selector import TrackSelector
from alian.analysis.base.event import Event, get_selected_tracks, get_selected_particles
from alian.analysis.base.utils import read_yaml, is_slurm
from alian.io.data_io import DataInput

# ghost assignment categories (values filled on the "assign" axis)
ASSIGN_A, ASSIGN_B, ASSIGN_NONE = 0, 1, 2
# perp-cone particles are reclustered as scaled copies with negligible pT, tagged by
# user_index = offset - its index
RECLUSTER_SCALE = 1e-50
PERP_INDEX_OFFSET = -10_000_000

# source code locations
# /home/youqi/yasp/.workdir/bundle/hepbase/fastjet/3.4.2/fastjet-3.5.1/include/fastjet
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/RecursiveTools/SoftDrop.cc
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/RecursiveTools/include/fastjet/contrib/SoftDrop.hh
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/LundPlane/LundGenerator.cc
# /home/youqi/yasp/.workdir/bundle/hepbase/fjcontrib/mp/fjcontrib-master/LundPlane/include/fastjet/contrib/LundGenerator.hh

class EmbeddingAnalysis:
    """Embed pp events from a pp file into OO events from an OO file.

    nev limits the number of event *pairs* processed. Events that fail their
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

        # --- background estimator ---
        self.bkg_estimator = BkgEstimator(self.cfg.get('bkg_estimator') or {}, self.logger)
        print("bkg_estimator:\n", self.bkg_estimator.bge.description())

        # --- constituent subtractor ---
        self.constituent_subtractor = CEventSubtractor(self.cfg.get('constituent_subtractor') or {}, self.logger, bkg_estimator=self.bkg_estimator)
        print("constituent_subtractor:\n", self.constituent_subtractor.subtractor.description())

        # --- jet finders ---
        self.pp_particle_jet_finder = JetFinder.load(pp_cfg)
        self.pp_jet_finder = JetFinder.load(pp_cfg)
        self.combined_jet_finder = JetFinder.load(oo_cfg)
        self.R = self.pp_jet_finder.R
        self.area_cut = 0.56*np.pi*self.R*self.R

        # --- substructure tools ---
        self.eec_trk_selector = TrackSelector(pt_min=self.pt_min_eec)
        self.lund_gen = fj.contrib.LundGenerator()
        # C/A with R large enough to merge all inputs into one jet (perp_recluster_closure)
        self.recluster_jet_def = fj.JetDefinition(fj.cambridge_algorithm, fj.JetDefinition.max_allowable_R)
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
        pp_particle_jets = self.pp_particle_jet_finder.find_jets(pp_particles, use_area=False)
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
        [self.hists['pp_pT'].Fill(j.pt(), weight) for j in pp_jets]
        [self.hists['pp_n'].Fill(len(j.constituents()), weight) for j in pp_jets]
        pp_jet_tracks = [c for j in pp_jets for c in j.constituents()]
        # pre-CS pp pT of each embedded signal particle, keyed by user_index (unique per pp track); used by _true_pt
        self._signal_pt = {c.user_index(): c.pt() for c in pp_jet_tracks}
        pp_subjets_list = [self.get_subjets(j, self.pt_min_pp_jet) for j in pp_jets] # one entry per jet: (gj, gj_pt, subjet_a, subjet_b), or None if that jet fails grooming
        
        for j, subjets in zip(pp_jets, pp_subjets_list):
            self.do_eec(j, "pp_eec", ew_denom=j.pt(), jet_pt_bin=j.pt(), event_weight=weight)
            if subjets is None:
                continue
            gj, gj_pt, subjet_a, subjet_b = subjets
            self.hists['pp_pTg'].Fill(gj_pt, weight)
            self.do_subjet_eec(j, subjet_a, subjet_b, "pp", ew_denom=gj_pt, jet_pt_bin=gj_pt, event_weight=weight)

        # --- combined event: pp jet tracks + oo tracks ---
        oo_tracks = get_selected_tracks(oo_ev, self.oo_selector.track, index_offset=-9999)
        combined = self._combined_event(pp_jet_tracks, oo_tracks)

        # --- rho from bkg_estimator on the combined event ---
        rho, sigma = self.bkg_estimator.estimate(combined)
        self.hists['cent_rho'].Fill(rho, oo_ev.data['centrality'])
        sub_combined = self.constituent_subtractor.process_event(combined)
        
        # --- combined jets: cluster combined event ---
        combined_jets = self.combined_jet_finder.find_jets(sub_combined, use_area=True)
        [self.hists['sub_pT'].Fill(j.pt(), weight) for j in combined_jets]
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
            subjets = self.get_subjets(j, self.pt_min_combined_jet, rho=0.0)
            if subjets is None:
                continue
            gj, gj_pt, subjet_a, subjet_b = subjets # subjet_a is harder after rhoA subtraction
            # keep going if combined passes grooming

            pp_ijet = combined_jet_matched_indices[combined_ijet] # -1 if unmatched to pp
            combined_constituents = [t for t in j.constituents() if not t.is_pure_ghost()]
            combined_g_constituents = [t for t in gj.constituents() if not t.is_pure_ghost()]

            if (pp_ijet != -1): # mateched to pp

                pp_j = pp_jets[pp_ijet]
                pp_subjets = pp_subjets_list[pp_ijet]
                if pp_subjets is None:
                    continue
                pp_gj, pp_gj_pt, pp_subjet_a, pp_subjet_b = pp_subjets
                # keep going if pp passes grooming

                self.hists['pp_pT_matched'].Fill(pp_j.pt(), weight)
                self.hists['pp_pTg_matched'].Fill(pp_gj.pt(), weight)
                self.hists['pp_pTg_pp_pT_matched'].Fill(pp_j.pt(), pp_gj.pt(), weight)

                self.hists['sub_pTg_matched'].Fill(gj.pt(), weight)
                self.hists['sub_pT_pp_pT_matched'].Fill(pp_j.pt(), j.pt(), weight)
                self.hists['sub_pTg_pp_pTg_matched'].Fill(pp_gj.pt(), gj.pt(), weight)
                self.hists['sub_n_pp_n_matched'].Fill(len(pp_j.constituents()), len(combined_constituents), weight)
                self.hists['sub_ng_pp_ng_matched'].Fill(len(pp_gj.constituents()), len(combined_g_constituents), weight)
                self.hists['residual_pp_pT_matched'].Fill(pp_j.pt(), j.pt()-pp_j.pt(), weight)
                self.hists['residual_sub_pT_matched'].Fill(j.pt(), j.pt()-pp_j.pt(), weight)
                self.hists['residual_pp_pTg_matched'].Fill(pp_gj.pt(), gj.pt()-pp_gj.pt(), weight)
                self.hists['residual_sub_pTg_matched'].Fill(gj.pt(), gj.pt()-pp_gj.pt(), weight)

                self.do_eec(pp_j, "pp_matched_eec", ew_denom=pp_j.pt(), jet_pt_bin=pp_j.pt(), event_weight=weight)
                self.do_subjet_eec(pp_j, pp_subjet_a, pp_subjet_b, "pp_matched", ew_denom=pp_gj_pt, jet_pt_bin=pp_gj_pt, event_weight=weight)

                self.do_eec(j, "combined_matched_eec", ew_denom=j.pt(), jet_pt_bin=j.pt(), event_weight=weight)
                self.do_subjet_eec(j, subjet_a, subjet_b, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
                self.do_subjet_eec_truth(j, subjet_a, subjet_b, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
                self.do_subjet_eec_perpcone(j, subjet_a, subjet_b, sub_combined, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)

    # -----------------------------------------------------------------------
    # event building
    # -----------------------------------------------------------------------

    def _combined_event(self, pp_tracks, oo_tracks):
        """Merge pp tracks with background tracks."""
        combined = list(oo_tracks)
        combined.extend(pp_tracks)
        return std.vector[fj.PseudoJet](combined)

    # -----------------------------------------------------------------------
    # pp jet selection and jet matching
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
                deltaR = delta_R(source_jet, target_jet)
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

    def set_jet_info(self, jet, jet_match_index, deltaR):
        """
        Add jet_match_index as a matching candidate in the JetInfo of jet, and
        update the closest match
        Adapted directly from:
        pyjetty/alice_analysis/process/base/process_base.py
        """
        if jet.has_user_info[alian.JetInfo]():
            jet_user_info = jet.user_info[alian.JetInfo]()
        else:
            jet_user_info = alian.JetInfo()
            jet.set_user_info(jet_user_info)
        jet_user_info.matching_candidates.push_back(jet_match_index)
        if deltaR < jet_user_info.closest_jet_deltaR:
            jet_user_info.closest_jet_index = jet_match_index
            jet_user_info.closest_jet_deltaR = deltaR

    def is_match_unique(self, jet, matched_jets):
        """
        Return whether a jet has a unique match
        Adapted directly from:
        pyjetty/alice_analysis/process/base/process_base.py
        """
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
        deltaR = delta_R(jet1, jet2)

        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < 0.6 * self.R:
            return True
        else:
            return False

    def mc_fraction(self, jet_pp, jet_combined):
        """
        Return pt-fraction of tracks in jet_pp that are contained in jet_combined
        """
        pt_total = jet_pp.pt()
        pt_contained = 0.
        for track in jet_combined.constituents():
          if track.user_index() >= 0:
            pt_contained += track.pt()
        return pt_contained/pt_total

    # -----------------------------------------------------------------------
    # grooming
    # -----------------------------------------------------------------------

    def select_soft_drop(self, lund_seq, z_cut=0.1, rho=0.0):
        """
        Return (l, swapped) for the first primary Lund splitting l satisfying
        z > z_cut (soft-drop condition with beta=0), or None if no splitting passes.

        If rho > 0, z is computed from the area-subtracted branch pTs
        (pt - rho*area); a branch with subtracted pT <= 0 fails. swapped is True
        if the subtracted pT of l.softer() exceeds that of l.harder().
        If rho = 0 (jets clustered without area), the raw pTs are used.

        Note: the primary sequence still follows the branch that is harder by
        raw pT (LundGenerator), which can differ from the harder by subtracted
        pT when the raw pTs are close and the areas very different.
        """
        for l in lund_seq:
            h, s = l.harder(), l.softer()
            pt_h = h.pt() - rho*h.area() if rho else h.pt()
            pt_s = s.pt() - rho*s.area() if rho else s.pt()
            pt_hi, pt_lo = max(pt_h, pt_s), min(pt_h, pt_s)
            if pt_lo <= 0:
                continue
            if pt_lo / (pt_hi + pt_lo) > z_cut:
                return l, pt_s > pt_h
        return None

    def get_subjets(self, jet, pt_min, rho=0.0):
        """
        Groom jet with soft drop (z_cut) on its Lund declustering, with the SD
        condition evaluated on area-subtracted branch pTs if rho > 0.

        Return (groomed jet, groomed pT - rho*area, harder subjet, softer subjet),
        or None if no splitting passes SD or the (subtracted) groomed pT < pt_min.
        Harder/softer are ordered by subtracted pT. rho = 0 for jets clustered
        without area.
        """
        sd = self.select_soft_drop(self.lund_gen.result(jet), z_cut=self.z_cut, rho=rho)
        if sd is None:
            return None
        l, swapped = sd # l is a LundDeclustering
        gj = l.pair()
        gj_pt = gj.pt() - rho*gj.area() if rho else gj.pt()
        if gj_pt < pt_min:
            return None
        if swapped:
            return gj, gj_pt, l.softer(), l.harder()
        return gj, gj_pt, l.harder(), l.softer()

    # -----------------------------------------------------------------------
    # EECs
    # -----------------------------------------------------------------------

    def do_eec(self, jet, hist_name, *, ew_denom, jet_pt_bin, event_weight=1.0):
        tracks = self._eec_tracks(jet.constituents())
        self._fill_eec_pairs(itertools.permutations(tracks, 2), hist_name, ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def do_eec_cross(self, subjet_a, subjet_b, hist_name, *, ew_denom, jet_pt_bin, event_weight=1.0):
        tracks_a = self._eec_tracks(subjet_a.constituents())
        tracks_b = self._eec_tracks(subjet_b.constituents())
        self._fill_eec_pairs(itertools.product(tracks_a, tracks_b), hist_name, ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def do_subjet_eec(self, jet, subjet_a, subjet_b, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """Fill {prefix}_sdjet_eec, _eec_aa, _eec_bb, _eec_ab and _rg, binned by groomed pT."""
        self.do_eec(jet, f"{prefix}_sdjet_eec", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight) # jet passes SD, but this includes stuff removed by SD
        self.do_eec(subjet_a, f"{prefix}_eec_aa", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.do_eec(subjet_b, f"{prefix}_eec_bb", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.do_eec_cross(subjet_a, subjet_b, f"{prefix}_eec_ab", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.hists[f'{prefix}_rg'].Fill(jet_pt_bin, delta_R(subjet_a, subjet_b), event_weight)

    def do_subjet_eec_truth(self, jet, subjet_a, subjet_b, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """
        Split {prefix}_sdjet_eec, _eec_aa, _eec_bb, _eec_ab by the truth origin of the pair:
        fills {prefix}_<region>_sigsig / _sigbkg / _bkgbkg (signal = user_index >= 0).
        Same pairs and weights as do_subjet_eec, so the three sum to {prefix}_<region>.
        """
        tracks = self._eec_tracks(jet.constituents())
        tracks_a = self._eec_tracks(subjet_a.constituents())
        tracks_b = self._eec_tracks(subjet_b.constituents())
        pair_types = ('bkgbkg', 'sigbkg', 'sigsig') # indexed by number of signal tracks in the pair
        for region, pairs in (
            ('sdjet_eec', itertools.permutations(tracks, 2)),
            ('eec_aa', itertools.permutations(tracks_a, 2)),
            ('eec_bb', itertools.permutations(tracks_b, 2)),
            ('eec_ab', itertools.product(tracks_a, tracks_b)),
        ):
            for p1, p2 in pairs:
                pair_type = pair_types[(p1.user_index() >= 0) + (p2.user_index() >= 0)] # adding 2 booleans, index will be 0, 1 or 2
                ew = self._true_pt(p1) * self._true_pt(p2) / ew_denom / ew_denom
                self.hists[f'{prefix}_{region}_{pair_type}'].Fill(jet_pt_bin, delta_R(p1, p2), ew*event_weight)

    def do_subjet_eec_perpcone(self, jet, subjet_a, subjet_b, combined, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """
        Fill perp-cone EEC terms for the SD jet and for subjets A and B of jet.

        Perp cones are placed at phi +/- pi/2 of the ungroomed jet axis, rotated
        under the jet, and their particles are assigned to A or B via the
        closest ghost. See the module docstring for the histograms filled.
        Also fills the ghost-assignment diagnostics (_fill_ghost_diagnostics).
        """
        perp1 = self.eec_trk_selector.select_from(self._perpcone_parts(jet, combined, +np.pi/2)) # a list of PseudoJets
        perp2 = self.eec_trk_selector.select_from(self._perpcone_parts(jet, combined, -np.pi/2))
        ghosts = self._jet_ghosts(jet) # a list of PseudoJets
        perp1_a, perp1_b, perp1_info = self._assign_to_subjets(perp1, ghosts, subjet_a, subjet_b) # lists of PseudoJets, list of (category, dR) tuples
        perp2_a, perp2_b, perp2_info = self._assign_to_subjets(perp2, ghosts, subjet_a, subjet_b)
        self._fill_ghost_diagnostics(perp1_info + perp2_info, subjet_a, subjet_b, prefix, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        if getattr(self, 'perp_recluster_closure', False):
            perp_categories = self._recluster_assign(jet, perp1+perp2, subjet_a, subjet_b)
            self._fill_recluster_assign(perp_categories, subjet_a, subjet_b, prefix, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

        tracks = self._eec_tracks(jet.constituents())
        tracks_a = self._eec_tracks(subjet_a.constituents())
        tracks_b = self._eec_tracks(subjet_b.constituents())

        terms = [] # (region, jet-P1 pairs, P1-P1 pairs, P1-P2 pairs)
        # inclusive, eec_aa and eec_bb
        for region, jet_tracks, p1, p2 in (
            ('sdjet_eec', tracks, perp1, perp2),
            ('eec_aa', tracks_a, perp1_a, perp2_a),
            ('eec_bb', tracks_b, perp1_b, perp2_b),
        ):
            terms.append((
                region,
                itertools.chain(itertools.product(jet_tracks, p1), itertools.product(p1, jet_tracks)),
                itertools.permutations(p1, 2),
                itertools.chain(itertools.product(p1, p2), itertools.product(p2, p1)),
            ))
        # eec_ab
        terms.append((
            'eec_ab',
            itertools.chain(itertools.product(tracks_a, perp1_b), itertools.product(perp1_a, tracks_b)),
            itertools.product(perp1_a, perp1_b),
            itertools.chain(itertools.product(perp1_a, perp2_b), itertools.product(perp2_a, perp1_b)),
        ))

        for region, jetperp, perpperp, perp1perp2 in terms:
            self._fill_eec_pairs(jetperp, f"{prefix}_{region}_jetperp", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
            self._fill_eec_pairs(perpperp, f"{prefix}_{region}_perpperp", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
            self._fill_eec_pairs(perp1perp2, f"{prefix}_{region}_perp1perp2", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def _true_pt(self, part):
        """pT used for the EEC weight: the pre-CS pp pT for an embedded signal particle
        (user_index >= 0, as set before the subtractor), the particle's own pT otherwise.
        Relies on the subtractor keeping user_index; self._signal_pt is set per event."""
        if part.user_index() >= 0:
            return self._signal_pt.get(part.user_index(), part.pt())
        return part.pt()

    def _eec_tracks(self, parts):
        """Particles passing the EEC track selection, with the pT cut applied to _true_pt."""
        return [t for t in parts if self._true_pt(t) >= self.pt_min_eec]

    def _fill_eec_pairs(self, pairs, hist_name, *, ew_denom, jet_pt_bin, event_weight=1.0):
        for p1, p2 in pairs:
            ew = self._true_pt(p1) * self._true_pt(p2) / ew_denom / ew_denom
            rl = delta_R(p1, p2)
            self.hists[hist_name].Fill(jet_pt_bin, rl, ew*event_weight)

    def _perpcone_parts(self, ref_jet, parts, dphi):
        """Return copies of parts within perpcone_R of ref_jet rotated by dphi, rotated back by -dphi."""
        perpcone = fj.PseudoJet()
        perpcone.reset_PtYPhiM(ref_jet.perp(), ref_jet.rapidity(), ref_jet.phi() + dphi, ref_jet.m())
        rotated = []
        for part in parts:
            if delta_R(perpcone, part) > self.perpcone_R:
                continue
            p = fj.PseudoJet()
            p.reset_PtYPhiM(self._true_pt(part), part.rapidity(), part.phi() - dphi, part.m())
            p.set_user_index(part.user_index())
            rotated.append(p)
        return rotated

    def _jet_ghosts(self, jet):
        """Return the explicit ghosts in jet."""
        return [c for c in jet.constituents() if c.is_pure_ghost()]

    def _assign_to_subjets(self, parts, ghosts, subjet_a, subjet_b):
        """
        Assign each part to subjet A or B by which subjet holds its closest ghost.

        Returns (parts_a, parts_b, info):
          parts_a, parts_b  lists of PseudoJets, the subsets of parts assigned to A / B
          info              list of (category, dR) tuples, one per part, with category
                            ASSIGN_A/ASSIGN_B/ASSIGN_NONE and dR to the closest ghost
                            (-1.0 if there are no ghosts)

        Ghosts are matched by (eta, phi), since the reclustered Lund subjets do not
        keep cluster_hist_index.
        """
        if len(ghosts) == 0:
            return [], [], [(ASSIGN_NONE, -1.0) for _ in parts]
        ghosts_a = {(c.eta(), c.phi()) for c in subjet_a.constituents() if c.is_pure_ghost()} # a set of tuples
        ghosts_b = {(c.eta(), c.phi()) for c in subjet_b.constituents() if c.is_pure_ghost()}
        parts_a, parts_b, info = [], [], []
        for p in parts:
            # for each p, find the ghost g that gives the min of delta_R(g, p). assign this ghost to "closest"
            dR, closest = min(((delta_R(g, p), g) for g in ghosts), key=lambda x: x[0]) # x is a tuple, x[0] is delta_R(g, p) and x[1] is g
            ghost = (closest.eta(), closest.phi())
            if ghost in ghosts_a:
                parts_a.append(p)
                info.append((ASSIGN_A, dR))
            elif ghost in ghosts_b:
                parts_b.append(p)
                info.append((ASSIGN_B, dR))
            else:
                info.append((ASSIGN_NONE, dR))
        return parts_a, parts_b, info

    def _fill_ghost_diagnostics(self, perp_info, subjet_a, subjet_b, prefix, *, jet_pt_bin, event_weight=1.0):
        """
        Diagnostics for the ghost-based A/B assignment, to check convergence in
        ghost_area:
          {prefix}_perp_ghost_dR_{a,b,none}  dR of perp-cone particle to its closest ghost vs pT_g
          {prefix}_perp_assign               assignment category of perp-cone particles vs pT_g
          {prefix}_perp_assign_rg            assignment category of perp-cone particles vs R_g
          {prefix}_area_{a,b}                area of subjet A / B vs pT_g
        """
        rg = delta_R(subjet_a, subjet_b)
        suffixes = {ASSIGN_A: 'a', ASSIGN_B: 'b', ASSIGN_NONE: 'none'}
        for category, dR in perp_info:
            self.hists[f'{prefix}_perp_ghost_dR_{suffixes[category]}'].Fill(jet_pt_bin, dR, event_weight)
            self.hists[f'{prefix}_perp_assign'].Fill(jet_pt_bin, category, event_weight)
            self.hists[f'{prefix}_perp_assign_rg'].Fill(rg, category, event_weight)

        self.hists[f'{prefix}_area_a'].Fill(jet_pt_bin, subjet_a.area(), event_weight)
        self.hists[f'{prefix}_area_b'].Fill(jet_pt_bin, subjet_b.area(), event_weight)

    def _recluster_assign(self, jet, perp, subjet_a, subjet_b):
        """
        Exact A/B assignment of perp-cone particles, as a closure test of the
        closest-ghost assignment.

        Scaled copies (momentum x RECLUSTER_SCALE) of the perp cone particles are
        clustered with C/A together with the real (non-ghost) constituents of jet,
        into a single jet, which the LundGenerator then declusters. The copies have
        negligible pT, so they do not change the C/A merging of the real particles
        or the SD splitting; each copy ends up where a soft particle at its position
        would.

        the ghost method can put a particle in ASSIGN_NONE because it’s outside the jet,
        but the reclustering can only do that if the particle is groomed away.

        Returns a list of size len(perp), one entry per perp cone particle:
        ASSIGN_A or ASSIGN_B if its copy is in subjet A / B, ASSIGN_NONE if
        it is groomed away.
        """
        inputs = [c for c in jet.constituents() if not c.is_pure_ghost()]
        for i, p in enumerate(perp):
            copy = fj.PseudoJet(p.px() * RECLUSTER_SCALE, p.py() * RECLUSTER_SCALE,
                                p.pz() * RECLUSTER_SCALE, p.e() * RECLUSTER_SCALE)
            copy.set_user_index(PERP_INDEX_OFFSET - i)
            inputs.append(copy)
        # cs must stay alive while its jets are used
        cs = fj.ClusterSequence(std.vector[fj.PseudoJet](inputs), self.recluster_jet_def)
        test_jet = fj.sorted_by_pt(cs.inclusive_jets())[0] # the only jet, with R = max_allowable_R

        # find the test subjets already chosen by get_subjets() on the original jet: the splitting
        # whose branches have the pTs of subjet_a (harder one after rhoA subtraction) and subjet_b.
        # test_subjet constituents are: combined jet constituents (no ghosts) + rotated perp cone
        # particles with pT -> 0
        # subjet constituents are: combined jet constituents (with ghosts)
        test_subjet_a = test_subjet_b = None
        for l in self.lund_gen.result(test_jet):
            if self._same_pt(l.harder(), subjet_a) and self._same_pt(l.softer(), subjet_b):
                test_subjet_a, test_subjet_b = l.harder(), l.softer()
                break
            if self._same_pt(l.harder(), subjet_b) and self._same_pt(l.softer(), subjet_a): # subjet_a has lower raw pT
                test_subjet_a, test_subjet_b = l.softer(), l.harder()
                break
        if test_subjet_a is None:
            print("unexpected")

        categories = [ASSIGN_NONE] * len(perp)
        for category, branch in ((ASSIGN_A, test_subjet_a), (ASSIGN_B, test_subjet_b)):
            for c in branch.constituents():
                index = c.user_index()
                if index <= PERP_INDEX_OFFSET:
                    categories[PERP_INDEX_OFFSET - index] = category

        return categories

    @staticmethod
    def _same_pt(j1, j2, rel_tol=1e-9):
        """Whether j1 and j2 have the same pT within relative tolerance rel_tol."""
        return abs(j1.pt() - j2.pt()) <= rel_tol * max(j1.pt(), j2.pt())

    def _fill_recluster_assign(self, categories, subjet_a, subjet_b, prefix, *, jet_pt_bin, event_weight=1.0):
        """
        Fill the reclustering counterparts of the ghost assignment histograms, for
        an ensemble-level comparison of the two methods:
          {prefix}_perp_assign_recl      assignment category of perp-cone particles vs pT_g
          {prefix}_perp_assign_rg_recl   assignment category of perp-cone particles vs R_g
        """
        rg = delta_R(subjet_a, subjet_b)
        for category in categories:
            self.hists[f'{prefix}_perp_assign_recl'].Fill(jet_pt_bin, category, event_weight)
            self.hists[f'{prefix}_perp_assign_rg_recl'].Fill(rg, category, event_weight)

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
