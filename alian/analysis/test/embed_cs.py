#!/usr/bin/env python3
"""
Embeds pp detector-level jet tracks in an OO MB event, like embed_perpcone.py, but
removes the background with constituent subtraction (CS) instead of rho*A. The
combined event is subtracted with the ConstituentSubtractor (constituent_subtractor
block, rho from the bkg_estimator block) and clustered with explicit ghosts, so the
combined jets are already subtracted: j.pt() is the subtracted pT and get_subjets is
called with rho = 0. The oo jet_finder pT_min / eta_max act on the subtracted jets.

CS supplies the jet pT, the jet selection, SD (subjets A and B) and the pp matching.
The EECs use the raw (pre-CS) particles of the CS jet, mapped to its constituents by
user_index (see _raw_jets): their unmodified pT enters the pair weights and the
pt_min_eec cut, and the denominators are the subtracted pT (pT,g for subjets). Raw
particles groomed away are in the jet but in neither subjet.

Jet histograms ("sub_" = subtracted):
  sub_pT, sub_area_sub_pT             all CS jets, before SD
  sub_pTg_matched                     subtracted pT,g
  sub_pT_pp_pT_matched, sub_pTg_pp_pTg_matched   subtracted vs pp pT (pT,g)
  residual_sub_pT[g]_matched          subtracted - pp pT vs subtracted pT (pT,g)
  residual_pp_pTg_matched             subtracted - pp pT,g vs pp pT,g
  sub_area_sub_pT_matched             area of the subtracted jet vs subtracted pT
  sub_area_{a,b}_sub_pTg_matched      area of CS subjet A / B vs subtracted pT,g
  sub_n[g]_pp_n[g]_matched            CS vs pp constituent count, without the particles
                                      fully removed by CS (kept with pT = 1e-50)
"_matched" histograms are filled for combined jets that pass SD with
pT,g >= pt_min_combined_jet, are uniquely matched to a pp det jet, and whose pp jet
also passes SD. The EEC histograms below use the same selection.

Perp-cone background for the subjet EECs (combined_matched_eec_aa/_ab/_bb) and the
inclusive EEC (combined_matched_sdjet_eec): two cones of radius perpcone_R at
phi +/- pi/2 of the ungroomed jet axis, from the raw combined event, rotated under
the jet. Their particles are assigned to A, B or neither (groomed away) by
reclustering each cone with the CS jet (see _assign_to_subjets).

Filled per subjet pair type xy in (aa, ab, bb), with P1 (P2) the perp cone at
phi + pi/2 (phi - pi/2):
  combined_matched_eec_xy_jetperp     jet-P1 cross pairs
  combined_matched_eec_xy_perpperp    P1-P1 pairs
  combined_matched_eec_xy_perp1perp2  P1-P2 cross pairs
The jet-jet term is combined_matched_eec_xy itself. Pair orientations follow
combined_matched_eec_xy (both orders for aa/bb, one A-B orientation for ab), so
  signal_xy ~ eec_xy - (jetperp_xy - perp1perp2_xy) - perpperp_xy

For closure, combined_matched_eec_xy is also split by the truth origin of the
two raw tracks (pp: user_index >= 0, OO: user_index < 0):
  combined_matched_eec_xy_sigsig      pp-pp pairs   (compare to signal_xy)
  combined_matched_eec_xy_sigbkg      pp-OO pairs   (compare to jetperp_xy - perp1perp2_xy)
  combined_matched_eec_xy_bkgbkg      OO-OO pairs   (compare to perpperp_xy)
with sigsig + sigbkg + bkgbkg = combined_matched_eec_xy.

Full-pT variant of the subjet perp terms (xy in aa, ab, bb):
  combined_matched_eec_xy_{jetperp,perpperp,perp1perp2}_fullpt
Same pairs as above, but each perp cone is reclustered with the CS jet using the raw
pT of its particles (_assign_to_subjets with scale = 1), so the perp particles can
change the SD splitting of the reclustered jet, and they are assigned to that jet's
A / B. The jet-side tracks are still those of the measured A / B. Compare to the same
truth terms (sigbkg, bkgbkg). No sdjet version (no A/B assignment there).

Assignment diagnostics (see _fill_assign_diagnostics):
  combined_matched_perp_assign[_rg]   perp particle category (A, B, neither) vs pT,g [R_g]

Subjet prong matching of pp det subjets to CS subjets, same "_matched" selection
(definition in _fill_prong_matching, categories in PRONG_CATEGORY):
  combined_matched_prong_match        category vs pp pT,g and CS pT,g (subjet EEC binning)
Leading purity = categories {0, 1, 3} / all; subleading purity = {0, 2, 4} / all.
Each matched jet fills it once, so the category axis sums to sub_pTg_matched.

Event pairing: |vz_OO - vz_pp| <= analysis.vz_max_diff (see _vz_matched_pairs), else 1:1.
  delta_vz                            vz_OO - vz_pp of every analysed pair (unweighted);
                                      empty outside +/- vz_max_diff

Example usage (-n caps the number of event pairs):
python analysis/test/embed_cs.py -i1 /home/youqi/temp/combined1.root -i2 /home/youqi/temp/BerkeleyTree.root -c config/embed_cs.yaml -o output/embed_test.root -n 100
"""

import argparse
import itertools
import math
from collections import Counter, defaultdict, deque
from types import SimpleNamespace
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
from alian.analysis.base.csubtractor import CEventSubtractor
from alian.analysis.base.output import Output
from alian.analysis.base.selector import TrackSelector
from alian.analysis.base.event import Event, get_selected_tracks, get_selected_particles
from alian.analysis.base.utils import read_yaml, is_slurm
from alian.io.data_io import DataInput

# perp-cone assignment categories (values filled on the "assign" axis)
ASSIGN_A, ASSIGN_B, ASSIGN_NONE = 0, 1, 2
# perp-cone particles are reclustered as scaled copies with negligible pT, tagged by
# user_index = offset - its index
RECLUSTER_SCALE = 1e-50
PERP_INDEX_OFFSET = -10_000_000
# particles fully removed by CS are kept with pT = 1e-50 (constituent_subtractor
# remove_zero_pt_particles: false); constituents below this pT (these and explicit ghosts,
# pT ~ 1e-100) are not counted as real (real_trk_selector)
CS_ZERO_PT = 1e-40
# vertex matching (_vz_matched_pairs): waiting OO events are stored in vz slices of this width (cm)
VZ_SLICE_WIDTH = 0.1
# prong matching (_fill_prong_matching): status of a pp subjet, i.e. which CS subjet holds
# more than prong_match_threshold of its pT
PRONG_SAME, PRONG_OTHER, PRONG_NONE = 0, 1, 2
# category (values filled on the "prong_match" axis), keyed by (status of A, status of B);
# for B, PRONG_OTHER means "in the CS leading subjet"
PRONG_CATEGORY = {
    (PRONG_SAME, PRONG_SAME): 0,   # pure match
    (PRONG_SAME, PRONG_NONE): 1,   # leading matched, subleading unmatched
    (PRONG_NONE, PRONG_SAME): 2,   # leading unmatched, subleading matched
    (PRONG_SAME, PRONG_OTHER): 3,  # leading matched, subleading in leading
    (PRONG_OTHER, PRONG_SAME): 4,  # leading in subleading, subleading matched
    (PRONG_NONE, PRONG_OTHER): 5,  # leading unmatched, subleading in leading
    (PRONG_OTHER, PRONG_NONE): 6,  # leading in subleading, subleading unmatched
    (PRONG_OTHER, PRONG_OTHER): 7, # swap
    (PRONG_NONE, PRONG_NONE): 8,   # both unmatched
}

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
        self.vz_max_diff = getattr(self, 'vz_max_diff', None) or 0
        self.vz_buffer_max = getattr(self, 'vz_buffer_max', 10000)
        self.vz_counts = Counter()

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
        self.logger.info(f"Background estimator configuration details:\n\n{self.bkg_estimator.bge.description()}\n")

        # --- constituent subtractor ---
        self.constituent_subtractor = CEventSubtractor(self.cfg.get('constituent_subtractor') or {}, self.logger, bkg_estimator=self.bkg_estimator)
        self.logger.info(f"Constituent subtractor configuration:\n{self.constituent_subtractor.subtractor.description()}")
        if self.constituent_subtractor.remove_zero_pt_particles:
            self.logger.warning("constituent_subtractor.remove_zero_pt_particles is true: raw particles fully removed by CS "
                                "have no counterpart in the CS jet and are put in neither subjet A nor B.")

        # --- jet finders ---
        self.pp_particle_jet_finder = JetFinder.load(pp_cfg)
        self.pp_jet_finder = JetFinder.load(pp_cfg)
        self.combined_jet_finder = JetFinder.load(oo_cfg)
        self.R = self.pp_jet_finder.R
        self.area_cut = 0.56*np.pi*self.R*self.R

        # --- substructure tools ---
        self.eec_trk_selector = TrackSelector(pt_min=self.pt_min_eec)
        # drops particles fully removed by CS (pT = 1e-50) and explicit ghosts (pT ~ 1e-100)
        self.real_trk_selector = TrackSelector(pt_min=CS_ZERO_PT)
        self.lund_gen = fj.contrib.LundGenerator()
        # C/A with R large enough to merge all inputs into one jet (_assign_to_subjets)
        self.recluster_jet_def = fj.JetDefinition(fj.cambridge_algorithm, fj.JetDefinition.max_allowable_R)
        # self.sd = fj.contrib.SoftDrop(0, self.z_cut)

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
        pp_events = self._selected_events(self.pp_source, self.pp_selector, slurm_check)
        oo_events = self._selected_events(self.oo_source, self.oo_selector, slurm_check)
        if self.vz_max_diff > 0:
            self.logger.info(f"Vertex matching: |vz_OO - vz_pp| <= {self.vz_max_diff} cm (buffer {self.vz_buffer_max} OO events).")
            pairs = self._vz_matched_pairs(pp_events, oo_events)
        else:
            self.logger.info("No vertex matching.")
            pairs = zip(pp_events, oo_events)
        for pp_ev, oo_ev in pairs:
            self.hists['delta_vz'].Fill(oo_ev.data['vtx_z'] - pp_ev.data['vtx_z'])
            self.analyze_event_pair(pp_ev, oo_ev)
            n_pairs += 1
            if self.nev > 0 and n_pairs >= self.nev:
                break
        self.logger.info(f"Processed {n_pairs} event pairs.")
        if self.vz_max_diff > 0:
            self.logger.info(f"Vertex matching counts: {self.vz_counts}")
        self.note_time("Events analyzed")

    def _vz_matched_pairs(self, pp_events, oo_events):
        """
        Yield (pp event, OO event) pairs with |vz_OO - vz_pp| <= vz_max_diff, each OO event used
        once: the oldest waiting OO event in the window, else the next one in the window read from
        file. OO events read on the way wait in FIFOs per VZ_SLICE_WIDTH slice (at most
        vz_buffer_max in total; the oldest is dropped when full). pp events with no OO event in
        their window are skipped. Counts go to self.vz_counts.
        """
        dvz = self.vz_max_diff
        slices = defaultdict(deque)  # slice index -> deque of (serial, vz, event); no empty deques
        n_waiting = 0
        serial = 0
        oo_iter = iter(oo_events)
        oo_done = False
        for pp_ev in pp_events:
            vz = float(pp_ev.data['vtx_z'])

            # oldest waiting OO event in the window: the first in-window entry of each slice
            best = None
            for k in range(math.floor((vz - dvz) / VZ_SLICE_WIDTH), math.floor((vz + dvz) / VZ_SLICE_WIDTH) + 1):
                for i, (s, oo_vz, _) in enumerate(slices.get(k, ())):
                    if abs(oo_vz - vz) <= dvz:
                        if best is None or s < best[0]:
                            best = (s, k, i)
                        break
            if best is not None:
                _, k, i = best
                oo_ev = slices[k][i][2]
                del slices[k][i]
                if not slices[k]:
                    del slices[k]
                n_waiting -= 1
                self.vz_counts['pairs_waiting'] += 1
                yield pp_ev, oo_ev
                continue

            # none waiting: read OO events until one is in the window
            match = None
            while not oo_done:
                oo_ev = next(oo_iter, None)
                if oo_ev is None:
                    oo_done = True
                    break
                oo_vz = float(oo_ev.data['vtx_z'])
                if abs(oo_vz - vz) <= dvz:
                    match = oo_ev
                    break
                if n_waiting >= self.vz_buffer_max:
                    k_oldest = min(slices, key=lambda k: slices[k][0][0])
                    slices[k_oldest].popleft()
                    if not slices[k_oldest]:
                        del slices[k_oldest]
                    n_waiting -= 1
                    self.vz_counts['oo_dropped_buffer_full'] += 1
                # the reader reuses one event object, so keep a copy of the data
                data = {key: v.copy() if isinstance(v, np.ndarray) else v for key, v in oo_ev.data.items()}
                slices[math.floor(oo_vz / VZ_SLICE_WIDTH)].append((serial, oo_vz, SimpleNamespace(data=data)))
                serial += 1
                n_waiting += 1
            if match is not None:
                self.vz_counts['pairs_read'] += 1
                yield pp_ev, match
                continue
            if n_waiting == 0:
                self.vz_counts['stopped_oo_exhausted'] += 1
                return
            self.vz_counts['pp_no_oo_in_window'] += 1

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
        
        # --- combined jets: cluster CS combined event ---
        combined_jets = self.combined_jet_finder.find_jets(sub_combined, use_area=True)
        for j in combined_jets:
            self.hists['sub_pT'].Fill(j.pt(), weight)
            self.hists['sub_area_sub_pT'].Fill(j.pt(), j.area(), weight)
        self.hists['sub_njet_pp_njet'].Fill(len(pp_jets), len(combined_jets), weight)

        # Match the retained pp det jets to combined jets.
        # This starts a fresh matching pass, clearing the truth-match JetInfo.
        combined_jet_matched_indices = self.match_pp_det_to_combined(pp_jets, combined_jets)

        # --- fill cut flow histogram per combined jet ---
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
        for combined_ijet, j in enumerate(combined_jets):

            subjets = self.get_subjets(j, self.pt_min_combined_jet, rho=0.0)
            if subjets is None:
                continue
            gj, gj_pt, subjet_a, subjet_b = subjets
            # keep going if combined passes grooming

            pp_ijet = combined_jet_matched_indices[combined_ijet] # -1 if unmatched to pp
            combined_constituents = self.real_trk_selector.select_from(j.constituents()) # drop ghosts
            combined_g_constituents = self.real_trk_selector.select_from(gj.constituents())

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
                self.hists['sub_area_sub_pT_matched'].Fill(j.pt(), j.area(), weight)
                self.hists['sub_area_a_sub_pTg_matched'].Fill(gj.pt(), subjet_a.area(), weight)
                self.hists['sub_area_b_sub_pTg_matched'].Fill(gj.pt(), subjet_b.area(), weight)
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

                # Raw counterparts of the CS jet and subjets, wrapped as (j_raw, subjet_a_raw, subjet_b_raw);
                # jet pT, pT,g and Rg stay those of the CS jet and subjets
                j_raw, subjet_a_raw, subjet_b_raw = self._raw_jets(j, subjet_a, subjet_b, combined)
                self.do_eec(j_raw, "combined_matched_eec", ew_denom=j.pt(), jet_pt_bin=j.pt(), event_weight=weight)
                self.do_subjet_eec(j_raw, subjet_a_raw, subjet_b_raw, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight, rg=delta_R(subjet_a, subjet_b))
                self.do_subjet_eec_truth(j_raw, subjet_a_raw, subjet_b_raw, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
                self.do_subjet_eec_perpcone(j, subjet_a, subjet_b, j_raw, subjet_a_raw, subjet_b_raw, combined, "combined_matched", ew_denom=gj.pt(), jet_pt_bin=gj.pt(), event_weight=weight)
                self._fill_prong_matching(pp_subjet_a, pp_subjet_b, subjet_a, subjet_b, "combined_matched", pp_jet_pt_bin=pp_gj_pt, jet_pt_bin=gj.pt(), event_weight=weight)

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
        Return the pt-fraction of the tracks in jet_pp that are contained in jet_combined.

        jet_combined is a CS jet whose constituents carry subtracted pT, so tracks are
        identified by user_index and their raw (unsubtracted) pT from jet_pp is summed;
        the denominator is the raw jet_pp pT. Only tracks of jet_pp count, not those of
        other pp jets. Tracks removed entirely by CS (kept with pT = 1e-50) do not count
        as contained, so a perfect match gives 1 only if CS removed none of them.
        """
        pt_total = jet_pp.pt()
        combined_pp_indices = {c.user_index() for c in self.real_trk_selector.select_from(jet_combined.constituents()) if c.user_index() >= 0}
        pt_contained = 0.
        for track in jet_pp.constituents():
          if track.user_index() in combined_pp_indices:
            pt_contained += track.pt()
        return pt_contained/pt_total

    def _fill_prong_matching(self, pp_subjet_a, pp_subjet_b, subjet_a, subjet_b, prefix, *, pp_jet_pt_bin, jet_pt_bin, event_weight=1.0):
        """
        Subjet prong matching between the SD splitting of the pp det jet and that of the
        matched CS jet (rho = 0), seen from the pp side:
          {prefix}_prong_match category (PRONG_CATEGORY) vs pp pT,g (pp_jet_pt_bin) and CS pT,g (jet_pt_bin)
        A pp subjet is matched to a CS subjet if mc_fraction(pp subjet, CS subjet)
        exceeds prong_match_threshold. Tracks fully removed by CS (pT = 1e-50) do not count as found.
        """
        statuses = []
        for pp_subjet, same, other in ((pp_subjet_a, subjet_a, subjet_b), (pp_subjet_b, subjet_b, subjet_a)):
            if self.mc_fraction(pp_subjet, same) > self.prong_match_threshold:
                statuses.append(PRONG_SAME)
            elif self.mc_fraction(pp_subjet, other) > self.prong_match_threshold:
                statuses.append(PRONG_OTHER)
            else:
                statuses.append(PRONG_NONE)
        category = PRONG_CATEGORY[tuple(statuses)]

        self.hists[f'{prefix}_prong_match'].Fill(pp_jet_pt_bin, jet_pt_bin, category, event_weight)

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
        tracks = self.eec_trk_selector.select_from(jet.constituents())
        self._fill_eec_pairs(itertools.permutations(tracks, 2), hist_name, ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def do_eec_cross(self, subjet_a, subjet_b, hist_name, *, ew_denom, jet_pt_bin, event_weight=1.0):
        tracks_a = self.eec_trk_selector.select_from(subjet_a.constituents())
        tracks_b = self.eec_trk_selector.select_from(subjet_b.constituents())
        self._fill_eec_pairs(itertools.product(tracks_a, tracks_b), hist_name, ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def do_subjet_eec(self, jet, subjet_a, subjet_b, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0, rg=None):
        """Fill {prefix}_sdjet_eec, _eec_aa, _eec_bb, _eec_ab and _rg, binned by groomed pT.
        rg is the value filled in _rg; default delta_R(subjet_a, subjet_b). Pass it when jet and
        subjets are raw-particle jets whose axes are not the groomed subjets'."""
        self.do_eec(jet, f"{prefix}_sdjet_eec", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight) # jet passes SD, but this includes stuff removed by SD
        self.do_eec(subjet_a, f"{prefix}_eec_aa", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.do_eec(subjet_b, f"{prefix}_eec_bb", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.do_eec_cross(subjet_a, subjet_b, f"{prefix}_eec_ab", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        self.hists[f'{prefix}_rg'].Fill(jet_pt_bin, delta_R(subjet_a, subjet_b) if rg is None else rg, event_weight)

    def do_subjet_eec_truth(self, jet, subjet_a, subjet_b, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """
        Split {prefix}_sdjet_eec, _eec_aa, _eec_bb, _eec_ab by the truth origin of the pair:
        fills {prefix}_<region>_sigsig / _sigbkg / _bkgbkg (signal = user_index >= 0).
        Same pairs and weights as do_subjet_eec, so the three sum to {prefix}_<region>.
        """
        tracks = self.eec_trk_selector.select_from(jet.constituents())
        tracks_a = self.eec_trk_selector.select_from(subjet_a.constituents())
        tracks_b = self.eec_trk_selector.select_from(subjet_b.constituents())
        pair_types = ('bkgbkg', 'sigbkg', 'sigsig') # indexed by number of signal tracks in the pair
        for region, pairs in (
            ('sdjet_eec', itertools.permutations(tracks, 2)),
            ('eec_aa', itertools.permutations(tracks_a, 2)),
            ('eec_bb', itertools.permutations(tracks_b, 2)),
            ('eec_ab', itertools.product(tracks_a, tracks_b)),
        ):
            for p1, p2 in pairs:
                pair_type = pair_types[(p1.user_index() >= 0) + (p2.user_index() >= 0)] # adding 2 booleans, index will be 0, 1 or 2
                ew = p1.pt() * p2.pt() / ew_denom / ew_denom
                self.hists[f'{prefix}_{region}_{pair_type}'].Fill(jet_pt_bin, delta_R(p1, p2), ew*event_weight)

    def do_subjet_eec_perpcone(self, jet, subjet_a, subjet_b, jet_raw, subjet_a_raw, subjet_b_raw, combined, prefix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """
        Fill perp-cone EEC terms for the SD jet and for subjets A and B of jet.

        jet, subjet_a, subjet_b: the CS jet and subjets.
        jet_raw, subjet_a_raw, subjet_b_raw: the raw counterparts of the CS jet 
        and subjets (see _raw_jets), which give the jet-side tracks.
        combined: the raw combined event, from which the perp cones are taken.
        """
        perp1 = self.eec_trk_selector.select_from(self._perpcone_parts(jet, combined, +np.pi/2)) # a list of PseudoJets
        perp2 = self.eec_trk_selector.select_from(self._perpcone_parts(jet, combined, -np.pi/2))
        perp1_a, perp1_b, perp1_categories = self._assign_to_subjets(jet, perp1) # lists of PseudoJets, list of ASSIGN_*
        perp2_a, perp2_b, perp2_categories = self._assign_to_subjets(jet, perp2)
        self._fill_assign_diagnostics(perp1_categories + perp2_categories, delta_R(subjet_a, subjet_b), prefix, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

        tracks = self.eec_trk_selector.select_from(jet_raw.constituents())
        tracks_a = self.eec_trk_selector.select_from(subjet_a_raw.constituents())
        tracks_b = self.eec_trk_selector.select_from(subjet_b_raw.constituents())

        fill_args = dict(ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)
        terms = [self._perp_term('sdjet_eec', tracks, perp1, perp2)]
        terms += self._subjet_perp_terms(tracks_a, tracks_b, perp1_a, perp1_b, perp2_a, perp2_b)
        self._fill_perp_terms(terms, prefix, '', **fill_args)

        # Full-pT variant: the perp particles are reclustered with their raw pT, so they can
        # change the C/A tree and the SD splitting of the reclustered jet; they are assigned to
        # that jet's A / B. The jet-side tracks stay those of the measured subjets A / B.
        perp1_a_full, perp1_b_full, _ = self._assign_to_subjets(jet, perp1, scale=1.0)
        perp2_a_full, perp2_b_full, _ = self._assign_to_subjets(jet, perp2, scale=1.0)
        terms_full = self._subjet_perp_terms(tracks_a, tracks_b, perp1_a_full, perp1_b_full, perp2_a_full, perp2_b_full)
        self._fill_perp_terms(terms_full, prefix, '_fullpt', **fill_args)

    @staticmethod
    def _perp_term(region, jet_tracks, p1, p2):
        """(region, jet-P1 pairs, P1-P1 pairs, P1-P2 pairs) for pairs within one region."""
        return (
            region,
            itertools.chain(itertools.product(jet_tracks, p1), itertools.product(p1, jet_tracks)),
            itertools.permutations(p1, 2),
            itertools.chain(itertools.product(p1, p2), itertools.product(p2, p1)),
        )

    def _subjet_perp_terms(self, tracks_a, tracks_b, perp1_a, perp1_b, perp2_a, perp2_b):
        """Perp terms (see _perp_term) for eec_aa, eec_bb and eec_ab."""
        return [
            self._perp_term('eec_aa', tracks_a, perp1_a, perp2_a),
            self._perp_term('eec_bb', tracks_b, perp1_b, perp2_b),
            (
                'eec_ab',
                itertools.chain(itertools.product(tracks_a, perp1_b), itertools.product(perp1_a, tracks_b)),
                itertools.product(perp1_a, perp1_b),
                itertools.chain(itertools.product(perp1_a, perp2_b), itertools.product(perp2_a, perp1_b)),
            ),
        ]

    def _fill_perp_terms(self, terms, prefix, suffix, *, ew_denom, jet_pt_bin, event_weight=1.0):
        """Fill {prefix}_{region}_{jetperp,perpperp,perp1perp2}{suffix} from (region, ...) terms."""
        for region, jetperp, perpperp, perp1perp2 in terms:
            for name, pairs in (('jetperp', jetperp), ('perpperp', perpperp), ('perp1perp2', perp1perp2)):
                self._fill_eec_pairs(pairs, f"{prefix}_{region}_{name}{suffix}", ew_denom=ew_denom, jet_pt_bin=jet_pt_bin, event_weight=event_weight)

    def _raw_jets(self, jet, subjet_a, subjet_b, parts):
        """
        Match subtracted jet constituents with the pre-subtracted ones through user_index
        This includes particles fully removed by CS, which are kept with pT = 1e-50
        (remove_zero_pt_particles: false).

        Returns (jet_raw, subjet_a_raw, subjet_b_raw):
          PseudoJets joined from the raw particles of the CS jet, subjet A and subjet B.
        """
        raw = []
        for cs_jet in (jet, subjet_a, subjet_b):
            indices = {c.user_index() for c in cs_jet.constituents() if not c.is_pure_ghost()} # {}: set
            raw.append(fj.join(std.vector[fj.PseudoJet]([p for p in parts if p.user_index() in indices])))
        return tuple(raw)

    def _fill_eec_pairs(self, pairs, hist_name, *, ew_denom, jet_pt_bin, event_weight=1.0):
        for p1, p2 in pairs:
            ew = p1.pt() * p2.pt() / ew_denom / ew_denom
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
            p.reset_PtYPhiM(part.pt(), part.rapidity(), part.phi() - dphi, part.m())
            p.set_user_index(part.user_index())
            rotated.append(p)
        return rotated

    def _assign_to_subjets(self, jet, perp, scale=RECLUSTER_SCALE):
        """
        A/B assignment of perp-cone particles by reclustering.

        Copies of the perp-cone particles scaled by scale are reclustered with
        C/A together with the non-ghost constituents of jet, and SD is redone with
        select_soft_drop (rho = 0). With the default RECLUSTER_SCALE each copy lands
        where a soft particle there would, and the tree and SD splitting of the real
        particles are unchanged. With scale = 1 the copies carry their raw pT and can
        change both; they are then assigned to the A / B of the reclustered jet, which
        can differ from the A / B of jet. If no splitting of the reclustered jet passes
        SD, every particle is ASSIGN_NONE.
        Jet membership is not checked: a copy just outside the anti-kT boundary can merge
        early with an edge particle and be assigned to A or B.

        Returns (perp_a, perp_b, categories):
          perp_a, perp_b  lists of PseudoJets, the perp-cone particles assigned to A / B
          categories      list of size len(perp), one entry per perp cone particle:
                          ASSIGN_A or ASSIGN_B if its copy is in subjet A / B, ASSIGN_NONE
                          if it is groomed away
        """
        if len(perp) == 0:
            return [], [], []
        inputs = [c for c in jet.constituents() if not c.is_pure_ghost()]
        for i, p in enumerate(perp):
            copy = fj.PseudoJet(p.px() * scale, p.py() * scale, p.pz() * scale, p.e() * scale)
            copy.set_user_index(PERP_INDEX_OFFSET - i)
            inputs.append(copy)
        # cs must stay alive while its jets are used
        cs = fj.ClusterSequence(std.vector[fj.PseudoJet](inputs), self.recluster_jet_def)
        test_jet = fj.sorted_by_pt(cs.inclusive_jets())[0] # the only jet, with R = max_allowable_R

        # Redo SD on test_jet, as get_subjets() did on the CS jet (rho = 0: CS jets are already
        # subtracted). With the default scale the copies have pT -> 0, so the C/A tree of the
        # real particles and the first splitting passing z_cut are unchanged.
        categories = [ASSIGN_NONE] * len(perp)
        sd = self.select_soft_drop(self.lund_gen.result(test_jet), z_cut=self.z_cut)
        if sd is None:
            return [], [], categories
        l, swapped = sd
        test_subjet_a, test_subjet_b = (l.softer(), l.harder()) if swapped else (l.harder(), l.softer())

        for category, branch in ((ASSIGN_A, test_subjet_a), (ASSIGN_B, test_subjet_b)):
            for c in branch.constituents():
                index = c.user_index()
                if index <= PERP_INDEX_OFFSET:
                    categories[PERP_INDEX_OFFSET - index] = category

        perp_a = [p for p, category in zip(perp, categories) if category == ASSIGN_A]
        perp_b = [p for p, category in zip(perp, categories) if category == ASSIGN_B]
        return perp_a, perp_b, categories

    def _fill_assign_diagnostics(self, categories, rg, prefix, *, jet_pt_bin, event_weight=1.0):
        """
        Diagnostics for the reclustering A/B assignment of the perp-cone particles:
          {prefix}_perp_assign               assignment category of perp-cone particles vs pT_g
          {prefix}_perp_assign_rg            assignment category of perp-cone particles vs R_g
        """
        for category in categories:
            self.hists[f'{prefix}_perp_assign'].Fill(jet_pt_bin, category, event_weight)
            self.hists[f'{prefix}_perp_assign_rg'].Fill(rg, category, event_weight)

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
