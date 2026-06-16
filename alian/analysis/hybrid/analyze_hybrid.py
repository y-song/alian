#!/usr/bin/env python3


import argparse
import itertools
import numpy as np
from alian.analysis.base import delta_R

import tqdm
import os
import sys
import array
import yasp
import cppyy
import ROOT

import heppyy.util.fastjet_cppyy
import heppyy.util.heppyy_cppyy

from cppyy.gbl import fastjet as fj
from cppyy.gbl.std import vector
from cppyy.gbl import heppyy
from heppyy.util.logger import Logger

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    arr = array.array('f', lspace)
    return arr

def logbins(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    arr = array.array('f', lspace)
    return arr

class AnalyzeHybrid:
    def __init__(self, args, log):
        self.args = args
        self.log = log

        fj.ClusterSequence.print_banner()
        print()

        ner = heppyy.NegativeEnergyRecombiner(1001)
        log.critical(ner.description())

        jet_R0 = args.jet_R
        self.jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
        if args.wake:
            self.jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0, ner)
        jet_def_lund = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
        self.lund_gen = fj.contrib.LundGenerator(jet_def_lund)
        log.critical(self.jet_def.description())
        log.critical(self.lund_gen.description())

        self.jet_selector = fj.SelectorPtMin(args.jet_min_pt) * fj.SelectorAbsEtaMax(args.part_max_eta - jet_R0)
        self.eec_trk_selector = fj.SelectorPtMin(args.eec_min_pt)

        self.input = heppyy.HybridFile(args.input)

        # outf_path = os.path.join(output_dir, args.tree_output_fname)
        self.output = ROOT.TFile("~/heppyy/heppyy/test.root", 'recreate')
        self.output.cd()

        self.hists = {}
        self.hists['jet_pT'] = ROOT.TH1D("jet_pT", "Jet #it{p}_{T} spectrum;jet #it{p}_{T} (GeV);Counts", 200, linbins(0,200,200))
        self.hists['jet_eta'] = ROOT.TH1D("jet_eta", "Jet #it{#eta};jet #it{#eta};Counts", 40, linbins(-1,1,40))
        self.hists['sdjet_pT_jet_pT'] = ROOT.TH2D("sdjet_pT_jet_pT", "SD jet #it{p}_{T} vs jet #it{p}_{T};jet #it{p}_{T} (GeV);SD jet #it{p}_{T} (GeV)", 200, linbins(0,200,200), 200, linbins(0,200,200))
        self.hists['rg'] = ROOT.TH2D("rg", "#it{R}_{g};jet #it{p}_{T} (GeV);#it{R}_{g}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))
        self.hists['zg'] = ROOT.TH2D("zg", "#it{z}_{g};jet #it{p}_{T} (GeV);#it{z}_{g}", 200, linbins(0,200,200), 30, linbins(-1,0.5,30))
        self.hists['eec'] = ROOT.TH2D("eec", "EEC;jet #it{p}_{T} (GeV);#it{R}_{L}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))
        self.hists['sdjet_eec'] = ROOT.TH2D("sdjet_eec", "EEC;jet #it{p}_{T} (GeV);#it{R}_{L}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))
        self.hists['eec_aa'] = ROOT.TH2D("eec_aa", "EEC;jet #it{p}_{T} (GeV);#it{R}_{L}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))
        self.hists['eec_bb'] = ROOT.TH2D("eec_bb", "EEC;jet #it{p}_{T} (GeV);#it{R}_{L}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))
        self.hists['eec_ab'] = ROOT.TH2D("eec_ab", "EEC;jet #it{p}_{T} (GeV);#it{R}_{L}", 200, linbins(0,200,200), 50, logbins(1E-4,1,50))

    def analyze_events(self, args, log):
        for ievent in tqdm.tqdm(range(args.nev)):
            if not self.input.nextEvent():
                break
            parts = self.input.getParticles(include_wake=args.wake, charged_only=False)
            partons = self.input.getPartons()
            sparts = self.input.getParticlesStr()
            spartons = self.input.getPartonsStr()
            ev_info = self.input.info()

            ana.analyze_event(ievent, parts, partons, sparts, spartons, ev_info)

    def analyze_event(self, ievent, parts, partons, sparts, spartons, ev_info):
        args = self.args
        log = self.log

        log.info(f'* event {ievent} has {len(partons)} partons')
        log.info(f'  ev_info.weight: {ev_info.weight()}, ev_info.cross: {ev_info.cross()}, ev_info.x: {ev_info.x()}, ev_info.y: {ev_info.y()}')
        for np in range(len(partons)):
            log.info(f'  parton pt: {partons[np].pt()} eta: {partons[np].eta()}')
            log.debug(f'- from psj: px={partons[np].px()}, py={partons[np].py()}, pz={partons[np].pz()}, E={partons[np].E()}, m={partons[np].m()}, ui={partons[np].user_index()}')
            log.debug(f'- from str: {spartons[np]}')
        log.debug(f'event {ievent} has {len(parts)} particles')
        for np in range(len(parts)):
            log.debug(f'- from psj: px={parts[np].px()}, py={parts[np].py()}, pz={parts[np].pz()}, E={parts[np].E()}, m={parts[np].m()}, ui={parts[np].user_index()}')
            log.debug(f'- from str: {sparts[np]}')

        parts_selected = vector[fj.PseudoJet](
            [
                part for part in parts
                if part.eta() < args.part_max_eta and part.pt() > args.part_min_pt
            ]
        )

        jets = fj.sorted_by_pt(self.jet_selector(self.jet_def(parts_selected)))
        if len(jets) == 0:
            return

        log.info(f'-> event {ievent} has {len(jets)} jets')
        for j in jets:
            log.info(f'   - jet pt: {j.pt()} eta: {j.eta()}')
        log.disable_console()
        log.info('debugging... - this should not write to console but show up in the log file')
        log.enable_console()

        for j in jets:
            log.debug(f'jet pt: {j.pt()} eta: {j.eta()}')
            self.hists['jet_pT'].Fill(j.pt())
            self.hists['jet_eta'].Fill(j.eta())

            # match partons to jets
            for np in range(len(partons)):
                p = partons[np]
                if j.delta_R(p) < 0.4:
                    log.debug(f' - matched parton {np} with pt: {p.perp()} eta: {p.eta()}')

            if j.pt() < args.jet_min_pt_ana:
                break
            self.do_eec(j, "eec")
            lund_seq = self.lund_gen.result(j)
            l = self.select_soft_drop(lund_seq, z_cut=args.z_cut) # class is LundDeclustering
            if l is None:
                self.hists['rg'].Fill(j.pt(), -0.99)
                self.hists['zg'].Fill(j.pt(), -0.99)
                self.hists['sdjet_pT_jet_pT'].Fill(j.pt(), 0)
                continue
            subjet_a = l.harder()
            subjet_b = l.softer()
            sd_pt = subjet_a.pt() + subjet_b.pt() # approximate
            self.hists['sdjet_pT_jet_pT'].Fill(j.pt(), sd_pt)
            self.hists['rg'].Fill(j.pt(), delta_R(subjet_a, subjet_b))
            self.hists['zg'].Fill(j.pt(), subjet_b.pt() / sd_pt)
            self.do_eec(j, "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=j.pt()) # jet passes SD, but this includes stuff removed by SD
            # self.do_eec(sd(j), "sdjet_eec", ew_denom=sd_pt, jet_pt_bin=j.pt()) # jet passes SD, and this only includes stuff passes by SD
            self.do_eec(subjet_a, "eec_aa", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec(subjet_b, "eec_bb", ew_denom=sd_pt, jet_pt_bin=j.pt())
            self.do_eec_cross(subjet_a, subjet_b, "eec_ab", ew_denom=sd_pt, jet_pt_bin=j.pt())

    def finalize(self):
        self.output.Write()
        self.output.Close()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze hybrid with fastjet on the fly', prog=os.path.basename(__file__))
    parser.add_argument('-i', '--input', help='input file', default='', required=True)
    parser.add_argument('-n', '--nev', help='number of events', default=10, type=int)
    parser.add_argument('-w', '--wake', help='include wake particles', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
    parser.add_argument('--part-max-eta', help="max eta of a particle to accept", default=0.9, type=float)
    parser.add_argument('--part-min-pt', help="min pT of a particle to accept", default=0.15, type=float)
    parser.add_argument('--jet-min-pt', help="minimum pT jet to accept", default=5., type=float)
    parser.add_argument('--jet-min-pt-ana', help="minimum pT jet to accept for substructure ana", default=5., type=float)
    parser.add_argument('--jet-R', help="jet R", default=0.4, type=float)
    parser.add_argument('--eec-min-pt', help="minimum pT for EEC constituents to accept", default=1.0, type=float)
    parser.add_argument('--z-cut', default=0.1, type=float)
    parser.add_argument('-g', '--debug', help="write debug things", default=False, action='store_true')
    args = parser.parse_args()

    # set up logging - this uses singleton Logger
    log_level = 'DEBUG' if args.debug else 'WARNING'
    log = Logger(console=args.verbose, level=log_level)
    if args.verbose:
        log.set_level('INFO')
    if args.debug:
        log.set_level('DEBUG')
    log.critical(args)

    ana = AnalyzeHybrid(args, log)
    ana.analyze_events(args, log)
    ana.finalize()
