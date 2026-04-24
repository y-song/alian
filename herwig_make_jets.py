
#!/usr/bin/env python
'''
Script for looking at the quark vs gluon dependence of substructure observables
Author: Beatrice Liang-Gilman, with most of the code from Ezra Lesser (elesser@berkeley.edu)
'''

from __future__ import print_function

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
# import fjext
# import ecorrel
import sys

import ROOT

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np
from array import array
import math
import uproot
import awkward as ak
import pandas as pd

# from pyjetty.mputils import *
from pyjetty.mputils.mputils import pinfo, pwarning

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

from pyjetty.alice_analysis.process.base import process_base

from enum import Enum
import fjtools

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()



################################################################
class HerwigMakeJets(process_base.ProcessBase):

	#---------------------------------------------------------------
	# Constructor
	#---------------------------------------------------------------
	def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

		super(HerwigMakeJets, self).__init__(
			input_file, config_file, output_dir, debug_level, **kwargs)

		# Call base class initialization
		process_base.ProcessBase.initialize_config(self)

		# Read config file
		with open(self.config_file, 'r') as stream:
			config = yaml.safe_load(stream)

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		self.jetR_list = config["jetR"]

		# self.nev = args.nev


		# self defined variables
		self.ev_num_start = args.ev_num_base


		# hadron level - ALICE tracking restriction
		self.max_eta_hadron = 0.9

		self.min_leading_track_pT = config["min_leading_track_pT"] if "min_leading_track_pT" in config else None


	#---------------------------------------------------------------
	# Main processing function
	#---------------------------------------------------------------
	def herwig_make_jets(self, args):

		# Create ROOT TTree file for storing raw PYTHIA particle information
		outf_path = os.path.join(self.output_dir, args.tree_output_fname)
		self.fout = ROOT.TFile(outf_path, 'recreate') #outf = ROOT.TFile(outf_path, 'recreate')
		self.fout.cd() #outf.cd()

		# Initialize response histograms
		self.initialize_hist()

		# print the banner first
		fj.ClusterSequence.print_banner()
		print()

		# print("----------------- OPEN FILE HERE -----------------")
		self.fin = uproot.open(args.input_file) 

		particles = self.fin["PWGHF_TreeCreator/tree_Particle_jse"]

		# Load full arrays
		df_pd = particles.arrays(["run_number", "ev_id", "ParticlePx", "ParticlePy", "ParticlePz", "ParticleE", "ParticlePID"], library="pd")

		# print("----------------- CLOSE FILE HERE -----------------")

		self.init_jet_tools()
		self.calculate_events(df_pd)
		# pythia.stat()
		print()

		self.scale_print_final_info()

		self.fout.Write() # outf.Write()
		# outf.Close()

		self.save_output_objects() # file gets closed in this function

		self.fin.close()

	#---------------------------------------------------------------
	# Initialize histograms
	#---------------------------------------------------------------
	def initialize_hist(self):

		self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
		self.hjetpT = ROOT.TH1D("hjetpT", "pT of jet", 200, 0, 200)

		
		for jetR in self.jetR_list:

			# Store a list of all the histograms just so that we can rescale them later
			hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
			setattr(self, hist_list_name, [])


	#---------------------------------------------------------------
	# Initiate jet defs, selectors, and sd (if required)
	#---------------------------------------------------------------
	def init_jet_tools(self):

		for jetR in self.jetR_list:
			jetR_str = str(jetR).replace('.', '')

			# set up our jet definition and a jet selector
			jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
			setattr(self, "jet_def_R%s" % jetR_str, jet_def)

		pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
		parts_selector_h = fj.SelectorAbsEtaMax(self.max_eta_hadron)
		track_selector_ch = fj.SelectorPtMin(0.15) & parts_selector_h #ALICE parameters
		setattr(self, "track_selector_ch", track_selector_ch)

		for jetR in self.jetR_list:
			jetR_str = str(jetR).replace('.', '')

			jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
			#jet_selector = fj.SelectorPtMin(0.) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
			setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

			count1 = 0  # Number of partonic parents which match to >1 ch-jets
			setattr(self, "count1_R%s" % jetR_str, count1)
			count2 = 0  # Number of partonic parents which match to zero ch-jets
			setattr(self, "count2_R%s" % jetR_str, count2)

	#---------------------------------------------------------------
	# Calculate events and pass information on to jet finding
	#---------------------------------------------------------------
	def calculate_events(self, df_pd):

		# iev = 0  # Event loop count
		self.ijet = 0 # Jet number count

		all_constituents = []

		# while iev < self.nev:
		for (run, iev), group in df_pd.groupby(["run_number", "ev_id"]):

			if (iev%5000 == 0): #10000
				print("Event", iev)
			# print("Event", iev)


			# Build all PseudoJets for this event at once
			pj_particles = [ fj.PseudoJet(row.ParticlePx, row.ParticlePy, row.ParticlePz, row.ParticleE) for row in group.itertuples() ]
                
			# pj_particles = [  fj.PseudoJet( row.ParticlePt * np.cos(row.ParticlePhi),   # px
			# 							  row.ParticlePt * np.sin(row.ParticlePhi),   # py
			# 							  row.ParticlePt * np.sinh(row.ParticleEta),  # pz
			# 							  row.ParticlePt * np.cosh(row.ParticleEta) )  # E (massless)
			# 				  for row in group.itertuples() ]

			# # charged-hadron level
			# parts_pythia_hch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
			
				
			# Some "accepted" events don't survive hadronization step -- keep track here
			self.hNevents.Fill(0)
			self.find_jets_fill_histograms(pj_particles, iev, all_constituents)

			# iev += 1

		# Convert to DataFrame
		print(len(all_constituents))
		df = pd.DataFrame(all_constituents)

		# Save to Parquet (FAST and SMALL)
		out_parq_file = os.path.join(self.output_dir, "JetsForAnalysis.parquet")
		df.to_parquet(out_parq_file, compression="snappy")


	#---------------------------------------------------------------
	# Find jets, do matching between levels, and fill histograms
	#---------------------------------------------------------------
	def find_jets_fill_histograms(self, pj_particles, iev, all_constituents):

		# Loop over jet radii
		for jetR in self.jetR_list:

			jetR_str = str(jetR).replace('.', '')
			jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
			jet_def = getattr(self, "jet_def_R%s" % jetR_str)
			track_selector_ch = getattr(self, "track_selector_ch")

			# Get the jets at different levels
			jets_ch = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(pj_particles)))) # charged hadron level



			# Fill histograms
			for i_jch, jet in enumerate(jets_ch):

				self.hjetpT.Fill(jet.pt())
				# print("Filling jet pt", jet.pt())
				# print("iev", iev, "ijet", self.ijet)


				# Now save jet information by looping through constituents
				constituents = fj.sorted_by_pt(jet.constituents())
				for c in constituents:
					all_constituents.append({
						"event_id": iev + self.ev_num_start,
						"jet_id": self.ijet,
						"jet_pt": jet.pt(),
						"parton_pid": 0,
						"c_px": c.px(), # "c_pt": c.pt(),
						"c_py": c.py(), # "c_eta": c.eta(),
						"c_pz": c.pz(), # "c_phi": c.phi() })
						"c_e": c.e() })
					# print("  added const", c.px(), c.py(), c.pz(), c.e())

				self.ijet += 1
					
					

	#---------------------------------------------------------------
	# Initiate scaling of all histograms and print final simulation info
	#---------------------------------------------------------------
	def scale_print_final_info(self):
		self.hNevents.SetBinError(1, 0)
		self.hjetpT.SetBinError(1, 0)



################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='herwig make jets',
									 prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('-i', '--input-file', action='store', type=str, default='./',
						help='Input root file with JSE-style tree of particles')
	parser.add_argument('-o', '--output-dir', action='store', type=str, default='./',
						help='Output directory for generated ROOT file(s)')
	parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
						help="Filename for the (unscaled) generated particle ROOT TTree")
	parser.add_argument('-cf', '--config_file', action='store', type=str, default='config/angularity.yaml',
						help="Path of config file for observable configurations")
	parser.add_argument('--ev-num-base', action='store', type=int, default=0, help="for merged files, this counts the total number of events")
	
	

	args = parser.parse_args()
	pinfo("The arguments to run are: ", args)

	# If invalid configFile is given, exit
	if not os.path.exists(args.config_file):
		print('File \"{0}\" does not exist! Exiting!'.format(args.config_file))
		sys.exit(0)

	# # Have at least 1 event
	# if args.nev < 1:
	# 	args.nev = 1


	process = HerwigMakeJets(config_file=args.config_file, output_dir=args.output_dir, args=args)
	process.herwig_make_jets(args)
