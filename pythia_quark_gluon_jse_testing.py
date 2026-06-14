
#!/usr/bin/env python
'''
Script for looking at the quark vs gluon dependence of substructure observables
Author: Beatrice Liang-Gilman, with most of the code from Ezra Lesser (elesser@berkeley.edu)
'''

from __future__ import print_function

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext
import ecorrel
import othercorrel
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

from pyjetty.mputils import *
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
class PythiaQuarkGluon(process_base.ProcessBase):

	#---------------------------------------------------------------
	# Constructor
	#---------------------------------------------------------------
	def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

		super(PythiaQuarkGluon, self).__init__(
			input_file, config_file, output_dir, debug_level, **kwargs)

		# Call base class initialization
		process_base.ProcessBase.initialize_config(self)

		# Read config file
		with open(self.config_file, 'r') as stream:
			config = yaml.safe_load(stream)

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		self.jetR_list = config["jetR"]

		self.user_seed = args.user_seed
		self.nev = args.nev

		self.noMPI = (bool)(1-args.MPIon)
		self.noISR = (bool)(1-args.ISRon)

		# self defined variables
		self.weighted = (bool)(args.weightON) #weightON=True(F) means turn weights on(off)
		self.ev_num_start = args.ev_num_base

		# PDG ID values for quarks and gluons
		self.quark_pdg_ids = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
		self.down_pdg_ids = [1, -1]
		self.up_pdg_ids = [2, -2]
		self.strange_pdg_ids = [3, -3]
		self.gluon_pdg_ids = [9, 21] 
		


		# hadron level - ALICE tracking restriction
		self.max_eta_hadron = 0.9

		self.min_leading_track_pT = config["min_leading_track_pT"] if "min_leading_track_pT" in config else None


		self.obs_bins_EEC = np.logspace(np.log10(1E-4), np.log10(1), 51)

		# self.observable_list = config['process_observables']
		# self.obs_settings = {}
		# self.obs_grooming_settings = {}
		# self.obs_names = {}
		# for observable in self.observable_list:

		# 	pinfo("OBSERVABLE", observable)

		# 	obs_config_dict = config[observable]
		# 	obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]

		# 	self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_config_list)
		# 	pinfo("self.obs_settings[observable]", self.obs_settings[observable])
		# 	self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)
		# 	pinfo("self.obs_grooming_settings[observable]", self.obs_grooming_settings[observable])

		# 	self.obs_names[observable] = obs_config_dict["common_settings"]["xtitle"]

	#---------------------------------------------------------------
	# Main processing function
	#---------------------------------------------------------------
	def pythia_quark_gluon(self, args):

		# Create ROOT TTree file for storing raw PYTHIA particle information
		outf_path = os.path.join(self.output_dir, args.tree_output_fname)
		self.fout = ROOT.TFile(outf_path, 'recreate') #outf = ROOT.TFile(outf_path, 'recreate')
		self.fout.cd() #outf.cd()

		# Initialize response histograms
		self.initialize_hist()

		pinfo('user seed for pythia', self.user_seed) #TODO: what does this do?? it doesn't work...
#        print('user seed for pythia', self.user_seed)
		mycfg = ['Random:setSeed=on', 'Random:seed={}'.format(self.user_seed)]
		mycfg.append('HadronLevel:all=off')


		# print the banner first
		fj.ClusterSequence.print_banner()
		print()

		# -------------------------------
		# Setting MPIs and ISRs
		print('Will run no MPI:',self.noMPI)
		print('Will run no ISR:',self.noISR)
		setattr(args, "py_noMPI", self.noMPI)
		setattr(args, "py_noISR", self.noISR)
		# -------------------------------

		pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
		# print("----------------- PARTICLE DATA INFO HERE -----------------")
		# pythia.particleData.listAll()
		# print("----------------- PARTICLE DATA INFO END -----------------")

		self.init_jet_tools()
		self.calculate_events(pythia)
		pythia.stat()
		print()

		self.scale_print_final_info(pythia)

		self.fout.Write() # outf.Write()
		# outf.Close()

		self.save_output_objects() # file gets closed in this function

	#---------------------------------------------------------------
	# Initialize histograms
	#---------------------------------------------------------------
	def initialize_hist(self):

		self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
		self.hjetpT_chjet = ROOT.TH1D("hjetpT_chjet", "pT of charged jet, with ALICE selections", 200, 0, 200)
		self.hjetpT_h = ROOT.TH1D("hjetpT_h", "pT of full jet", 200, 0, 200)
		self.hjetpT_ha = ROOT.TH1D("hjetpT_ha", "pT of full jet, with ALICE selections", 200, 0, 200)

		self.hjetpT_chjet_matchedtoparton = ROOT.TH1D("hjetpT_chjet_matchedtoparton", "pT of charged jet, with ALICE selections, incl (jet matched to init. parton)", 200, 0, 200)
		self.hjetpT_h_matchedtoparton = ROOT.TH1D("hjetpT_h_matchedtoparton", "pT of full jet, incl (jet matched to init. parton)", 200, 0, 200)
		
		self.hDeltaR = ROOT.TH1F("hDeltaR", 'Delta R between jet and each parent', 40, 0, 0.4)
		
		# for jetR in self.jetR_list:

		# 	# Store a list of all the histograms just so that we can rescale them later
		# 	hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
		# 	setattr(self, hist_list_name, [])


		# 	for observable in self.observable_list:
		# 		# Should only be one: observable == "EEC"
		# 		if observable != "corr_deltajt":
		# 			raise ValueError("Observable %s is not implemented in this script" % observable)

		# 		obs_bins = getattr(self, "obs_bins_" + observable)
		# 		# Use more finely binned pT bins for TH2s than for the RMs


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
	def calculate_events(self, pythia):

		iev = 0  # Event loop count
		self.ijet = 0 # Jet number count

		while iev < self.nev:
			if not pythia.next():
				continue

			if (iev%5000 == 0): #10000
				print("Event", iev)
			# print("Event", iev)

			self.parents = []
			fs_parton_5 = fj.PseudoJet(pythia.event[5].px(), pythia.event[5].py(), pythia.event[5].pz(), pythia.event[5].e())
			fs_parton_6 = fj.PseudoJet(pythia.event[6].px(), pythia.event[6].py(), pythia.event[6].pz(), pythia.event[6].e())
			self.parents = [fs_parton_5, fs_parton_6] # parent partons in dijet

			# Save PDG code of the parent partons
			self.parent_ids = [pythia.event[5].id(), pythia.event[6].id()]


			# parton level
			#parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

			hstatus = pythia.forceHadronLevel()
			if not hstatus:
				continue

			# full-hadron level
			parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

			# charged-hadron level
			parts_pythia_hch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

				
			# Some "accepted" events don't survive hadronization step -- keep track here
			self.hNevents.Fill(0)
			self.find_jets_fill_histograms(parts_pythia_hch, parts_pythia_h, iev)

			iev += 1

	def assign_poss_parton_match(self, jch, jet_matching_distance, jetR):
		for i_parent, parent in enumerate(self.parents):
			parentmatch_name = "parent%imatch" % i_parent
			#plot 
			self.hDeltaR.Fill(jch.delta_R(parent))
			if jch.delta_R(parent) < jet_matching_distance * jetR:
				match = getattr(self, parentmatch_name)
				if not match:
					setattr(self, parentmatch_name, jch)
				else:  # Already found a match
					# Set flag value so that we know to ignore this one
					setattr(self, parentmatch_name, 0)
					

	#---------------------------------------------------------------
	# Find jets, do matching between levels, and fill histograms
	#---------------------------------------------------------------
	def find_jets_fill_histograms(self, parts_pythia_hch, parts_pythia_h, iev):

		# Loop over jet radii
		for jetR in self.jetR_list:

			jetR_str = str(jetR).replace('.', '')
			jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
			jet_def = getattr(self, "jet_def_R%s" % jetR_str)
			track_selector_ch = getattr(self, "track_selector_ch")

			count1 = getattr(self, "count1_R%s" % jetR_str)
			count2 = getattr(self, "count2_R%s" % jetR_str)

			# Get the jets at different levels
			#jets_p  = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p  ))) # parton level
			jets_h  = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h  ))) # full hadron level
			jets_ha  = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(parts_pythia_h )))) # full hadron level with ALICE tracking selections
			jets_ch  = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(parts_pythia_hch)))) # charged hadron level  with ALICE tracking selections

			# anothacounter=0

			# Find the charged jet closest to the axis of the original parton
			# Require that the match is within some small angle, and that it is unique
			jet_matching_distance = 0.6  # Match jets with deltaR < jet_matching_distance*jetR
			self.parent0match, self.parent1match = None, None
			
			# Fill histogram - charged jets
			for i_jch, jch in enumerate(jets_ch):
				self.hjetpT_chjet.Fill(jch.pt())
				# print("Filling jet pt", jet.pt())

				self.assign_poss_parton_match(jch, jet_matching_distance, jetR)

				# do stuff here
				# for i_parent, parent in enumerate(self.parents):
				# 	parentmatch_name = "parent%imatch" % i_parent
				# 	#plot 
				# 	self.hDeltaR.Fill(jch.delta_R(parent))
				# 	if jch.delta_R(parent) < jet_matching_distance * jetR:
				# 		match = getattr(self, parentmatch_name)
				# 		if not match:
				# 			setattr(self, parentmatch_name, jch)
				# 		else:  # Already found a match
				# 			# Set flag value so that we know to ignore this one
				# 			setattr(self, parentmatch_name, 0)

			# If we have matches, fill histograms
			for i_parent, parent in enumerate(self.parents):
				jet = getattr(self, "parent%imatch" % i_parent)
				if not jet:
					if jet == 0: # More than one match -- take note and continue
						count1 += 1
						continue
					else:  # jet == None
						# No matches -- take note and continue
						count2 += 1
						continue
				self.hjetpT_chjet_matchedtoparton.Fill(jet.pt())

			# Fill histogram - full jets
			self.parent0match, self.parent1match = None, None
			for i_jh, jh in enumerate(jets_h):
				self.hjetpT_h.Fill(jh.pt())
				self.assign_poss_parton_match(jh, jet_matching_distance, jetR)

			# If we have matches, fill histograms
			for i_parent, parent in enumerate(self.parents):
				jet = getattr(self, "parent%imatch" % i_parent)
				if not jet:
					if jet == 0: # More than one match -- take note and continue
						count1 += 1
						continue
					else:  # jet == None
						# No matches -- take note and continue
						count2 += 1
						continue
				self.hjetpT_h_matchedtoparton.Fill(jet.pt())

			# Fill histogram - full jets, with alice selections
			for i_jha, jha in enumerate(jets_ha):
				self.hjetpT_ha.Fill(jha.pt())

					
					

	#---------------------------------------------------------------
	# Calculate the observable given a jet
	#---------------------------------------------------------------
	

	#---------------------------------------------------------------
	# Initiate scaling of all histograms and print final simulation info
	#---------------------------------------------------------------
	def scale_print_final_info(self, pythia):
		# Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
		scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)

		# for jetR in self.jetR_list:
		# 	hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
		# 	for h in getattr(self, hist_list_name):
		# 		h.Scale(scale_f)

		self.hNevents.SetBinError(1, 0)
		self.hjetpT_chjet.SetBinError(1, 0)
		self.hjetpT_h.SetBinError(1, 0)
		self.hjetpT_ha.SetBinError(1, 0)



################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
									 prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	# Could use --py-seed
	parser.add_argument('--user-seed', help='PYTHIA starting seed', default=1111, type=int)
	parser.add_argument('-o', '--output-dir', action='store', type=str, default='./',
						help='Output directory for generated ROOT file(s)')
	parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
						help="Filename for the (unscaled) generated particle ROOT TTree")
	parser.add_argument('--MPIon', action='store', type=int, default=1,
						help="MPI on or off")
	parser.add_argument('--ISRon', action='store', type=int, default=1,
						help="ISR on or off")
	parser.add_argument('-cf', '--config_file', action='store', type=str, default='config/angularity.yaml',
						help="Path of config file for observable configurations")
	parser.add_argument('--weightON', action='store', type=int, default=1, help="'1' turns weights on, '0' turns them off")
	parser.add_argument('--ev-num-base', action='store', type=int, default=0, help="for merged files, this counts the total number of events")
	
	

	args = parser.parse_args()
	pinfo("The arguments to run are: ", args)

	# If invalid configFile is given, exit
	if not os.path.exists(args.config_file):
		print('File \"{0}\" does not exist! Exiting!'.format(args.config_file))
		sys.exit(0)

	# Use PYTHIA seed for event generation
	if args.user_seed < 0:
		args.user_seed = 1111

	# Have at least 1 event
	if args.nev < 1:
		args.nev = 1


	process = PythiaQuarkGluon(config_file=args.config_file, output_dir=args.output_dir, args=args)
	process.pythia_quark_gluon(args)
