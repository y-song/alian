from time import perf_counter

import tqdm
from ROOT import TFile
import heppyy.util.fastjet_cppyy
import heppyy.util.heppyy_cppyy
from cppyy.gbl import heppyy
from cppyy.gbl import fastjet as fj
from cppyy.gbl import std

from .jet_finder import JetFinder
from .logs import set_up_logger
from .output import Output
from .selector import AnalysisSelector
from .utils import read_yaml


class AnalysisBaseHepMC:
    """Base class for analysis over HepMC-format files read via heppyy.HybridFile.

    Override analyze_event() at minimum. Override init_analysis() to parse
    analysis-specific config. Override init_output() to customize histograms.

    After build_event_objs() runs, each call to analyze_event() has access to:
        self.parts          — all particles from the event
        self.tracks  — particles passing the track selection (pt_min)
        self.partons        — parton-level objects
        self.ev_info        — event info (weight, cross, x, y)
        self.jets           — jets found from parts_selected (if jet_finder in config)

    Wake particles are included when self.wake is True. Set this in init_analysis()
    by reading the 'wake' key from the analysis config block.
    """

    _defaults = {}

    def __init__(self, input_file, output_file, cfg, nev=-1):
        self.input_file  = input_file
        self.output_file = output_file
        self.cfg_file    = cfg
        self.nev         = nev
        self.wake        = False

    def run(self):
        self.start_time = perf_counter()
        self.logger = set_up_logger(__name__)
        self.logger.info("Starting analysis!")
        self.logger.info("Configuring analysis pipeline...")
        self.init()
        self.logger.info("Analysis pipeline configured.")
        self.analyze_events()
        self.finalize()
        self.save()
        self.note_time("Analysis complete")

    def init(self):
        self.cfg = read_yaml(self.cfg_file)
        self._validate_cfg(self.cfg)
        self.logger.info("Config schema validated.")

        self.data_source = heppyy.HybridFile(self.input_file)

        self.logger.info("Configuring selectors...")
        self.selector = AnalysisSelector.load(self.cfg)
        self.selector.dump()
        self.logger.info("Selectors configured.")

        if "jet_finder" in self.cfg:
            self.logger.info("Jet finder config found, jets will be found per event.")
            self.load_jets = True
            self.logger.info("Configuring jet finder...")
            self.jet_finder = JetFinder.load(self.cfg)
            self.jet_finder.dump()
            self.logger.info("Jet finder configured.")
        else:
            self.logger.info("No jet_finder block in config — no jets will be found.")
            self.load_jets = False

        self.logger.info("Configuring output...")
        self.init_output()
        self.logger.info("Output configured.")

        self.logger.info("Configuring analysis parameters...")
        self.init_analysis(self.cfg.get("analysis", {}))
        self.dump()
        self.logger.info("Analysis parameters configured.")

    def _validate_cfg(self, cfg: dict):
        req_headers = ["selections", "output"]
        if headers_not_in_cfg := [h for h in req_headers if h not in cfg]:
            raise KeyError(
                f"The following blocks must be present in the YAML config: {headers_not_in_cfg}"
            )

    def init_analysis(self, analysis_cfg: dict):
        """Parse analysis-specific config parameters. Override if needed."""
        self.logger.info("No analysis configuration to parse.")

    def init_output(self):
        self.output = Output.load(self.cfg)
        self.hists  = self.output.hists
        self.trees  = self.output.trees

    def analyze_events(self):
        self.logger.info("Analyzing events...")
        total = self.nev if self.nev > 0 else None
        pbar = tqdm.tqdm(total=total)
        ievent = 0
        while self.data_source.nextEvent():
            self.build_event_objs()
            self.analyze_event()
            ievent += 1
            pbar.update(1)
            if self.nev > 0 and ievent >= self.nev:
                break
        pbar.close()
        self.note_time("Events analyzed")

    def analyze_event(self):
        raise NotImplementedError("Must implement analyze_event()!")

    def build_event_objs(self):
        """Read one event from the HepMC file and build tracks and jets."""
        self.parts          = self.data_source.getParticles(include_wake=self.wake, charged_only=False)
        self.partons        = self.data_source.getPartons()
        self.weight         = self.data_source.info().weight()
        self.tracks = std.vector[fj.PseudoJet]([p for p in self.parts if self.selector.track.selects(p)])
        if self.load_jets:
            self.jets = self.jet_finder.find_jets(self.tracks)

    def finalize(self):
        """Called after event loop. Override if needed."""
        self.logger.info("Nothing to finalize.")

    def save(self):
        self.logger.info(f"Saving output to: {self.output_file}")
        with TFile(self.output_file, "RECREATE") as f:
            self.output.save(f)
        self.logger.info("Output saved.")

    def __getattr__(self, attr):
        if attr == "jets":
            raise AttributeError(
                "Jets not defined; include a `jet_finder` block in your config!"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def dump(self):
        cfg = "\n".join(
            [f"\t{p}: {repr(getattr(self, p))}" for p in self._defaults]
        )
        self.logger.info(f"{type(self).__name__} configuration:\n{cfg}", stacklevel=2)

    def note_time(self, msg):
        self.logger.info(
            f"{msg}: ------- {self.fmt_time(perf_counter() - self.start_time)} -------",
            stacklevel=2,
        )

    def fmt_time(self, seconds):
        m, s = divmod(round(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}h {m:02d}m {s:02d}s"


def add_default_args_hepmc(parser):
    parser.add_argument("-i", "--input-file",  type=str, required=True,
                        help="Input HepMC/hybrid file.")
    parser.add_argument("-o", "--output-file", type=str, default="analysis.root",
                        help="Output ROOT file.")
    parser.add_argument("-c", "--config-file", type=str, required=True,
                        help="YAML config file.")
    parser.add_argument("-n", "--nev",         type=int, default=-1,
                        help="Number of events to process (-1 for all).")
    return parser
