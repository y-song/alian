from time import perf_counter

from ROOT import TFile
import heppyy
import heppyy.util.fastjet_cppyy
from cppyy.gbl import fastjet as fj
from cppyy.gbl import std

from alian.io.data_io import FlatFileInput
from .event import FlatEvent, get_selected_tracks_from_pxpypze, get_partons
from .jet_finder import JetFinder
from .logs import set_up_logger
from .output import Output
from .selector import AnalysisSelector
from .utils import is_slurm, read_yaml


class BaseAnalysis(heppyy.GenericObject):
    _defaults = {}

    def __init__(self, **kwargs):
        super(BaseAnalysis, self).__init__(**kwargs)
        for k, val in self.__class__._defaults.items():
            if not hasattr(self, k) or getattr(self, k) is None:
                setattr(self, k, val)


class AnalysisBaseFlat:
    """Base class for analysis over a flat track tree.

    The tree is expected to have branches: px, py, pz, energy, label,
    grouped by eventID.

    Track-level and event-level selectors are supported via the `selections`
    block in the config YAML. If no `event` block is present, all events
    pass. If no `track` block is present, all tracks are kept.

    Override analyze_event() at minimum. Override init_analysis() to
    parse analysis-specific config. Override init_output() to add
    custom histograms or trees.
    """

    _defaults = {}

    def __init__(self, input_file, output_file, cfg, tree_struct, nev=-1):
        self.input_file  = input_file
        self.output_file = output_file
        self.cfg_file    = cfg
        self.tree_struct = tree_struct
        self.nev         = nev

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

        self.data_source = FlatFileInput(
            self.input_file,
            yaml_file=self.tree_struct,
            n_events=self.nev,
        )

        # selectors
        self.logger.info("Configuring selectors...")
        self.selector = AnalysisSelector.load(self.cfg)
        self.selector.dump()
        self.logger.info("Selectors configured.")

        # jet finder
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
        slurm_check = is_slurm()
        for ev in self.data_source.next_event(disable_bar=slurm_check):
            # phase 1: build event-level object and apply event selection
            self.build_event(ev)
            if self.selector.event and not self.selector.event.selects(self.event):
                continue
            # phase 2: build tracks and jets, then analyze
            self.build_event_objs(ev)
            self.analyze_event()
        self.note_time("Events analyzed")

    def analyze_event(self):
        raise NotImplementedError("Must implement analyze_event()!")

    def build_event(self, ev):
        """Wrap raw event struct into a FlatEvent."""
        self.event = FlatEvent(ev)

    def build_event_objs(self, ev):
        """Build selected tracks and optionally find jets."""
        self.tracks = get_selected_tracks_from_pxpypze(ev, self.selector.track)
        self.partons = get_partons(ev)
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


def add_default_args_flat(parser):
    parser.add_argument("-i", "--input-file",  type=str, required=True,
                        help="Input ROOT file.")
    parser.add_argument("-o", "--output-file", type=str, default="analysis.root",
                        help="Output ROOT file.")
    parser.add_argument("-c", "--config-file", type=str, required=True,
                        help="YAML config file.")
    parser.add_argument("-n", "--nev",         type=int, default=-1,
                        help="Number of events to process (-1 for all).")
    parser.add_argument("-t", "--tree-struct", type=str, default="/home/youqi/alian/alian/config/flat_tstruct.yaml",
                        help="YAML file describing the tree structure.")
    return parser