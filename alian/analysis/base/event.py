import heppyy.util.fastjet_cppyy
from cppyy.gbl import fastjet as fj
from cppyy.gbl import std

import heppyy

from .selection import EventSel, RCTSel, TrigSel

alian = heppyy.load_cppyy("alian")

class Event:
    def __init__(self, ev):
        self.run_number = ev.data["run_number"]
        self.multiplicity = ev.data["multiplicity"]
        self.centrality = ev.data["centrality"]
        self.occupancy = ev.data["occupancy"]
        self.event_sel = EventSel(ev.data["event_sel"])
        self.trig_sel = TrigSel(ev.data["trig_sel"])
        self.rct = RCTSel(ev.data["rct"])

class FlatEvent:
    """Lightweight event wrapper for a flat track tree.

    Exposes event-level attributes for the event selector to check against.
    Currently only track_count is available since the flat tree has no
    centrality, vertex, or other event-level metadata. Add more attributes
    here as needed when event-level cuts are defined.
    """
    def __init__(self, ev):
        self.event_id    = ev.event_id
        self.track_count = ev.track_count
        self.counter     = ev.counter
        self._px         = ev.data["px"]
        self._py         = ev.data["py"]
        self._pz         = ev.data["pz"]
        self._energy     = ev.data["energy"]
        self._label      = ev.data["label"]

def get_tracks(ev):
    """Get all tracks from an event (no track selections applied)."""
    return alian.numpy_ptetaphi_to_tracks(
        ev.data["track_pt"],
        ev.data["track_eta"],
        ev.data["track_phi"],
        ev.data["track_sel"],
        0,
    )


def get_selected_tracks(ev, selector):
    """Get selected tracks from an event (track selections applied)."""
    return std.vector[fj.PseudoJet](
        [
            t
            for t in alian.numpy_ptetaphi_to_tracks(
                ev.data["track_pt"],
                ev.data["track_eta"],
                ev.data["track_phi"],
                ev.data["track_sel"],
                0,
            )
            if selector.selects(t)
        ]
    )

def get_selected_tracks_from_pxpypze(ev, selector):
    """Get tracks from a flat tree with px/py/pz/energy branches."""
    tracks = std.vector[fj.PseudoJet](
        [
            fj.PseudoJet(float(px), float(py), float(pz), float(e))
            for px, py, pz, e in zip(
                ev.data["px"],
                ev.data["py"],
                ev.data["pz"],
                ev.data["energy"],
            )
        ]
    )
    return std.vector[fj.PseudoJet]([t for t in tracks if selector.selects(t)])

def get_partons(ev):
    """Get the two hard partons from an event (stored as event-level scalars)."""
    parton1 = fj.PseudoJet(
        float(ev.px_parton1),
        float(ev.py_parton1),
        float(ev.pz_parton1),
        float(ev.energy_parton1)
    )
    parton1.set_user_index(int(ev.label_parton1))

    parton2 = fj.PseudoJet(
        float(ev.px_parton2),
        float(ev.py_parton2),
        float(ev.pz_parton2),
        float(ev.energy_parton2)
    )
    parton2.set_user_index(int(ev.label_parton2))

    return fj.sorted_by_pt(std.vector[fj.PseudoJet]([parton1, parton2]))

def get_clusters(ev):
    """Get all clusters from an event (no cluster selections applied)."""
    return alian.numpy_energyetaphi_to_clusters(
        ev.data["cluster_energy"],
        ev.data["cluster_eta"],
        ev.data["cluster_phi"],
        ev.data["cluster_m02"],
        ev.data["cluster_m20"],
        ev.data["cluster_ncells"],
        ev.data["cluster_time"],
        ev.data["cluster_exoticity"],
        ev.data["cluster_dbc"],
        ev.data["cluster_nlm"],
        ev.data["cluster_defn"],
        ev.data["cluster_matched_track_n"],
        ev.data["cluster_matched_track_delta_eta"],
        ev.data["cluster_matched_track_delta_phi"],
        ev.data["cluster_matched_track_p"],
        ev.data["cluster_matched_track_pt"],
        ev.data["cluster_matched_track_sel"],
        0,
    )


def get_selected_clusters(ev, selector):
    """Get selected clusters from an event (cluster selections applied)."""
    return [
        c
        for c in alian.numpy_energyetaphi_to_clusters(
            ev.data["cluster_energy"],
            ev.data["cluster_eta"],
            ev.data["cluster_phi"],
            ev.data["cluster_m02"],
            ev.data["cluster_m20"],
            ev.data["cluster_ncells"],
            ev.data["cluster_time"],
            ev.data["cluster_exoticity"],
            ev.data["cluster_dbc"],
            ev.data["cluster_nlm"],
            ev.data["cluster_defn"],
            ev.data["cluster_matched_track_n"],
            ev.data["cluster_matched_track_delta_eta"],
            ev.data["cluster_matched_track_delta_phi"],
            ev.data["cluster_matched_track_p"],
            ev.data["cluster_matched_track_pt"],
            ev.data["cluster_matched_track_sel"],
            0,
        )
        if selector.selects(c)
    ]
