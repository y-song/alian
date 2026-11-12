import heppyy
fj = heppyy.load_cppyy('fastjet')


class BkgEstimator:
    """Event-wide rho estimator (grid median or jet median) built from a
    `bkg_estimator` config block.

    Keys: type ('grid' or 'jet'), max_eta, and bge_rho_grid_size (grid) or
    bge_R and ghost_area (jet; ghost_area is the active-area ghost density, default 0.01). Holds the fastjet objects so that they outlive the estimator.
    """

    def __init__(self, bge_cfg, logger=None):
        bge_cfg = bge_cfg or {}
        max_eta = bge_cfg.get('max_eta', 0.9)
        self.type = bge_cfg.get('type', 'grid')
        self.max_eta = max_eta
        if self.type == 'grid':
            grid_size = bge_cfg.get('bge_rho_grid_size', 0.1)
            self.bge = fj.GridMedianBackgroundEstimator(max_eta, grid_size)
            params = [f"\ttype: {self.type!r}", f"\tmax_eta: {repr(max_eta)}", f"\tbge_rho_grid_size: {repr(grid_size)}"]
        elif self.type == 'jet':
            bge_R = bge_cfg.get('bge_R', 0.4)
            ghost_area = bge_cfg.get('ghost_area', 0.01)
            sel_not = getattr(fj, "operator!")  # cppyy doesn't bind fastjet's free-function Selector negation to ~/-
            self.jet_selector = fj.SelectorAbsEtaMax(max_eta - bge_R) * sel_not(fj.SelectorNHardest(2))
            self.jet_def = fj.JetDefinition(fj.kt_algorithm, bge_R)
            self.area_def = fj.AreaDefinition(fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(max_eta, 1, ghost_area))
            self.bge = fj.JetMedianBackgroundEstimator(self.jet_selector, self.jet_def, self.area_def)
            params = [f"\ttype: {self.type!r}", f"\tmax_eta: {repr(max_eta)}", f"\tbge_R: {repr(bge_R)}", f"\tghost_area: {repr(ghost_area)}"]
        else:
            raise ValueError(f"Unknown bkg_estimator type: {self.type!r} (expected 'grid' or 'jet')")
        if logger is not None:
            logger.info("Background estimator configuration:\n" + "\n".join(params), stacklevel=2)

    def estimate(self, particles):
        """Set the particles of the event and return (rho, sigma)."""
        self.bge.set_particles(particles)
        return self.bge.rho(), self.bge.sigma()

    def rho_per_grid(self, particles):
        """Background density (scalar pT sum / area) for every grid cell, i.e. the
        full per-cell distribution that GridMedianBackgroundEstimator.rho() takes
        the median of. Mirrors the binning logic in fastjet's
        GridMedianBackgroundEstimator::set_particles(). particles must be those
        passed to estimate()."""
        scalar_pt = [0.0] * self.bge.n_tiles()
        for t in particles:
            j = self.bge.tile_index(t)
            if j >= 0:
                scalar_pt[j] += t.pt()
        return [pt / self.bge.tile_area(j) for j, pt in enumerate(scalar_pt)]

    def rho_per_jet(self):
        """Background density (pT / area) for every background jet used in the
        last rho()/sigma() call, i.e. the full per-jet distribution that
        JetMedianBackgroundEstimator.rho() takes the median of. Mirrors the
        per-jet density used in fastjet's compute_rho() (default
        _use_area_4vector = True, so area_4vector().perp() is used)."""
        return [j.pt() / j.area_4vector().perp() for j in self.bge.jets_used()]

    def rho_per_cell(self, particles):
        """Full per-cell (grid) or per-jet distribution that rho() takes the
        median of, dispatching on the configured estimator type. Lets analyses
        stay agnostic to which estimator is active."""
        if self.type == 'grid':
            return self.rho_per_grid(particles)
        return self.rho_per_jet()
