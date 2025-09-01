import logging
from typing import Any
from __future__ import annotations

# # 1. Create a custom handler that flushes after every log record is emitted
class _LiveLogHandler(logging.StreamHandler):
    """
    A custom logging handler that flushes the stream after each log record.
    Ensures that log messages appear in real-time in buffered environments
    like Jupyter notebooks.
    """
    def emit(self, record):
        # Call the original emit method to format and write the record
        super().emit(record)
        # Explicitly flush the stream to ensure immediate output
        self.flush()

# --- Your Logging Setup ---

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the lowest level the logger will handle

# Remove any existing handlers to ensure a clean setup
if logger.hasHandlers():
    logger.handlers.clear()

# Create an instance of our custom "live" handler
# We no longer use basicConfig, as we are creating the handler manually.
handler = _LiveLogHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(handler)

# --- Main Function ---
def compute_iterative_boundness(
    positions_dark: np.ndarray, 
    velocity_dark: np.ndarray, 
    mass_dark: np.ndarray | float,
    positions_star: np.ndarray | None = None, 
    velocity_star: np.ndarray | None = None, 
    mass_star: np.ndarray | float | None = None,
    center_position: np.ndarray | list = [], 
    center_velocity: np.ndarray | list = [],
    recursive_iter_converg: int = 50, 
    potential_compute_method: str = 'tree',
    BFE_lmax: int = 8, 
    softening_tree: float = 0.03, 
    G: float = 4.30092e-6,
    center_method: str = 'density_peak', 
    center_params: Any = None, 
    center_vel_with_KDE: bool = True,
    center_on: str = 'star', 
    vel_rmax: float = 5.0, 
    tol_frac_change: float = 0.0001,
    verbose: bool = True, 
    return_history: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    
    """
    Compute boundness of DM (and optionally stars) via iterative unbinding.
    
    Uses either Agama multipole expansion (BFE) or pyfalcon tree code to compute
    gravitational potential. Iteratively removes unbound particles until convergence.
    Boundness criterion: total energy = potential + kinetic < 0.

    Parameters
    ----------
    positions_dark : (N_d, 3) array
        Dark matter positions (kpc).
    velocity_dark : (N_d, 3) array
        Dark matter velocities (km/s).
    mass_dark : (N_d,) array or scalar
        Dark matter masses (M_sun). If scalar, applied to all particles.
    positions_star : (N_s, 3) array, optional
        Star positions (kpc).
    velocity_star : (N_s, 3) array, optional
        Star velocities (km/s).
    mass_star : (N_s,) array or scalar, optional
        Star masses (M_sun). If scalar, applied to all particles.
    center_position : (3,) array, optional
        User-specified center position (kpc). If empty, computed automatically.
    center_velocity : (3,) array, optional
        User-specified center velocity (km/s). If empty, computed automatically.
    
    Algorithm Parameters
    --------------------
    recursive_iter_converg : int, default=50
        Maximum iterations for convergence.
    potential_compute_method : {'tree', 'bfe'}, default='tree'
        Potential computation method:
        - 'tree': pyfalcon tree code (more accurate with sensible choice of softening)
        - 'bfe': Agama multipole expansion (approximate.)
    BFE_lmax : int, default=8
        Maximum multipole order for BFE method.
    softening_tree : float, default=0.03
        Gravitational softening length for tree method (kpc). 
        Recommended: ~0.5-1 × your simulation's force softening.
    G : float, default=4.30092e-6
        Gravitational constant in units (L:kpc, vel: km/s, M:Msun units).
    tol_frac_change : float, default=0.0001
        Convergence tolerance on bound fraction change between iterations.

    Centering Parameters
    -------------------
    center_method : {'density_peak', 'kde', 'shrinking_sphere'}, default='density_peak'
        Method for finding center position.
    center_vel_with_KDE : bool, default=True
        Use KDE for velocity centering (vs simple average within vel_rmax).
    center_on : {'star', 'dark', 'both'}, default='star'
        Which particle type to use for center finding.
    center_params : dict, optional
        Additional parameters for center finding method.
    vel_rmax : float, default=5.0
        Maximum radius for velocity centering when not using KDE (kpc).

    Output Control
    --------------
    verbose : bool, default=True
        Print iteration diagnostics.
    return_history : bool, default=False
        Return bound masks for each iteration.

    Returns
    -------
    If no stars provided:
        bound_dark_final : (N_d,) bool array
            Final bound mask for dark matter (True=bound, False=unbound).
        [bound_history_dm] : list of arrays, optional
            Bound masks for each iteration if return_history=True.
            
    If stars provided:
        bound_dark_final : (N_d,) bool array
        bound_star_final : (N_s,) bool array  
        [bound_history_dm, bound_history_star] : list of arrays, optional
    
    Notes
    -----
    - Ensure agama.setUnits() matches your data units before calling: agama.setUnits(mass=1, length=1, velocity=1) kpc, km/s, Msun
    - For tree method: softening should ideally match simulation resolution. Make sure G is in right units. 
    - For BFE method: higher lmax = more accurate but slower        
    """
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)

    # Input validation and normalization
    # ==================================

    # Validate string parameters (case-insensitive)
    potential_compute_method = potential_compute_method.lower()
    center_method = center_method.lower()
    center_on = center_on.lower()
    
    valid_potential_methods = ['tree', 'bfe']
    valid_center_methods = ['density_peak', 'kde', 'shrinking_sphere']
    valid_center_on = ['star', 'dark', 'both']
    
    assert potential_compute_method in valid_potential_methods, \
        f"potential_compute_method must be one of {valid_potential_methods}, got '{potential_compute_method}'"
    assert center_method in valid_center_methods, \
        f"center_method must be one of {valid_center_methods}, got '{center_method}'"
    assert center_on in valid_center_on, \
        f"center_on must be one of {valid_center_on}, got '{center_on}'"
    
    # Validate dark matter inputs
    positions_dark = np.asarray(positions_dark)
    velocity_dark = np.asarray(velocity_dark)
    assert positions_dark.shape[1] == 3, "positions_dark must have shape (N, 3)"
    assert velocity_dark.shape[1] == 3, "velocity_dark must have shape (N, 3)"
    assert positions_dark.shape[0] == velocity_dark.shape[0], \
        "positions_dark and velocity_dark must have same length"
    
    n_dark = positions_dark.shape[0]
    
    # Handle mass_dark (scalar or array)
    if np.isscalar(mass_dark):
        mass_dark = np.full(n_dark, mass_dark)
    else:
        mass_dark = np.asarray(mass_dark)
        assert len(mass_dark) == n_dark, \
            f"mass_dark array length ({len(mass_dark)}) must match positions ({n_dark})"
    
    # Validate star inputs if provided
    has_stars = positions_star is not None
    if has_stars:
        positions_star = np.asarray(positions_star)
        assert velocity_star is not None, "velocity_star required when positions_star provided"
        assert mass_star is not None, "mass_star required when positions_star provided"
        
        velocity_star = np.asarray(velocity_star)
        assert positions_star.shape[1] == 3, "positions_star must have shape (N, 3)"
        assert velocity_star.shape[1] == 3, "velocity_star must have shape (N, 3)"
        assert positions_star.shape[0] == velocity_star.shape[0], \
            "positions_star and velocity_star must have same length"
            
        n_star = positions_star.shape[0]
        
        # Handle mass_star (scalar or array)
        if np.isscalar(mass_star):
            mass_star = np.full(n_star, mass_star)
        else:
            mass_star = np.asarray(mass_star)
            assert len(mass_star) == n_star, \
                f"mass_star array length ({len(mass_star)}) must match positions ({n_star})"
    
    # Validate center_on makes sense given available data
    if center_on == 'star' and not has_stars:
        raise ValueError("center_on='star' requires star data to be provided")
    # Validate numerical parameters
    assert recursive_iter_converg > 0, "recursive_iter_converg must be positive"
    assert BFE_lmax > 0, "BFE_lmax must be positive"
    assert softening_tree > 0, "softening_tree must be positive"
    assert G > 0, "G must be positive"
    assert vel_rmax > 0, "vel_rmax must be positive"
    assert 0 < tol_frac_change < 1, "tol_frac_change must be between 0 and 1"
    
    # Validate center arrays if provided
    if len(center_position) > 0:
        center_position = np.asarray(center_position)
        assert len(center_position) == 3, "center_position must be length 3"
    if len(center_velocity) > 0:
        center_velocity = np.asarray(center_velocity)
        assert len(center_velocity) == 3, "center_velocity must be length 3"
    
    if potential_compute_method == 'tree':
        try:
            import pyfalcon
        except ImportError:
            logger.info(f'falcON not available for tree method.')
            potential_compute_method = 'bfe'
    
    # Setup logger
    logger = logging.getLogger('boundness')
    if verbose and not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Stack inputs for center determination
    if positions_star is not None and center_on == 'both':
        pos_for_center = np.vstack((positions_dark, positions_star))
        mass_for_center = np.concatenate((mass_dark, mass_star))
        vel_for_center = np.vstack((velocity_dark, velocity_star))
        
    elif positions_star is not None and center_on == 'star':
        pos_for_center = positions_star
        mass_for_center = mass_star
        vel_for_center = velocity_star
    else:
        pos_for_center = positions_dark
        mass_for_center = mass_dark
        vel_for_center = velocity_dark

    # Main function
    # ==================================

    # Determine center position
    if len(center_position) < 1:
        if center_vel_with_KDE:
            assert center_method != 'shrinking_sphere', "Centering method should be density_peak/KDE. or set center_vel_with_KDE = False"
            computed_central_loc = _find_center_position(
                np.hstack((pos_for_center, vel_for_center)), mass_for_center,
                method=center_method, **(center_params or {}))
                
            center_position, center_velocity = computed_central_loc[:3], computed_central_loc[3:]
        else:
            center_position = _find_center_position(
                pos_for_center, mass_for_center,
                method=center_method, **(center_params or {}))
        
    # Determine center velocity
    if len(center_velocity) < 1:
        ## base it on 10% dispersion... 
        vel_rmax = min(vel_rmax, 0.1 * np.std(np.linalg.norm(vel_for_center, axis=1)))
        dist2 = np.sum((pos_for_center - center_position)**2, axis=1)
        sel = dist2 < vel_rmax**2
        if np.any(sel):
            center_velocity = np.average(
                vel_for_center[sel], axis=0, weights=mass_for_center[sel])
        else:
            center_velocity = np.average(
                vel_for_center, axis=0, weights=mass_for_center)
    
    logger.info(f"Center position (method={center_method}): {np.around(center_position, decimals=2)}")
    logger.info(f"Center velocity: {np.around(center_velocity, decimals=2)}")

    # Prepare arrays for recursion
    if positions_star is not None:
        pos_all = np.vstack((positions_dark, positions_star))
        vel_all = np.vstack((velocity_dark, velocity_star))
        mass_all = np.concatenate((mass_dark, mass_star))
        n_dark = len(positions_dark)
    else:
        pos_all, vel_all, mass_all = positions_dark.copy(), velocity_dark.copy(), mass_dark.copy()
        n_dark = len(pos_all)

    # Recenter positions and velocities
    pos_rel = pos_all - center_position
    vel_rel = vel_all - center_velocity

    bound_history_dm = []
    bound_history_star = []
    mask_all = np.ones(len(pos_all), dtype=bool)

    # Define a minimum number of particles required to build a reliable potential
    min_particles_for_model = 5  # Example value, adjust as needed
    
    # Recursive energy cut
    for i in range(recursive_iter_converg):
         # --- New Check 1: Ensure enough particles are bound before iteration ---
        num_bound = np.sum(mask_all)
        if num_bound < min_particles_for_model:
            logger.info(f"Stopping: Only {num_bound} particles remaining, which is below the threshold of {min_particles_for_model}.")
            break
            
        if potential_compute_method == 'tree':
            # --- make unbound particles negligible mass so they don't source gravity.
            mass_all[~mask_all] = 0.01 # can not be zero, pos still required to find phi at those locations.
            
            _, phi = pyfalcon.gravity(
                pos_rel, mass_all*G, 
                eps=softening_tree, theta=0.4) # theta=0.4 has very high accuracy compared to regular codes.
        else:
            # --- simply remove unbound particles from Agama potential creation. 
            pot = agama.Potential(
                type='Multipole', particles=(pos_rel[mask_all], mass_all[mask_all]),
                symmetry='n', lmax=BFE_lmax)
            
            phi = pot.potential(pos_rel)
            
        
        kin = 0.5 * np.sum(vel_rel**2, axis=1)
        bound_mask = (phi + kin) < 0

        bound_history_dm.append(bound_mask[:n_dark].copy())
        if positions_star is not None:
            bound_history_star.append(bound_mask[n_dark:].copy())

        frac_change = np.mean(bound_mask != mask_all)
        logger.info(f"Iter {i}: Δ bound mask = {frac_change:.5f}")
        mask_all = bound_mask
        if frac_change < tol_frac_change:
            logger.info(f"Converged after {i+1} iterations.")
            break

    # Final masks
    bound_dark_final = mask_all[:n_dark].astype(int)
    results = [bound_dark_final]
    if positions_star is not None:
        results.append(mask_all[n_dark:].astype(int))
    if return_history:
        results.append(bound_history_dm)
        if positions_star is not None:
            results.append(bound_history_star)

    return tuple(results), center_position, center_velocity 

def _find_center_position(
    positions: np.ndarray, masses: np.ndarray, 
    method: str = 'shrinking_sphere', **kwargs
) -> np.ndarray:
    """
    Dispatch to various center-finding routines.
    """
    if method == 'shrinking_sphere':
        return _shrinking_sphere_center(positions, masses,
                                        r_init=kwargs.get('r_init', 30.0),
                                        shrink_factor=kwargs.get('shrink_factor', 0.9),
                                        min_particles=kwargs.get('min_particles', 100))

    # # does the KDE by default on either pos or posvel array. The pos will also be used to compute density_peak. 
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(positions.T, weights=masses)
    sample = positions[np.random.choice(len(positions), size=min(10_000, len(positions)), replace=False)]
    dens = kde(sample.T)
    Npick = max(10, int(len(dens)*0.01))
    idxs = np.argsort(dens)[-Npick:]
    centroid = np.average(sample[idxs], axis=0)
    
    if method == 'density_peak':
        print(f'Using the KDE center: {centroid}')
        pos_c = positions - centroid

        dens = agama.Potential(type='Multipole', particles=(pos_c, masses), symmetry='n', lmax=kwargs.get('lmax', 8)).density(pos_c[:, :3])
        
        ## pick max of 1% particles or 20.
        Npick = max(20, int(len(dens)*0.01))
        idxs = np.argsort(dens)[-Npick:]

        ## add the max dens loc to the original centroid
        centroid += np.average(pos_c[idxs], axis=0, weights=dens[idxs])
        print(f'Density peak found at: {centroid}')
    
    else:
        logger.info(f"Returning the KDE center.")
    
    return centroid
    
def _shrinking_sphere_center(
    positions: np.ndarray, masses: np.ndarray, 
    r_init: float = 30.0, shrink_factor: float = 0.9, 
    min_particles: int = 10
) -> np.ndarray:
    """
    Mass-weighted shrinking spheres algorithm.
    """
    center = np.average(positions, axis=0, weights=masses)
    radius = r_init
    while True:
        dist2 = np.sum((positions - center)**2, axis=1)
        mask = dist2 < radius**2
        if np.sum(mask) < min_particles:
            break
        center = np.average(positions[mask], axis=0, weights=masses[mask])
        radius *= shrink_factor
    return center