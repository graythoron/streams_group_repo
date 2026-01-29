import numpy as np
import agama
from scipy.stats import gaussian_kde
import logging
import matplotlib.pyplot as plt

# Configure root logger for Jupyter output
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1, length=1, velocity=1)

# --- Main Function ---
def compute_boundness_recursive_BFE(
    positions_dark, velocity_dark, mass_dark,
    positions_star=None, velocity_star=None, mass_star=None,
    center_position=[], center_velocity=[],
    recursive_iter_converg=50, BFE_lmax=8,
    center_method='density_peak', center_params=None, center_vel_with_KDE=True,
    center_on='star', vel_rmax=5.0, tol_frac_change=0.0001,
    verbose=True, return_history=False, return_energy=False
):
    """
    Compute boundness of DM (and optionally stars) via recursive Agama multipole fits.
    Please make sure the agama.setUnits are appropriately picked based on the data.  
    Boundness criterion: total energy = potential + kinetic < 0.

    Parameters
    ----------
    positions_dark : (N_d,3) array
        Dark matter positions (kpc).
    velocity_dark : (N_d,3) array
        Dark matter velocities (km/s).
    mass_dark : (N_d,) array
        Dark matter masses (M_sun).
    positions_star : (N_s,3) array, optional
        Star positions.
    velocity_star : (N_s,3) array, optional
        Star velocities.
    mass_star : (N_s,) array, optional
        Star masses.
    center_position : (3,) array, optional
        User-specified center. If [], computed by `center_method` on `center_on` data.
    center_velocity : (3,) array, optional
        User-specified center velocity. If [], computed with KDE center.
    recursive_iter_converg : int, default=10
        Max iterations.
    BFE_lmax : int, default=8
        Multipole order.
    center_method : str, default='kde', 
        Options: 'shrinking_sphere', 'density_peak', 'kde'.
    center_vel_with_KDE: bool, default=True, centers KDE & density peak on phase-space, not just positions.  
    center_params : dict, optional
        Extra kwargs for center finder.
    center_on : {'dark','star','both'}, default='both'
        Which particles to use for center finding.
    vel_rmax : float, default=5.0
        Min radius for velocity center (kpc). Only active in non-KDE method. 
    tol_frac_change : float, default=0.0001
        Convergence on bound fraction change.
    verbose : bool, default=False
        Print diagnostic messages.
    return_history : bool, default=False
        Return per-iteration bound masks if True.
    Returns
    -------
    If no stars:
      bound_dark_final : (N_d,) int array (1 bound, 0 unbound)
      [bound_history_dm] if return_history
    If stars:
      bound_dark_final, bound_star_final [, bound_history_dm, bound_history_star]
    """
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

        
    # Determine center position
    if len(center_position) < 1:
        if center_vel_with_KDE:
            assert center_method != 'shrinking_sphere', "Centering method should be densit_peak/KDE. or set center_vel_with_KDE = False"
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
        logger.info(f"Iter {i}: Δ bound mask = {frac_change:.4f}")
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

    if (return_energy=='total') or (return_energy==True):
        return ((phi+kin)[:n_dark].copy(), (phi+kin)[n_dark:].copy())
    if return_energy=='phi':
        return (phi[:n_dark].copy(), phi[n_dark:].copy())
    else:
        return tuple(results), center_position, center_velocity ##only for Nora's purposes. 

def _find_center_position(positions, masses, method='shrinking_sphere', **kwargs):
    """
    Dispatch to various center-finding routines.
    """
    if method == 'shrinking_sphere':
        return _shrinking_sphere_center(positions, masses,
                                        r_init=kwargs.get('r_init', 30.0),
                                        shrink_factor=kwargs.get('shrink_factor', 0.9),
                                        min_particles=kwargs.get('min_particles', 100))

    # # does the KDE by default on either pos or posvel array. The pos will also be used to compute density_peak. 
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
        logger = logging.getLogger('boundness')
        logger.info(f"Returning the KDE center.")
    
    return centroid
    

def _shrinking_sphere_center(positions, masses, r_init=30.0,
                             shrink_factor=0.9, min_particles=10):
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