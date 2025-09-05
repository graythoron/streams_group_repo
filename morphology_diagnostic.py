import numpy as np
from scipy import linalg

# Optional Numba import
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(nopython=True, cache=True): # Dummy decorator
        def decorator(func):
            return func
        return decorator
    print("Numba not found. Running with pure NumPy/SciPy (might be slower for some operations).")

@jit(nopython=True, cache=True)
def _calculate_particle_distances_sq(xyz_coords):
    """
    Computes squared Euclidean distances from the origin for an array of coordinates.

    Parameters
    ----------
    xyz_coords : numpy.ndarray, shape (n, 3)
        Array of n particle coordinates (x, y, z).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Array of squared distances from the origin for each particle.
    """
    return np.sum(xyz_coords**2, axis=1)


@jit(nopython=True, cache=True)
def _compute_weighted_structure_tensor(coords, masses, weights):
    """
    Computes the weighted structure tensor (also known as the second moment tensor).

    The structure tensor S_ij is calculated as:
    S_ij = sum_k (mass_k * weight_k * coord_i_k * coord_j_k) / sum_k (mass_k * weight_k)
    where k iterates over particles. Eigenvectors of this tensor give the principal axes
    of the particle distribution, and eigenvalues are proportional to the square of the
    lengths of these principal axes.

    Parameters
    ----------
    coords : numpy.ndarray, shape (m, 3)
        Coordinates of m selected particles.
    masses : numpy.ndarray, shape (m,)
        Masses of the m selected particles.
    weights : numpy.ndarray, shape (m,)
        Additional weights for each particle (e.g., 1/Rsph_k^2 for reduced tensor).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The 3x3 weighted structure tensor. Returns a small diagonal matrix if no
        particles or zero total effective weight.
    """
    if coords.shape[0] == 0: # No particles
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    effective_weights = masses * weights # Element-wise
    sum_effective_weights = np.sum(effective_weights)

    if sum_effective_weights == 0:
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    structure = np.zeros((3, 3), dtype=coords.dtype)
    for k in range(coords.shape[0]):
        w_k = effective_weights[k]
        x_k = coords[k, 0]
        y_k = coords[k, 1]
        z_k = coords[k, 2]
        # Accumulating outer product x_k * x_k.T weighted by w_k
        structure[0, 0] += w_k * x_k * x_k
        structure[0, 1] += w_k * x_k * y_k
        structure[0, 2] += w_k * x_k * z_k
        # structure[1, 0] is structure[0, 1]
        structure[1, 1] += w_k * y_k * y_k
        structure[1, 2] += w_k * y_k * z_k
        # structure[2, 0] is structure[0, 2]
        # structure[2, 1] is structure[1, 2]
        structure[2, 2] += w_k * z_k * z_k
    
    # Symmetrize due to potential floating point asymmetries if calculated fully
    structure[1,0] = structure[0,1]
    structure[2,0] = structure[0,2]
    structure[2,1] = structure[1,2]
    
    return structure / sum_effective_weights


@jit(nopython=True, cache=True)
def _calculate_Rsphall_and_extract(
    all_coords,
    transform_matrix_cols_bca, 
    q_ratio, 
    s_ratio, 
    Rmax_scaled,
    Rmin_val 
    ):
    """
    Calculates ellipsoidal radius (Rsph) for all particles and extracts those within bounds.

    Rsph is a measure of distance in a coordinate system scaled by the current ellipsoidal
    shape (defined by q_ratio = b/a, s_ratio = c/a).
    The transformation `transform_matrix_cols_bca` (with columns e_b, e_c, e_a) rotates
    particles into the principal axis frame. Coordinates are then scaled:
    X_scaled_b = (X_b / (b/a)), X_scaled_c = (X_c / (c/a)), X_scaled_a = (X_a / 1).
    Rsph = norm([X_scaled_b, X_scaled_c, X_scaled_a]).
    Particles are selected if their Rsph is less than `Rmax_scaled` (and greater than
    a similarly scaled `Rmin_val`).

    Parameters
    ----------
    all_coords : numpy.ndarray, shape (N, 3)
        Coordinates of all N particles in the system.
    transform_matrix_cols_bca : numpy.ndarray, shape (3, 3)
        Transformation matrix whose columns are the eigenvectors e_b, e_c, e_a
        (intermediate, minor, major axes of the ellipsoid from the previous iteration).
    q_ratio : float
        Axis ratio b/a from the previous iteration.
    s_ratio : float
        Axis ratio c/a from the previous iteration.
    Rmax_scaled : float
        The scaled maximum radius (aperture) for selection. This is typically
        Rmax_physical / ( (q_ratio * s_ratio)**(1/3) ).
    Rmin_val : float
        The physical minimum radius for shell selection.

    Returns
    -------
    Rsphall_values : numpy.ndarray, shape (N,)
        Ellipsoidal radius for all N particles.
    extract_mask : numpy.ndarray, dtype bool, shape (N,)
        Boolean mask indicating which particles are selected.
    Rsph_extracted : numpy.ndarray
        Ellipsoidal radii of the selected particles.
    """
    if all_coords.shape[0] == 0:
        return (
            np.empty(0, dtype=all_coords.dtype),
            np.empty(0, dtype=np.bool_), # Numba needs np.bool_ for boolean arrays
            np.empty(0, dtype=all_coords.dtype)
            )

    coords_in_eigenframe_bca = np.dot(all_coords, transform_matrix_cols_bca)
            
    # Ensure scales have the same dtype as coordinates for Numba compatibility
    scales = np.array([q_ratio, s_ratio, 1.0], dtype=all_coords.dtype) 
    epsilon = 1e-9 # Smallest allowed ratio to prevent division by zero
    if scales[0] < epsilon: scales[0] = epsilon
    if scales[1] < epsilon: scales[1] = epsilon

    scaled_coords_bca = np.empty_like(coords_in_eigenframe_bca)
    for i in range(coords_in_eigenframe_bca.shape[0]):
        scaled_coords_bca[i,0] = coords_in_eigenframe_bca[i,0] / scales[0]
        scaled_coords_bca[i,1] = coords_in_eigenframe_bca[i,1] / scales[1]
        scaled_coords_bca[i,2] = coords_in_eigenframe_bca[i,2] / scales[2]

    Rsphall_values_sq = np.sum(scaled_coords_bca**2, axis=1)
    Rsphall_values = np.sqrt(Rsphall_values_sq)

    extract_mask = Rsphall_values < Rmax_scaled
    if Rmin_val > 0:
        vol_scale_factor_inv = 1.0
        # Check (q_ratio * s_ratio) to avoid issues if it's zero or negative (shouldn't be)
        # Product can be zero if s_ratio or q_ratio is zero.
        prod_qs = q_ratio * s_ratio
        if prod_qs > 1e-9 : 
             vol_scale_factor_inv = prod_qs**(1./3.)
        
        Rmin_ap_scaled = Rmin_val / vol_scale_factor_inv if vol_scale_factor_inv > 1e-9 else Rmin_val * 1e9
        extract_mask &= (Rsphall_values >= Rmin_ap_scaled)

    Rsph_extracted = Rsphall_values[extract_mask]
    return Rsphall_values, extract_mask, Rsph_extracted


def compute_morphological_diagnostics(
    XYZ,
    mass=None,
    Vxyz=None,
    Rmin=0.0,
    Rmax=1.0,
    reduced_structure=True,
    orient_with_momentum=True,
    tol=1e-4,
    max_iter=50,
    verbose=False,
    return_ellip_triax=True
    ):
    """
    Computes morphological diagnostics (axis lengths, principal axes, ellipticity,
    triaxiality) for a distribution of particles using an iterative approach based
    on the structure/inertia tensor.

    Method Overview:
    ----------------
    The function iteratively determines the shape of a particle distribution.
    1. Particle Selection: Initially, particles within a spherical shell (Rmin, Rmax)
       are selected. In subsequent iterations (if `reduced_structure` is True),
       particles are selected if they fall within an ellipsoid defined by the shape
       (axis ratios q=b/a, s=c/a) determined in the previous iteration. The
       ellipsoid's volume is typically kept comparable to the initial spherical volume.
    2. Structure Tensor: For the selected particles, a weighted structure tensor
       S_ij = sum(w_k * x_i_k * x_j_k) / sum(w_k) is computed.
       If `reduced_structure` is True, weights w_k are proportional to 1/Rsph_k^2,
       where Rsph_k is the particle's ellipsoidal radius normalized by the median
       ellipsoidal radius. This "reduces" the contribution of outer particles,
       focusing on the shape of the inner distribution.
    3. Diagonalization: The structure tensor is diagonalized to find its eigenvalues
       (proportional to squared principal axis lengths a^2, b^2, c^2) and
       eigenvectors (the principal axes e_a, e_b, e_c).
    4. Orientation (Optional): If `orient_with_momentum` is True and velocities
       (Vxyz) are provided, the minor principal axis (e_c) is aligned with the
       total angular momentum vector of the currently selected particles. The other
       axes are adjusted to maintain a right-handed orthonormal system.
    5. Convergence: New axis ratios (q_new = b/a, s_new = c/a) are calculated.
       The process repeats from step 1 (if `reduced_structure`) until the change
       in q and s between iterations is below the tolerance `tol`, or `max_iter`
       is reached.
    6. Output: The function returns the normalized principal semi-axis lengths (a,b,c
       such that a>=b>=c and a=1), the transformation matrix (rows are principal
       axes), and optionally ellipticity and triaxiality.

    Parameters
    ----------
    XYZ : numpy.ndarray, shape (n, 3)
        Particle coordinates (units of length L). XYZ[:,0]=X, XYZ[:,1]=Y, XYZ[:,2]=Z.
    mass : numpy.ndarray, optional, shape (n,)
        Particle masses (units of mass M). If None, assumes mass of 1.0 for all.
    Vxyz : numpy.ndarray, optional, shape (n, 3)
        Particle velocities (units of velocity V). Vxyz[:,0]=Vx, etc.
        Required if `orient_with_momentum` is True.
    Rmin : float, optional
        Minimum radius of the initial spherical shell for particle selection (L).
        For iterative reduced structure, this Rmin is effectively applied within
        the deformed ellipsoidal selection. Default is 0.0.
    Rmax : float, optional
        Maximum radius (aperture) for initial spherical particle selection (L).
        This also sets the scale for the iterative ellipsoidal selection. Default is 1.0.
    reduced_structure : bool, optional
        If True, adopts the iterative reduced form of the inertia tensor, where
        particle contributions are weighted by 1/Rsph^2 and the selection ellipsoid
        adapts. If False, a single pass is made on the initial spherical selection
        with uniform weights. Default is True.
    orient_with_momentum : bool, optional
        If True and `Vxyz` is provided, aligns the minor principal axis with the
        angular momentum vector of the selected particles *during each iteration*.
        This influences subsequent particle selection if `reduced_structure` is True.
        Default is True.
    tol : float, optional
        Tolerance for convergence of axis ratios (q=b/a, s=c/a) in the iterative
        process. Convergence is checked based on the maximum squared fractional
        change: max( (1 - q_new/q_old)^2, (1 - s_new/s_old)^2 ). Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations for convergence. Default is 50.
    verbose : bool, optional
        If True, print iteration progress and diagnostic information. Default is False.
    return_ellip_triax : bool, optional
        If True, return ellipticity and triaxiality in addition to `abc` and
        `Transform`. Default is False.

    Returns
    -------
    abc_normalized : numpy.ndarray, shape (3,)
        The principal semi-axis lengths (a,b,c) normalized such that a=1.0.
        Ordered as a >= b >= c.
    final_transform_matrix : numpy.ndarray, shape (3, 3)
        Orthogonal transformation matrix. Rows are the unit vectors of the
        principal axes (e_a, e_b, e_c) in the original coordinate system,
        corresponding to a_out, b_out, c_out.
        `Transform[0]` = major axis (e_a)
        `Transform[1]` = intermediate axis (e_b)
        `Transform[2]` = minor axis (e_c)
    ellip : float, optional
        Ellipticity (1 - c/a). Returned if `return_ellip_triax` is True.
        Value is between 0 (spherical) and 1 (infinitely thin disk/needle).
    triax : float, optional
        Triaxiality parameter ((a^2 - b^2) / (a^2 - c^2)), assuming a >= b >= c.
        Ranges from 0 (oblate, a=b>c) to 1 (prolate, a>b=c).
        Is NaN if a=c (spherical) and handled as 0.0 if a=b=c.
        Returned if `return_ellip_triax` is True.

    Guidance on `orient_with_momentum`:
    -----------------------------------
    - Iterative Alignment (this function's behavior if `orient_with_momentum=True`):
      Aligning the minor axis with the angular momentum of the *currently selected particles*
      in each iteration means the particle selection for the *next* iteration is influenced
      by this orientation. This can be useful for studying systems where shape and internal
      kinematics are expected to be tightly coupled at all scales considered by the iteration.
      It seeks a self-consistent shape-orientation state.

    - Final Alignment (Alternative Approach, not directly implemented here):
      An alternative is to compute the shape iteratively *without* momentum alignment during
      iterations (i.e., `orient_with_momentum=False`). After the shape converges, calculate the
      total angular momentum of the final particle set and then report the orientation of the
      derived ellipsoid with respect to this global angular momentum vector. This is simpler
      and often preferred if the goal is a single global shape and its relation to global spin.

    - For Dark Matter Halos:
      - If analyzing the intrinsic shape profile and how it correlates with the local angular
        momentum at different ellipsoidal radii, iterative alignment might be informative.
      - For a single, overall shape estimate (e.g., within R200), setting
        `orient_with_momentum=False` and then calculating the spin parameter or
        shape-spin alignment *after convergence* using the final particle set is common.
        This decouples the shape determination from a potentially noisy or radially varying
        momentum vector.
      - Iterative alignment can be sensitive if the angular momentum vector is weak or its
        direction changes significantly within the region being iterated upon.

    Warning for Shell Analysis with Iterative Momentum Orientation:
    -------------------------------------------------------------
    When using `orient_with_momentum=True` for a shell analysis (i.e., `Rmin > 0`),
    be aware that:
    - The angular momentum is calculated *only* for particles within that specific shell.
    - The iterative alignment attempts to align the minor axis of the *shell's inertia tensor*
      with the *shell's angular momentum vector*.
    - This orientation (and the subsequent shape refinement based on it) reflects the
      properties of that specific shell. It may not represent the global orientation or
      shape characteristics of the entire halo, as the shell's angular momentum and
      particle distribution can differ from the overall system or inner regions.
    - This approach is valid for studying the properties of discrete shells but exercise
      caution when extrapolating these shell-specific oriented shapes to the entire object.
    """
    # --- Input Validation ---
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must be a 2D array with shape (n, 3).")
    n_particles = XYZ.shape[0]

    if mass is None:
        mass_arr = np.ones(n_particles, dtype=float)
    else:
        mass_arr = np.asarray(mass, dtype=float)
        if mass_arr.ndim != 1 or mass_arr.shape[0] != n_particles:
            raise ValueError("mass must be a 1D array with shape (n,).")
        if np.any(mass_arr < 0):
            raise ValueError("Particle masses must be non-negative.")

    use_momentum_orientation_flag = False
    if Vxyz is not None:
        Vxyz_arr = np.asarray(Vxyz, dtype=float)
        if Vxyz_arr.ndim != 2 or Vxyz_arr.shape[1] != 3:
            raise ValueError("Vxyz must be a 2D array with shape (n, 3).")
        if Vxyz_arr.shape[0] != n_particles:
            raise ValueError("Vxyz must have the same number of particles as XYZ.")
        if orient_with_momentum:
            use_momentum_orientation_flag = True
    elif orient_with_momentum: # Vxyz is None but orient_with_momentum is True
        if verbose:
            print("Warning: `orient_with_momentum` is True, but `Vxyz` (velocities) not provided. Disabling momentum orientation.")
        # Vxyz_arr will be zeros, use_momentum_orientation_flag remains False
    
    # Ensure Vxyz_arr exists even if not used for momentum, for consistent typing later if accessed
    if Vxyz is None:
        Vxyz_arr = np.zeros_like(XYZ, dtype=float)


    if not (Rmin >= 0 and Rmax > 0 and Rmax > Rmin):
        raise ValueError("Rmin must be >= 0, Rmax > 0, and Rmax > Rmin.")

    XYZ_all = XYZ
    mass_all = mass_arr
    Vxyz_all = Vxyz_arr

    # q_iter, s_iter are b/a, c/a from the *previous* iteration. Start spherical.
    q_iter = 1.0 
    s_iter = 1.0  
    
    # Defines principal axes (columns e_b, e_c, e_a) for Rsphall calculation.
    # Initialized to align with Y, X, Z (b-like, c-like, a-like for initial q,s=1).
    current_transform_cols_bca = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=float).T

    # Stores [b^2, c^2, a^2] from current iteration's structure tensor, used for q_new, s_new.
    eigval_loop_bca = np.array([1.0,1.0,1.0], dtype=float) 

    # Initial particle selection based on spherical shell
    initial_distances_sq_all = _calculate_particle_distances_sq(XYZ_all)
    extract_mask = (initial_distances_sq_all < Rmax**2) & (initial_distances_sq_all >= Rmin**2)

    # --- Iteration loop ---
    for iter_num in range(max_iter):
        if verbose:
            print(f"Iteration {iter_num + 1}/{max_iter}...")

        # 1. Select particles for this iteration
        if iter_num > 0 and reduced_structure:
            # Scale Rmax by volume factor (q*s)^(1/3) to keep enclosed volume ~constant
            # q_iter, s_iter are from the *previous* iteration's shape
            vol_scale_factor_inv = 1.0
            prod_qs_prev = q_iter * s_iter
            if prod_qs_prev > 1e-9: # Avoid issues with zero or tiny q/s
                vol_scale_factor_inv = prod_qs_prev**(1./3.)
            
            Rmax_scaled_aperture = Rmax / vol_scale_factor_inv if vol_scale_factor_inv > 1e-9 else Rmax * 1e9 # Effectively infinity if vol_scale is zero

            # `current_transform_cols_bca` and `q_iter, s_iter` are from the previous iteration
            _, extract_mask, Rsph_extracted = _calculate_Rsphall_and_extract(
                XYZ_all,
                current_transform_cols_bca, 
                q_iter, 
                s_iter, 
                Rmax_scaled_aperture,
                Rmin # Physical Rmin, gets scaled inside helper
            )
            
            if np.sum(extract_mask) < 10: # Need enough particles for stable tensor
                if verbose: print("Too few particles selected in reduced step. Using previous iteration's shape or stopping.")
                break # Exit loop, will use results from previous successful iteration
            
            # Calculate Rsph weights for structure tensor based on extracted particles
            Rsph_values_for_weighting = Rsph_extracted # These are for the currently extracted particles
            if Rsph_values_for_weighting.size > 0 :
                median_Rsph = np.median(Rsph_values_for_weighting)
                if median_Rsph < 1e-9: median_Rsph = 1.0 # Avoid division by zero or tiny median
                Rsph_factors = Rsph_values_for_weighting / median_Rsph
                # Prevent Rsph_factors from being zero if median_Rsph was tiny or Rsph_values_for_weighting had zeros
                Rsph_factors[Rsph_factors < 1e-6] = 1e-6 
                structure_tensor_weights = 1.0 / (Rsph_factors**2)
            else: # Should not happen if sum(extract_mask) >= 10
                num_extracted = np.sum(extract_mask)
                structure_tensor_weights = np.ones(num_extracted if num_extracted > 0 else 0, dtype=float)
        else: # First iteration or not using reduced_structure
            if iter_num == 0 : # Use pre-calculated initial spherical mask
                pass # extract_mask is already set
            else: # Not reduced_structure, but iter_num > 0 (loop should break after 1 iter if not reduced)
                 # This re-selects spherically if fixed_aperture and more than one iter (which it shouldn't do)
                 distances_sq_all = _calculate_particle_distances_sq(XYZ_all)
                 extract_mask = (distances_sq_all < Rmax**2) & (distances_sq_all >= Rmin**2)

            if np.sum(extract_mask) < 10:
                if verbose: print(f"Too few particles ({np.sum(extract_mask)}) initially or in fixed selection. Check Rmin/Rmax.")
                # If initial selection is bad, return NaNs
                nan_abc = np.full(3, np.nan)
                nan_transform = np.full((3,3), np.nan)
                if return_ellip_triax: return nan_abc, nan_transform, np.nan, np.nan
                return nan_abc, nan_transform
            
            # Uniform weighting for first iteration or non-reduced structure
            structure_tensor_weights = np.ones(np.sum(extract_mask), dtype=float)

        # Current particles for this iteration's tensor calculation
        current_XYZ = XYZ_all[extract_mask]
        current_mass = mass_all[extract_mask]
        
        # 2. Compute (weighted) structure tensor
        structure = _compute_weighted_structure_tensor(current_XYZ, current_mass, structure_tensor_weights)

        # 3. Diagonalize structure tensor
        # eigval_std has eigenvalues c_s^2, b_s^2, a_s^2 (ascending order)
        # eigvec_std columns are corresponding eigenvectors e_c, e_b, e_a
        try:
            eigval_std, eigvec_std = linalg.eigh(structure)
        except linalg.LinAlgError:
            if verbose: print("linalg.eigh failed during iteration. Using previous iteration's shape or stopping.")
            break 
        eigval_std = np.maximum(eigval_std, 1e-12) # Ensure eigenvalues are non-negative

        # Principal axis squared lengths (a^2 >= b^2 >= c^2)
        current_a_sq = eigval_std[2] # Corresponds to e_a = eigvec_std[:,2]
        current_b_sq = eigval_std[1] # Corresponds to e_b = eigvec_std[:,1]
        current_c_sq = eigval_std[0] # Corresponds to e_c = eigvec_std[:,0]

        # Principal eigenvectors (columns of eigvec_std are e_c, e_b, e_a)
        e_a_vec = eigvec_std[:, 2].copy()
        e_b_vec = eigvec_std[:, 1].copy()
        e_c_vec = eigvec_std[:, 0].copy()
        
        # 4. Orient axes: Ensure right-handed system (e_a, e_b, e_c)
        # Check determinant of [e_a, e_b, e_c]. If < 0, flip one axis (e.g., e_c).
        temp_evec_mat_abc = np.column_stack((e_a_vec, e_b_vec, e_c_vec))
        if np.linalg.det(temp_evec_mat_abc) < 0:
            e_c_vec *= -1.0 # Flipping e_c to make it right-handed with e_a, e_b

        # Optional: Align minor axis (e_c_vec) with angular momentum
        if use_momentum_orientation_flag and current_XYZ.shape[0] > 0: # Vxyz provided and particles exist
            current_Vxyz_selected = Vxyz_all[extract_mask]
            specific_momenta = np.cross(current_XYZ, current_Vxyz_selected)
            total_momentum_vec = np.sum(current_mass[:, np.newaxis] * specific_momenta, axis=0)

            if np.linalg.norm(total_momentum_vec) > 1e-9: # If there is non-negligible momentum
                # Align e_c_vec with total_momentum_vec
                if np.dot(e_c_vec, total_momentum_vec) < 0:
                    e_c_vec *= -1.0
                
                # Re-orthogonalize to maintain e_a, e_b, e_c as a right-handed system
                # with e_c now fixed by momentum.
                # We can preserve the direction of e_a (original major axis from S) as much as possible.
                e_a_original_dir = eigvec_std[:, 2].copy() # Original e_a from S tensor
                
                # If e_a_original_dir is not (anti)parallel to the new e_c_vec
                if np.abs(np.dot(e_a_original_dir, e_c_vec)) < 0.9999:
                    # e_b = normalize(e_c x e_a_original_dir)
                    e_b_vec = np.cross(e_c_vec, e_a_original_dir)
                    norm_eb = np.linalg.norm(e_b_vec)
                    if norm_eb > 1e-9: e_b_vec /= norm_eb
                    else: # Should not happen if e_a_orig and e_c are not parallel
                          # Fallback: if e_b is zero, choose arbitrary perpendicular
                        if abs(e_c_vec[0]) < 0.9: temp_perp = np.array([1.,0.,0.],dtype=float)
                        else: temp_perp = np.array([0.,1.,0.],dtype=float)
                        e_b_vec = np.cross(e_c_vec, temp_perp)
                        e_b_vec /= (np.linalg.norm(e_b_vec) + 1e-9) # Add epsilon

                    # e_a = normalize(e_b x e_c)
                    e_a_vec = np.cross(e_b_vec, e_c_vec)
                    # e_a_vec should already be unit norm if e_b, e_c are unit and ortho.
                    # For safety, re-normalize, though typically not needed if math is precise.
                    norm_ea = np.linalg.norm(e_a_vec)
                    if norm_ea > 1e-9: e_a_vec /= norm_ea
                    else: # e_b and e_c became parallel: Problem! Fallback.
                        if abs(e_b_vec[0]) < 0.9: temp_perp_ea = np.array([1.,0.,0.],dtype=float)
                        else: temp_perp_ea = np.array([0.,1.,0.],dtype=float)
                        e_a_vec = np.cross(e_b_vec, temp_perp_ea)
                        e_a_vec /= (np.linalg.norm(e_a_vec) + 1e-9)

                else: # e_a_original_dir was (anti)parallel to new e_c_vec (e.g., very prolate along L)
                      # Pick an arbitrary e_b perpendicular to e_c_vec
                    if abs(e_c_vec[0]) < 0.9: temp_perp_vec = np.array([1.,0.,0.],dtype=float)
                    else: temp_perp_vec = np.array([0.,1.,0.],dtype=float)
                    
                    e_b_vec = np.cross(e_c_vec, temp_perp_vec)
                    e_b_vec /= (np.linalg.norm(e_b_vec) + 1e-9) # Add epsilon
                    
                    e_a_vec = np.cross(e_b_vec, e_c_vec)
                    e_a_vec /= (np.linalg.norm(e_a_vec) + 1e-9) # Add epsilon


        # For next iteration's Rsphall calculation (needs e_b, e_c, e_a as columns)
        # and corresponding eigenvalues b^2, c^2, a^2 for q,s definition.
        current_transform_cols_bca = np.column_stack((e_b_vec, e_c_vec, e_a_vec))
        eigval_loop_bca = np.array([current_b_sq, current_c_sq, current_a_sq], dtype=float)

        # 5. Update q, s for the *next* iteration's Rsphall (or for convergence check)
        # q_new = b/a, s_new = c/a based on current iteration's shape.
        # Uses eigval_loop_bca which is [b^2, c^2, a^2]
        if eigval_loop_bca[2] > 1e-12: # If a^2 is not essentially zero
            q_new = np.sqrt(eigval_loop_bca[0] / eigval_loop_bca[2]) # sqrt(b^2/a^2)
            s_new = np.sqrt(eigval_loop_bca[1] / eigval_loop_bca[2]) # sqrt(c^2/a^2)
        else: # Degenerate case (a vanishes), treat as spherical to avoid NaNs
            q_new = 1.0
            s_new = 1.0
        
        # Sanitize q_new, s_new: ensure they are valid ratios (0 < s <= q <= 1)
        q_new = np.nan_to_num(q_new, nan=1.0, posinf=1.0, neginf=1.0) # Handle potential NaNs/Infs
        s_new = np.nan_to_num(s_new, nan=1.0, posinf=1.0, neginf=1.0)
        q_new = np.clip(q_new, 1e-3, 1.0) # b/a is between 0 (needle) and 1 (oblate/sphere)
        s_new = np.clip(s_new, 1e-3, q_new) # c/a is between 0 and q_new (c <= b)

        # 6. Convergence check (compare new q,s with q_iter,s_iter from PREVIOUS step)
        if iter_num > 0:
            # Use q_iter, s_iter (from previous iter) as q_old, s_old
            q_old_safe = q_iter if q_iter > 1e-6 else 1.0 # Avoid division by zero
            s_old_safe = s_iter if s_iter > 1e-6 else 1.0
            
            convergence_metric = np.max(np.array([
                (1.0 - q_new / q_old_safe)**2, 
                (1.0 - s_new / s_old_safe)**2
            ]))

            if convergence_metric < tol**2 : 
                if verbose: print(f"Converged at iteration {iter_num + 1} with metric {convergence_metric:.2e}.")
                break # Exit loop
        
        # Update q_iter, s_iter for the *next* iteration (or for output if loop finishes/breaks)
        q_iter = q_new
        s_iter = s_new

        if verbose:
            # current_a_sq, current_b_sq, current_c_sq are from sorted eigenvalues of S
            print(f"  sqrt(eigvals S): a_s={np.sqrt(current_a_sq):.3f}, b_s={np.sqrt(current_b_sq):.3f}, c_s={np.sqrt(current_c_sq):.3f}")
            print(f"  Current axis ratios for next step: q(b/a) = {q_iter:.4f}, s(c/a) = {s_iter:.4f}")
        
        # If not using reduced structure, only one iteration is needed.
        if not reduced_structure and iter_num == 0:
            if verbose: print("Not using reduced structure. Single pass results.")
            break
    else: # Loop finished due to max_iter
        if verbose: print(f"Reached max_iter ({max_iter}) without full convergence to tol={tol:.1e}.")

    # --- Prepare final results ---
    # eigval_loop_bca from the last successful iteration holds [b_k^2, c_k^2, a_k^2]
    # current_transform_cols_bca columns are [e_b_k, e_c_k, e_a_k]
    if np.any(np.isnan(eigval_loop_bca)) or np.sum(extract_mask) < 3:
        if verbose: print("Warning: Finalizing with potentially invalid eigenvalues or too few particles.")
        nan_abc = np.full(3, np.nan)
        nan_transform = np.full((3,3), np.nan)
        if return_ellip_triax: return nan_abc, nan_transform, np.nan, np.nan
        return nan_abc, nan_transform

    # Axis lengths from the last determined shape (a >= b >= c)
    # Based on eigval_loop_bca = [b^2, c^2, a^2]
    # So a_val = sqrt(eigval_loop_bca[2]), b_val = sqrt(eigval_loop_bca[0]), c_val = sqrt(eigval_loop_bca[1])
    a_val_final = np.sqrt(eigval_loop_bca[2]) if eigval_loop_bca[2] > 0 else 0.0
    b_val_final = np.sqrt(eigval_loop_bca[0]) if eigval_loop_bca[0] > 0 else 0.0
    c_val_final = np.sqrt(eigval_loop_bca[1]) if eigval_loop_bca[1] > 0 else 0.0
    
    # Ensure a >= b >= c order for output and normalize by a
    # The loop's q_iter, s_iter already reflect b/a and c/a from this ordering.
    # For output, ensure a_out >= b_out >= c_out
    # Create pairs of (value, vector) for sorting
    # Vectors are e_a_loop, e_b_loop, e_c_loop from current_transform_cols_bca
    e_a_loop_final = current_transform_cols_bca[:, 2] # Vector for a_val_final
    e_b_loop_final = current_transform_cols_bca[:, 0] # Vector for b_val_final
    e_c_loop_final = current_transform_cols_bca[:, 1] # Vector for c_val_final

    # Store as [(value, vector), ...] then sort by value descending for a, b, c output
    # These are the physical axis lengths before normalization
    axis_data_tuples = [
        (a_val_final, e_a_loop_final),
        (b_val_final, e_b_loop_final),
        (c_val_final, e_c_loop_final)
    ]
    # Sort by axis length, descending to get a, b, c
    axis_data_tuples.sort(key=lambda x: x[0], reverse=True)
    
    a_out = axis_data_tuples[0][0]
    b_out = axis_data_tuples[1][0]
    c_out = axis_data_tuples[2][0]

    e_a_out = axis_data_tuples[0][1]
    e_b_out = axis_data_tuples[1][1]
    e_c_out = axis_data_tuples[2][1]
    
    # Normalize abc so that a=1 (dimensionless ratios)
    if a_out > 1e-9: # Avoid division by zero if a_out is effectively zero
      abc_normalized = np.array([a_out, b_out, c_out])
    else: 
      abc_normalized = np.array([1.0, 1.0, 1.0]) # Fallback for degenerate case (point-like)
      if verbose: print("Warning: Major axis length a_out is near zero. Resulting shape is ill-defined.")
      # Make b/a and c/a also 1 if a is zero, to represent a sphere of zero size.
      # Or could return NaNs: np.full(3, np.nan) if more appropriate.

    # Final transformation matrix: rows are e_a, e_b, e_c for the sorted a,b,c
    final_transform_matrix = np.vstack((e_a_out, e_b_out, e_c_out))

    if not return_ellip_triax:
        return abc_normalized, final_transform_matrix

    # Ellipticity and Triaxiality using a_out, b_out, c_out
    ellip = 0.0
    if a_out > 1e-9: # Avoid division by zero
        ellip = 1.0 - (c_out / a_out)
    ellip = np.nan_to_num(ellip) # Should not be NaN if a_out > 0

    triax = 0.0 # Default for spherical or cases leading to NaN
    a_out_sq, b_out_sq, c_out_sq = a_out**2, b_out**2, c_out**2
    denominator_triax = a_out_sq - c_out_sq
    if denominator_triax > 1e-12: # Avoid division by zero if a approx c
        triax = (a_out_sq - b_out_sq) / denominator_triax
    elif np.abs(a_out_sq - b_out_sq) < 1e-12 : # Denominator is zero, check if numerator is also zero (sphere a=b=c)
        triax = 0.0 # Sphere (a=b=c)
    # else: triax is 0.0 (e.g. oblate where a=b, so num=0, or prolate where b=c, so T=1 if formula applied before check)
    # The (a^2-b^2)/(a^2-c^2) definition:
    # Sphere: a=b=c -> 0/0 (conventionally 0)
    # Oblate: a=b>c -> 0 / (a^2-c^2) -> 0
    # Prolate: a>b=c -> (a^2-c^2) / (a^2-c^2) -> 1
    # The above logic sets T=0 for oblate. For prolate (b=c), if denom!=0, it's (a^2-c^2)/(a^2-c^2)=1.
    if denominator_triax > 1e-12 and np.abs(b_out_sq - c_out_sq) < 1e-12: # Check for prolate (b=c, a>b)
        triax = 1.0

    triax = np.nan_to_num(triax) # Catch any other NaNs, though logic should prevent them.
    return abc_normalized, final_transform_matrix, ellip, triax
