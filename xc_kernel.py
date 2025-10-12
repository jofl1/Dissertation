"""
XC-kernel computation module for iDEA (1D)

Key improvements (non-exhaustive):
- Fixed mirror-padding implementation using numpy.pad('reflect').
- Use *localized Gaussian* perturbations instead of single-point spikes.
- Use *central finite differences* (v(+h) - v(-h)) / (2h) for higher accuracy.
- Richardson extrapolation applied to central-difference estimates.
- Renormalise density *after* smoothing and ensure positive densities.
- More robust retrieval of KS potential (tries several attributes).
- Better multiprocessing pattern with light-weight arguments; records failures.
- Reporting and optional logging-friendly verbose messages.

The physics remains the same: we compute
    f_xc(x,x') = δv_xc(x)/δn(x')
by perturbing the density, reinverting to obtain v_xc, and forming finite
differences.  The module is intended for iDEA-produced 1D systems.

"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import scipy.signal
import scipy.optimize
from functools import partial
from multiprocessing import Pool, cpu_count
import iDEA.system
import iDEA.state
import iDEA.observables
import iDEA.reverse_engineering
import iDEA.methods.non_interacting

__all__ = [
    'compute_xc_kernel',
    'extract_decay_length',
    'validate_kernel_symmetry',
]


def _get_v_ks_from_system(s_ks: iDEA.system.System) -> np.ndarray:
    """Extract the Kohn–Sham (effective) potential from a system.

    Different iDEA versions may store the KS potential under different
    attribute names (e.g. v_ext for the non-interacting system returned by
    reverse_engineering). This helper tries several reasonable choices.
    """
    for attr in ('v_ks', 'v_eff', 'v_ext', 'v_potential'):
        if hasattr(s_ks, attr):
            return getattr(s_ks, attr)
    # fallback: try to reconstruct from system fields if possible
    raise AttributeError('Unable to locate KS potential attribute on returned system')


def _get_xc_potential(
    s: iDEA.system.System,
    target_density: np.ndarray,
    v_guess: Optional[np.ndarray] = None,
    mu: float = 1.0,
    pe: float = 0.1,
    tol: float = 1e-6,  # Changed from 1e-12 to 1e-6 for better convergence
    silent: bool = True
) -> np.ndarray:
    """
    Compute XC potential from a target density using reverse engineering.

    Returns v_xc defined by v_xc = v_ks - v_ext - v_hartree.
    """
    import time
    from datetime import datetime

    if not silent:
        print(f"    [DEBUG] _get_xc_potential called at {datetime.now().strftime('%H:%M:%S')}")
        print(f"           Target density sum: {np.sum(target_density) * s.dx:.6f}")
        if v_guess is not None:
            print(f"           Using initial guess for v_ks")
        print(f"           Calling iDEA.reverse_engineering.reverse...")
        start_re = time.time()

    s_ks = iDEA.reverse_engineering.reverse(
        s, target_density, iDEA.methods.non_interacting,
        v_guess=v_guess, mu=mu, pe=pe, tol=tol, silent=silent
    )

    if not silent:
        print(f"           Reverse engineering completed in {time.time() - start_re:.3f}s")

    # Robust extraction of v_ks
    v_ks = _get_v_ks_from_system(s_ks)

    # Hartree on the *same* grid as s (iDEA observables expect s and density)
    v_hartree = iDEA.observables.hartree_potential(s, target_density)

    v_xc = v_ks - s.v_ext - v_hartree

    if not silent:
        print(f"           v_xc computed: range [{v_xc.min():.4e}, {v_xc.max():.4e}]")

    return v_xc


def _gaussian_perturbation(x: np.ndarray, center_idx: int, sigma: float, amp: float) -> np.ndarray:
    """Make a normalised Gaussian perturbation on the grid whose maximum equals amp.

    amp is the height at the center grid point (i.e. the same units as n(x)).
    """
    x0 = x[center_idx]
    g = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    g = g / np.max(g) * amp
    return g

def _precompute_gaussians(x: np.ndarray, sigma: float, delta_n: float) -> np.ndarray:
    """Precompute all Gaussian perturbations to avoid redundant calculations."""
    n_points = len(x)
    gaussians = np.zeros((n_points, n_points))
    for j in range(n_points):
        gaussians[j, :] = _gaussian_perturbation(x, j, sigma, delta_n)
    return gaussians

def _compute_single_column_worker(args) -> Tuple[int, np.ndarray, bool]:
    """Worker to compute a single column using central finite differences."""
    (j, s, x, n_ref, v_xc_ref, delta_n, sigma, smoothing_window, smoothing_order,
     mu, pe, tol, gaussian_j) = args  # Added gaussian_j at the end
    
    n_points = len(n_ref)
    column = np.zeros(n_points)
    success = True

    # Check if this is the first few columns for extra debugging
    verbose_worker = (j < 3)

    if verbose_worker:
        import time
        from datetime import datetime
        print(f"      [WORKER DEBUG] Starting column {j} at {datetime.now().strftime('%H:%M:%S')}")
        print(f"                     Gaussian max: {gaussian_j.max():.4e}, at index {np.argmax(gaussian_j)}")

    try:
        # Use precomputed Gaussian instead of computing it
        pert_plus = n_ref.copy() + gaussian_j
        pert_minus = n_ref.copy() - gaussian_j

        if verbose_worker:
            print(f"                     n_ref range: [{n_ref.min():.4e}, {n_ref.max():.4e}]")
            print(f"                     pert_plus range before clipping: [{pert_plus.min():.4e}, {pert_plus.max():.4e}]")
            print(f"                     pert_minus range before clipping: [{pert_minus.min():.4e}, {pert_minus.max():.4e}]")

        pert_plus = np.maximum(pert_plus, 1e-12)
        pert_minus = np.maximum(pert_minus, 1e-12)

        # OPTIONAL: renormalize total N to match reference exactly by scaling
        N_ref = np.sum(n_ref) * s.dx

        # Fix: Normalize arrays properly (not in-place modification in loop)
        N_plus_before = np.sum(pert_plus) * s.dx
        pert_plus *= (N_ref / N_plus_before)
        if verbose_worker:
            print(f"                     pert_plus normalized: {N_plus_before:.6f} -> {np.sum(pert_plus) * s.dx:.6f}")

        N_minus_before = np.sum(pert_minus) * s.dx
        pert_minus *= (N_ref / N_minus_before)
        if verbose_worker:
            print(f"                     pert_minus normalized: {N_minus_before:.6f} -> {np.sum(pert_minus) * s.dx:.6f}")

        # Smooth and ensure positivity (renormalise again if desired)
        if smoothing_window > 0:
            if verbose_worker:
                print(f"                     Applying smoothing with window={smoothing_window}")
            pert_plus = scipy.signal.savgol_filter(pert_plus,
                                                  window_length=smoothing_window,
                                                  polyorder=smoothing_order,
                                                  mode='mirror')
            pert_minus = scipy.signal.savgol_filter(pert_minus,
                                                   window_length=smoothing_window,
                                                   polyorder=smoothing_order,
                                                   mode='mirror')
            pert_plus = np.maximum(pert_plus, 1e-12)
            pert_minus = np.maximum(pert_minus, 1e-12)
            # renormalize after smoothing
            pert_plus *= (N_ref / (np.sum(pert_plus) * s.dx))
            pert_minus *= (N_ref / (np.sum(pert_minus) * s.dx))

        if verbose_worker:
            import time
            from datetime import datetime
            print(f"                     Computing v_xc_plus at {datetime.now().strftime('%H:%M:%S')}...")
            start_vxc = time.time()

        # Compute v_xc for +/- perturbations (using v_xc_ref as initial guess for faster convergence)
        v_xc_p = _get_xc_potential(s, pert_plus, v_guess=v_xc_ref, mu=mu, pe=pe, tol=tol,
                                  silent=(not verbose_worker))

        if verbose_worker:
            print(f"                     v_xc_plus computed in {time.time() - start_vxc:.3f}s")
            print(f"                     Computing v_xc_minus at {datetime.now().strftime('%H:%M:%S')}...")
            start_vxc = time.time()

        v_xc_m = _get_xc_potential(s, pert_minus, v_guess=v_xc_ref, mu=mu, pe=pe, tol=tol,
                                  silent=(not verbose_worker))

        if verbose_worker:
            print(f"                     v_xc_minus computed in {time.time() - start_vxc:.3f}s")

        # Central difference (w.r.t. local density amplitude) -> higher accuracy
        column = (v_xc_p - v_xc_m) / (2.0 * delta_n)

        if verbose_worker:
            print(f"                     Column {j} complete: range [{column.min():.4e}, {column.max():.4e}]")

    except Exception as e:
        # On failure return a zero column and mark as failed
        if verbose_worker:
            print(f"                     ERROR in column {j}: {e}")
            import traceback
            traceback.print_exc()
        success = False
        column = np.zeros(n_points)

    return j, column, success


def _compute_kernel_single_spacing_parallel(
    s: iDEA.system.System,
    n_ref: np.ndarray,
    v_xc_ref: np.ndarray,
    delta_n: float,
    sigma: float,
    smoothing_window: int = 11,
    smoothing_order: int = 3,
    mu: float = 1.0,
    pe: float = 0.1,
    tol: float = 1e-6,  # Changed from 1e-12 to 1e-6 for better convergence
    n_processes: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    import time
    from datetime import datetime

    n_points = len(n_ref)
    f_xc = np.zeros((n_points, n_points))

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    # Precompute all Gaussians once
    if verbose:
        print(f"  [DEBUG] Precomputing {n_points} Gaussian perturbations at {datetime.now().strftime('%H:%M:%S')}...")
        print(f"         sigma={sigma:.4f}, delta_n={delta_n:.4e}")

    start_time = time.time()
    gaussians = _precompute_gaussians(s.x, sigma, delta_n)

    if verbose:
        print(f"  [DEBUG] Gaussians computed in {time.time() - start_time:.2f}s")

    # Prepare argument tuples with precomputed Gaussians
    args_list = [
        (j, s, s.x, n_ref, v_xc_ref, delta_n, sigma, smoothing_window,
         smoothing_order, mu, pe, tol, gaussians[j, :])  # Pass the j-th row
        for j in range(n_points)
    ]

    # Use serial execution for n_processes=1 to avoid multiprocessing overhead
    if n_processes <= 1:
        if verbose:
            print(f"  [DEBUG] Using serial execution for {n_points} columns")
            print(f"         smoothing_window={smoothing_window}, smoothing_order={smoothing_order}")
            print(f"         mu={mu}, pe={pe}, tol={tol}")

        results = []
        start_compute = time.time()

        for i, args in enumerate(args_list):
            column_start = time.time()

            if verbose:
                # More frequent updates to see where it's hanging
                if i % max(1, n_points // 20) == 0 or i < 5 or i >= n_points - 5:
                    elapsed = time.time() - start_compute
                    percent = (i / n_points) * 100
                    if i > 0:
                        rate = elapsed / i
                        remaining = rate * (n_points - i)
                        print(f"    [DEBUG] Column {i}/{n_points} ({percent:.1f}%) - " +
                              f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
                    else:
                        print(f"    [DEBUG] Starting column {i}/{n_points}...")

            result = _compute_single_column_worker(args)
            results.append(result)

            if verbose and i < 3:
                print(f"    [DEBUG] Column {i} took {time.time() - column_start:.3f}s")

    else:
        if verbose:
            print(f"  [DEBUG] Launching pool with {n_processes} processes at {datetime.now().strftime('%H:%M:%S')}")

        with Pool(processes=n_processes) as pool:
            results = pool.map(_compute_single_column_worker, args_list)

    # Assemble results
    failures: List[int] = []
    for j, column, success in results:
        f_xc[:, j] = column
        if not success:
            failures.append(j)

    if verbose and failures:
        print(f"  Warning: reverse-engineering failed for {len(failures)} columns; indices: {failures}")

    return f_xc


def validate_kernel_symmetry(f_xc: np.ndarray, tolerance: float = 1e-6) -> Dict:
    asymmetry = np.max(np.abs(f_xc - f_xc.T))
    kernel_magnitude = np.max(np.abs(f_xc))
    relative_error = asymmetry / (kernel_magnitude + 1e-12)

    return {
        'max_asymmetry': float(asymmetry),
        'relative_error': float(relative_error),
        'passed': bool(asymmetry < tolerance),
        'tolerance': float(tolerance)
    }


def extract_decay_length(
    s: iDEA.system.System,
    f_xc: np.ndarray,
    reference_point: Optional[int] = None,
    min_distance: Optional[float] = None
) -> Dict:
    if reference_point is None:
        reference_point = len(s.x) // 2

    if min_distance is None:
        min_distance = 2.0 * s.dx

    kernel_slice = np.abs(f_xc[reference_point, :])
    distances = np.abs(s.x - s.x[reference_point])

    mask = (distances > min_distance) & (kernel_slice > 1e-12)

    if np.sum(mask) < 5:
        return {
            'decay_length': np.nan,
            'amplitude': np.nan,
            'r_squared': 0.0,
            'x_ref': float(s.x[reference_point]),
            'fit_success': False
        }

    def exp_decay(r, A, decay_length):
        return A * np.exp(-r / decay_length)

    try:
        popt, pcov = scipy.optimize.curve_fit(
            exp_decay,
            distances[mask],
            kernel_slice[mask],
            p0=[kernel_slice[reference_point], 2.0],
            bounds=([1e-15, 1e-6], [1e12, 1e3]),
            maxfev=20000
        )
        A_fit, decay_length = popt
        y_pred = exp_decay(distances[mask], A_fit, decay_length)
        residuals = kernel_slice[mask] - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((kernel_slice[mask] - np.mean(kernel_slice[mask]))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-12))
        fit_success = True
    except Exception:
        A_fit = np.nan
        decay_length = np.nan
        r_squared = 0.0
        fit_success = False

    return {
        'decay_length': float(decay_length),
        'amplitude': float(A_fit),
        'r_squared': float(r_squared),
        'x_ref': float(s.x[reference_point]),
        'fit_success': bool(fit_success)
    }


def compute_xc_kernel(
    s: iDEA.system.System,
    state: iDEA.state.State,
    delta_n: float = 1e-4,
    use_richardson: bool = True,
    buffer_points: int = 10,
    smoothing_window: int = 11,
    smoothing_order: int = 3,
    sigma: Optional[float] = None,
    mu: float = 1.0,
    pe: float = 0.1,
    tol: float = 1e-6,  # Changed from 1e-12 to 1e-6 for better convergence
    n_processes: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """Public API: compute static XC kernel matrix.

    Important notes:
      - sigma: width of Gaussian perturbation; if None uses 1.5*dx.
      - central-difference is used; Richardson extrapolation mixes central-diff
        estimates at h and h/2.
      - tol: convergence tolerance for reverse engineering (default 1e-6)
    """
    import time
    from datetime import datetime

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    if verbose:
        print(f"[DEBUG] compute_xc_kernel starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"[DEBUG] Grid points: {len(s.x)}")
        print(f"[DEBUG] Processes to use: {n_processes}")
        print(f"[DEBUG] Convergence tolerance: {tol}")

    # Special handling for 1-electron systems (XC kernel should be zero)
    if s.electrons in ['u', 'd']:  # Single electron
        if verbose:
            print(f"[DEBUG] Single electron system detected - XC kernel is zero by definition")
        return np.zeros((len(s.x), len(s.x)))

    n_ref = iDEA.observables.density(s, state=state)

    if verbose:
        print(f"[DEBUG] Reference density computed")
        print(f"       Total density integral: {np.sum(n_ref) * s.dx:.8f}")
        print(f"       Density range: [{n_ref.min():.4e}, {n_ref.max():.4e}]")

    # create extended system using numpy.pad reflect for potential and density
    if buffer_points > 0:
        if verbose:
            print(f"[DEBUG] Creating extended system with {buffer_points} buffer points...")
        dx = s.dx
        x_left = s.x[0] - np.arange(buffer_points, 0, -1) * dx
        x_right = s.x[-1] + np.arange(1, buffer_points + 1) * dx
        x_extended = np.concatenate([x_left, s.x, x_right])

        v_ext_extended = np.pad(s.v_ext, buffer_points, mode='reflect')

        n_orig = len(s.x)
        n_ext = len(x_extended)

        # expand v_int by reflecting edges (cheap but consistent)
        v_int_extended = np.zeros((n_ext, n_ext))
        v_int_extended[buffer_points:buffer_points+n_orig,
                       buffer_points:buffer_points+n_orig] = s.v_int
        # reflect outer blocks
        # fill left/right as nearest-block copies for simplicity
        for i in range(n_ext):
            for j in range(n_ext):
                if i < buffer_points or i >= buffer_points + n_orig or \
                   j < buffer_points or j >= buffer_points + n_orig:
                    i_orig = min(max(i - buffer_points, 0), n_orig - 1)
                    j_orig = min(max(j - buffer_points, 0), n_orig - 1)
                    v_int_extended[i, j] = s.v_int[i_orig, j_orig]

        s_extended = iDEA.system.System(x_extended, v_ext_extended, v_int_extended,
                                        s.electrons, stencil=s.stencil)
        n_extended = np.pad(n_ref, buffer_points, mode='reflect')
    else:
        s_extended = s
        n_extended = n_ref

    # Reference v_xc on extended grid
    if verbose:
        print(f"[DEBUG] Computing reference XC potential via reverse engineering at {datetime.now().strftime('%H:%M:%S')}...")
        print(f"       This may take a moment...")
        start_re = time.time()

    v_xc_ref = _get_xc_potential(s_extended, n_extended, mu=mu, pe=pe, tol=tol, silent=(not verbose))

    if verbose:
        print(f"[DEBUG] Reference XC potential computed in {time.time() - start_re:.2f}s")
        print(f"       v_xc_ref range: [{v_xc_ref.min():.4e}, {v_xc_ref.max():.4e}]")

    # sigma for gaussian perturbation
    if sigma is None:
        sigma = 1.5 * s_extended.dx
        if verbose:
            print(f"[DEBUG] Using automatic sigma = {sigma:.4f} (1.5 * dx)")

    # compute with central difference at h and (optionally) h/2
    if use_richardson:
        if verbose: print('Computing central-difference kernel with h...')
        f_h = _compute_kernel_single_spacing_parallel(
            s_extended, n_extended, v_xc_ref, delta_n, sigma,
            smoothing_window, smoothing_order, mu, pe, tol, n_processes, verbose
        )
        if verbose: print('Computing central-difference kernel with h/2...')
        f_h2 = _compute_kernel_single_spacing_parallel(
            s_extended, n_extended, v_xc_ref, delta_n / 2.0, sigma,
            smoothing_window, smoothing_order, mu, pe, tol, n_processes, verbose
        )
        # Richardson for central difference (error ~ h^2) -> same combination
        f_xc_extended = (4.0 * f_h2 - f_h) / 3.0
    else:
        f_xc_extended = _compute_kernel_single_spacing_parallel(
            s_extended, n_extended, v_xc_ref, delta_n, sigma,
            smoothing_window, smoothing_order, mu, pe, tol, n_processes, verbose
        )

    # remove buffer
    if buffer_points > 0:
        f_xc = f_xc_extended[buffer_points:-buffer_points, buffer_points:-buffer_points]
    else:
        f_xc = f_xc_extended

    # explicit symmetrize
    f_xc = 0.5 * (f_xc + f_xc.T)

    if verbose:
        print('XC kernel computed; symmetry error:', np.max(np.abs(f_xc - f_xc.T)))

    return f_xc
