# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import copy  # For deepcopy during resampling
import time  # For timing
import gc  # Garbage collector interface
import matplotlib.pyplot as plt

# --- Constants ---
# Define possible steps (N, E, S, W) as (dx, dy) tuples
MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


# --- Symmetry Operations for Square Lattice (D4 Group) ---
# Functions take a relative coordinate (dx, dy) and return transformed (dx', dy')
def op_identity(dx, dy):
    return (dx, dy)


def op_rot90(dx, dy):
    return (-dy, dx)


def op_rot180(dx, dy):
    return (-dx, -dy)


def op_rot270(dx, dy):
    return (dy, -dx)


def op_reflect_x(dx, dy):
    return (dx, -dy)


def op_reflect_y(dx, dy):
    return (-dx, dy)


def op_reflect_y_eq_x(dx, dy):
    return (dy, dx)


def op_reflect_y_eq_negx(dx, dy):
    return (-dy, -dx)


# List of non-identity symmetry operations for the Pivot Algorithm
SYMMETRIES = [
    op_rot90,
    op_rot180,
    op_rot270,
    op_reflect_x,
    op_reflect_y,
    op_reflect_y_eq_x,
    op_reflect_y_eq_negx,
]


# --- Particle Class (Stores full path for plotting and pivoting) ---
class Particle:
    """Represents a single self-avoiding walk particle state.
    Stores the full path for plotting and pivoting.
    """

    def __init__(self, initial_pos=(0, 0)):
        self.positions = [initial_pos]  # List of (x, y) tuples
        self.visited = {initial_pos}  # Set for O(1) lookup
        self.length = 0

    def get_endpoint(self):
        """Returns the last position (x, y) of the walk"""
        return self.positions[-1]

    def get_valid_moves(self):
        """Returns a list of valid next positions (tuples)"""
        cx, cy = self.get_endpoint()
        valid_next_pos = []
        for dx, dy in MOVES:
            next_pos = (cx + dx, cy + dy)
            if next_pos not in self.visited:
                valid_next_pos.append(next_pos)
        return valid_next_pos

    def extend(self, next_pos):
        """Creates a new particle extended by one step.
        Copies positions list and visited set.
        """
        # Create a new instance using __new__ and manually initialize
        # Might be slightly faster than calling __init__ again
        new_particle = self.__class__.__new__(self.__class__)
        # Copy positions list and append
        new_particle.positions = self.positions + [next_pos]
        # Copy visited set and add
        new_particle.visited = self.visited.copy()
        new_particle.visited.add(next_pos)
        new_particle.length = self.length + 1
        return new_particle

    def __repr__(self):
        """String representation"""
        return f"L={self.length} @ {self.get_endpoint()}"


# --- Pivot Move Function ---
def apply_pivot_move_inplace(particle):
    """
    Applies one attempt of the Pivot Algorithm to the particle IN PLACE.
    Modifies particle.positions and particle.visited if accepted.

    Args:
        particle (Particle): The particle object to modify (must store positions).
    """
    if particle.length < 2:  # Cannot pivot walks shorter than length 2
        return  # No change

    L = particle.length  # Number of steps (positions = L+1)

    # 1. Choose a random pivot index k (1 <= k <= L-1) -> index in positions list
    # This means the pivot point is positions[k]
    k = random.randint(1, L - 1)

    # 2. Choose a random non-identity symmetry operation
    S = random.choice(SYMMETRIES)

    # 3. Separate head and tail based on index k
    pivot_coord = particle.positions[k]
    head_positions = particle.positions[0 : k + 1]  # Includes pivot point
    tail_positions_orig = particle.positions[k + 1 :]

    # 4. Apply transformation to the tail relative to the pivot point
    new_tail_positions = []
    px, py = pivot_coord
    for j in range(len(tail_positions_orig)):
        ox, oy = tail_positions_orig[j]  # Original absolute position
        dx, dy = ox - px, oy - py  # Vector relative to pivot
        ndx, ndy = S(dx, dy)  # Apply symmetry to relative vector
        new_tail_positions.append((px + ndx, py + ndy))  # New absolute position

    # 5. Check for self-intersection
    #    a) Check if the new tail intersects itself
    new_tail_set = set(new_tail_positions)
    if len(new_tail_set) != len(new_tail_positions):
        # print("Pivot rejected: New tail intersects itself") # DEBUG
        return  # Reject

    #    b) Check if the new tail intersects the head (excluding the pivot point itself)
    head_set_no_pivot = set(head_positions[:-1])  # Exclude particle.positions[k]
    intersection_found = False
    for pos in new_tail_set:
        if pos in head_set_no_pivot:
            intersection_found = True
            break
    if intersection_found:
        # print(f"Pivot rejected: New tail intersects head (k={k})") # DEBUG
        return  # Reject

    # 6. If checks pass, accept the move and update the particle IN PLACE
    # print(f"Pivot accepted: k={k}, S={S.__name__}") # DEBUG
    particle.positions = head_positions + new_tail_positions
    # Reconstruct the visited set efficiently from the new full path
    particle.visited = set(particle.positions)
    # Length remains the same


# --- Main SMC Function with Pivoting ---
def run_smc_saw_perf(
    N_particles, L_max, resample_threshold=0.5, pivot_steps=0, verbose=True
):
    """
    Runs the SMC simulation for SAWs with optional Pivot Algorithm steps.
    Uses Particle class that stores full path.

    Args:
        N_particles (int): Number of particles.
        L_max (int): Maximum length.
        resample_threshold (float): Relative ESS threshold.
        pivot_steps (int): Number of pivot attempts per particle after resampling.
        verbose (bool): Print progress.

    Returns:
        tuple: (log_c_L_estimates, ratio_estimates, final_particles, ess_history)
    """
    if verbose:
        print(
            f"Starting SMC: N={N_particles}, L_max={L_max}, Resample Threshold={resample_threshold}, Pivot Steps={pivot_steps}"
        )

    # Initialization L=0
    particles = [Particle() for _ in range(N_particles)]
    log_c_L_estimates = [0.0]
    ratio_estimates = []
    ess_history = []
    current_log_c = 0.0
    normalized_weights = np.full(N_particles, 1.0 / N_particles, dtype=np.float64)

    start_time = time.time()
    last_print_time = start_time
    time_spent_extending = 0
    time_spent_resampling = 0
    time_spent_pivoting = 0  # Track pivot time

    for L in range(1, L_max + 1):
        time_loop_start = time.time()

        propagated_particles = (
            []
        )  # Stores successfully propagated particles for this step
        unnormalized_weight_contributions = np.zeros(N_particles, dtype=np.float64)

        # --- Propagation & Weighting ---
        time_extend_start = time.time()
        for i in range(N_particles):
            particle = particles[i]  # Get particle from *previous* step
            valid_moves = particle.get_valid_moves()
            k_L_i = len(valid_moves)

            # Weight contribution based on previous weight and choices available
            weight_contribution = normalized_weights[i] * k_L_i
            unnormalized_weight_contributions[i] = weight_contribution

            if k_L_i > 0:
                next_pos = random.choice(valid_moves)
                extended_particle = particle.extend(next_pos)  # Creates new particle
                propagated_particles.append(extended_particle)
        time_spent_extending += time.time() - time_extend_start

        # --- Ratio & Count Estimation ---
        Z_L = unnormalized_weight_contributions.sum()

        if Z_L < 1e-300:  # Handle simulation death
            print(
                f"\nERROR: Z_L ({Z_L}) is effectively zero at L={L}. Simulation failed."
            )
            num_remaining = L_max - L + 1
            log_c_L_estimates.extend([-np.inf] * num_remaining)
            ratio_estimates.extend([0.0] * num_remaining)
            ess_history.extend([0.0] * num_remaining)
            return log_c_L_estimates, ratio_estimates, [], ess_history

        r_L_estimate = Z_L
        ratio_estimates.append(r_L_estimate)
        try:
            current_log_c += math.log(r_L_estimate)
        except ValueError:
            print(
                f"\nERROR: math.log domain error for r_L={r_L_estimate} at L={L}. Simulation failed."
            )
            num_remaining = L_max - L + 1
            log_c_L_estimates.extend([-np.inf] * num_remaining)
            ratio_estimates.extend([0.0] * num_remaining)
            ess_history.extend([0.0] * num_remaining)
            return log_c_L_estimates, ratio_estimates, [], ess_history
        log_c_L_estimates.append(current_log_c)

        # --- ESS Calculation ---
        num_live_particles = len(propagated_particles)
        if num_live_particles == 0:  # Should be caught by Z_L check
            print(f"\nERROR: No live particles found at L={L} despite Z_L > 0.")
            num_remaining = L_max - L + 1
            log_c_L_estimates.extend([-np.inf] * num_remaining)
            ratio_estimates.extend([0.0] * num_remaining)
            ess_history.extend([0.0] * num_remaining)
            return log_c_L_estimates, ratio_estimates, [], ess_history

        # Get weights corresponding to the live particles
        live_particle_weights = unnormalized_weight_contributions[
            unnormalized_weight_contributions > 1e-300
        ]
        # Normalize weights of live particles
        normalized_live_weights = live_particle_weights / Z_L

        # Calculate ESS (handle possibility of zero variance -> division by zero)
        sum_sq_weights = np.sum(normalized_live_weights**2)
        ess = (
            1.0 / sum_sq_weights if sum_sq_weights > 1e-300 else float("inf")
        )  # Treat 0 var as infinite ESS
        # Cap ESS at N_particles (can happen due to floating point)
        ess = min(ess, N_particles)
        ess_history.append(ess)

        # --- Decision: Resample or Keep/Fill ---
        time_resample_start = time.time()
        resampled_this_step = False
        apply_pivots = False  # Flag to trigger pivoting

        if L == L_max:
            # Last step: Keep survivors as they are
            particles_next_L = propagated_particles
            weights_next_L = normalized_live_weights  # Not used, but consistent
        elif ess < N_particles * resample_threshold or num_live_particles < N_particles:
            # Resample to N particles (either due to low ESS or particle death)
            resampled_this_step = True
            apply_pivots = True  # Apply pivots only if we resampled
            # Use indices corresponding to normalized_live_weights
            indices = np.random.choice(
                num_live_particles,
                size=N_particles,
                replace=True,
                p=normalized_live_weights,
            )
            # Use deepcopy for particles that will be pivoted to ensure independence
            particles_next_L = [copy.deepcopy(propagated_particles[i]) for i in indices]
            weights_next_L = np.full(N_particles, 1.0 / N_particles, dtype=np.float64)
        else:
            # ESS is high, N particles survived: Keep all
            particles_next_L = propagated_particles
            weights_next_L = normalized_live_weights

        time_spent_resampling += time.time() - time_resample_start

        # --- PIVOT REJUVENATION STEP ---
        time_pivot_start = time.time()
        if apply_pivots and pivot_steps > 0 and L >= 2:  # Check L>=2 for pivoting
            pivot_count_this_step = 0
            for i in range(N_particles):  # Apply to each particle in the new list
                for _ in range(pivot_steps):
                    apply_pivot_move_inplace(particles_next_L[i])
                    pivot_count_this_step += 1
            # if verbose: print(f"DEBUG: Applied {pivot_count_this_step} pivot attempts at L={L}")
        time_spent_pivoting += time.time() - time_pivot_start

        # --- Update for Next Iteration ---
        if L < L_max:
            particles = particles_next_L
            normalized_weights = weights_next_L
            # Hint to garbage collector (optional)
            if L % 100 == 0:
                gc.collect()
        else:  # L == L_max
            # The final set of particles is particles_next_L
            particles = particles_next_L  # Assign for the return value

        # --- Progress Reporting ---
        current_time = time.time()
        time_this_loop = current_time - time_loop_start
        if verbose and (
            current_time - last_print_time > 10 or L % 10 == 0 or L == 1 or L == L_max
        ):
            eta_seconds = (L_max - L) * (current_time - start_time) / L if L > 0 else 0
            eta_str = (
                f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                if L > 0
                else "ETA: Calculating..."
            )
            mem_usage = gc.get_count()  # Basic indicator
            resample_msg = "YES" if resampled_this_step else "NO"
            pivot_msg = "YES" if apply_pivots and pivot_steps > 0 and L >= 2 else "NO"

            print(
                f"L={L}/{L_max} | "
                f"r_L={r_L_estimate:.4f} | "
                f"logC={current_log_c:.4f} | "
                f"Live={num_live_particles}/{N_particles} | "
                f"ESS={ess:.1f} | "
                f"Resamp={resample_msg} | "
                f"Pivot={pivot_msg} | "
                f"T_loop={time_this_loop:.3f}s | "
                f"GC={mem_usage} | "
                f"{eta_str}"
            )
            last_print_time = current_time

    # End of simulation loop
    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"\nSimulation finished in {total_time:.2f} seconds.")
        print(f"Avg time per step: {total_time / L_max:.3f} seconds.")
        print(f"Total time extending: {time_spent_extending:.2f}s")
        print(f"Total time resampling: {time_spent_resampling:.2f}s")
        print(f"Total time pivoting: {time_spent_pivoting:.2f}s")

    # Return the final set of particles from the last step
    return log_c_L_estimates, ratio_estimates, particles, ess_history


# --- Function to Estimate Mu ---
def estimate_mu_from_smc(log_c_L_estimates, L_max, regression_range_factor=0.5):
    """Estimates mu using different methods from SMC results.
    (Robust version handling inf/nan/overflow)
    """
    # ... (Implementation from previous answers - check for inf/nan/overflow) ...
    num_estimates = len(log_c_L_estimates)
    actual_L_max = num_estimates - 1

    if actual_L_max < 1:
        print("Warning: Insufficient data to estimate mu (L_max < 1 reached).")
        return None, None, None

    mu_from_last_ratio = None
    if actual_L_max >= 1:
        log_c_curr = log_c_L_estimates[actual_L_max]
        log_c_prev = log_c_L_estimates[actual_L_max - 1]
        if np.isfinite(log_c_curr) and np.isfinite(log_c_prev):
            try:
                log_ratio = log_c_curr - log_c_prev
                if log_ratio < 700:
                    mu_from_last_ratio = math.exp(log_ratio)
                else:
                    mu_from_last_ratio = float("inf")
            except OverflowError:
                mu_from_last_ratio = None

    mu_from_last_count = None
    if actual_L_max >= 1:
        log_c_curr = log_c_L_estimates[actual_L_max]
        if np.isfinite(log_c_curr):
            try:
                log_c_div_L = log_c_curr / actual_L_max
                if log_c_div_L < 700:
                    mu_from_last_count = math.exp(log_c_div_L)
                else:
                    mu_from_last_count = float("inf")
            except (OverflowError, ZeroDivisionError):
                mu_from_last_count = None

    mu_from_regression = None
    start_L = max(2, int(actual_L_max * (1 - regression_range_factor)))
    end_L = actual_L_max
    if end_L >= start_L + 2:
        L_vals_potential = np.arange(start_L, end_L + 1)
        valid_indices = [
            L
            for L in L_vals_potential
            if L < len(log_c_L_estimates) and np.isfinite(log_c_L_estimates[L])
        ]
        if len(valid_indices) >= 3:
            L_vals_valid = np.array(valid_indices)
            y = np.array([log_c_L_estimates[L] / L for L in L_vals_valid])
            valid_y_mask = np.isfinite(y)
            if valid_y_mask.sum() >= 3:
                L_vals_final = L_vals_valid[valid_y_mask]
                y_final = y[valid_y_mask]
                x1 = 1.0 / L_vals_final
                x2 = np.log(L_vals_final) / L_vals_final
                A = np.vstack([np.ones(len(L_vals_final)), x1, x2]).T
                try:
                    beta, _, _, _ = np.linalg.lstsq(A, y_final, rcond=None)
                    log_mu_regr = beta[0]
                    if np.isfinite(log_mu_regr):
                        if log_mu_regr < 700:
                            mu_from_regression = math.exp(log_mu_regr)
                        else:
                            mu_from_regression = float("inf")
                    else:
                        print("Warning: Regression resulted in non-finite log(mu).")
                except np.linalg.LinAlgError:
                    print("Warning: Linear regression (linalg.lstsq) failed.")
            else:
                print(
                    f"Warning: Not enough finite y=log(c_L)/L values ({valid_y_mask.sum()}) for regression."
                )
        else:
            print(
                f"Warning: Not enough valid data points ({len(valid_indices)}) for regression."
            )
    else:
        print(f"Warning: Regression range {start_L}-{end_L} too small.")

    print("\n--- Mu Estimates ---")
    print(
        f"From last ratio (r_{actual_L_max}): {mu_from_last_ratio:.8f}"
        if mu_from_last_ratio is not None
        else "From last ratio: N/A"
    )
    print(
        f"From last count (c_{actual_L_max}^(1/{actual_L_max})): {mu_from_last_count:.8f}"
        if mu_from_last_count is not None
        else "From last count: N/A"
    )
    print(
        f"From regression ({start_L}-{end_L}): {mu_from_regression:.8f}"
        if mu_from_regression is not None
        else "From regression: N/A"
    )

    return mu_from_last_ratio, mu_from_last_count, mu_from_regression


# --- Function to Plot Walks ---
def plot_saw(particle, title="Self-Avoiding Walk"):
    """Plots a single SAW using matplotlib."""
    # ... (Implementation from previous answer - uses particle.positions) ...
    if not particle or not hasattr(particle, "positions") or not particle.positions:
        print("Cannot plot invalid or empty particle.")
        return
    positions = np.array(particle.positions)
    if positions.shape[0] < 2:
        # print(f"Cannot plot walk of length {positions.shape[0]-1}.") # Too verbose
        return
    plt.figure(figsize=(6, 6))
    plt.plot(
        positions[:, 0],
        positions[:, 1],
        marker=".",
        linestyle="-",
        markersize=4,
        linewidth=1,
        label=f"Length {particle.length}",
    )
    plt.plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="Start (0,0)")
    plt.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=8, label="End")
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()


# === Main Execution Block ===

# --- Simulation Parameters ---
N_PARTICLES = 2000  # Number of particles
L_MAX = 300  # Maximum walk length
RESAMPLE_THRESH = 0.5  # Resample if ESS < N * threshold
PIVOT_STEPS = 5  # Number of pivot attempts per particle after resampling (0 to disable)

# --- Run Simulation ---
log_c_L, ratios_L, final_particles, ess_history = run_smc_saw_perf(
    N_PARTICLES, L_MAX, RESAMPLE_THRESH, PIVOT_STEPS, verbose=True
)

# --- Estimate Mu ---
actual_L_max_reached = len(log_c_L) - 1
if actual_L_max_reached >= 1:
    mu_est_ratio, mu_est_count, mu_est_regr = estimate_mu_from_smc(
        log_c_L, actual_L_max_reached, regression_range_factor=0.5
    )
    if mu_est_regr is not None:
        print(f"\n==> Final estimate for mu (using regression): {mu_est_regr:.8f}")
    else:
        print("\n==> Final estimate for mu (using regression): N/A")
else:
    print("Simulation ended too early (L<1) to estimate mu.")


# --- Visualize a few final walks ---
num_walks_to_plot = min(3, len(final_particles))
print(f"\nPlotting {num_walks_to_plot} final walks...")
if final_particles:
    for i in range(num_walks_to_plot):
        try:
            plot_saw(
                final_particles[i],
                title=f"Final SAW #{i+1} (L={final_particles[i].length})",
            )
        except AttributeError as e:
            print(f"Skipping plot for particle {i}: Error accessing attributes ({e}).")
        except Exception as e:
            print(f"Error plotting walk {i}: {e}")
else:
    print("No final particles generated to plot.")


# --- Plot Simulation Results (Separate Figures) ---
print("\nPlotting simulation results in separate figures...")
if actual_L_max_reached >= 1:
    L_values = np.arange(actual_L_max_reached + 1)  # 0 to L_max
    L_values_from_1 = np.arange(1, actual_L_max_reached + 1)  # 1 to L_max

    # === Figure 1: Plot log(c_L) vs L ===
    plt.figure(figsize=(8, 6))
    # ... (Plotting code from previous answer) ...
    valid_log_c = np.array(log_c_L)
    valid_mask = np.isfinite(valid_log_c)
    plt.plot(
        L_values[valid_mask],
        valid_log_c[valid_mask],
        marker=".",
        linestyle="-",
        markersize=4,
    )
    plt.title("Logarithm of Estimated Count $\log(\hat{c}_L)$ vs L")
    plt.xlabel("Length L")
    plt.ylabel("$\log(\hat{c}_L)$")
    plt.grid(True)
    plt.tight_layout()

    # === Figure 2: Plot mu estimates vs L ===
    plt.figure(figsize=(8, 6))
    # ... (Plotting code from previous answer, including mu_known line) ...
    log_c_arr = np.array(log_c_L[1:])
    valid_mask_count = np.isfinite(log_c_arr) & (L_values_from_1 > 0)
    mu_est_count = np.full_like(log_c_arr, np.nan, dtype=float)
    if valid_mask_count.any():
        log_c_div_L = log_c_arr[valid_mask_count] / L_values_from_1[valid_mask_count]
        safe_mask = log_c_div_L < 700
        mu_est_count[valid_mask_count] = np.where(
            safe_mask, np.exp(log_c_div_L[safe_mask]), np.inf
        )
    valid_mask_ratio = np.isfinite(log_c_L[1:]) & np.isfinite(log_c_L[:-1])
    mu_est_ratio = np.full_like(log_c_arr, np.nan, dtype=float)
    if valid_mask_ratio.any():
        log_ratio = (
            log_c_arr[valid_mask_ratio] - np.array(log_c_L[:-1])[valid_mask_ratio]
        )
        safe_mask = log_ratio < 700
        mu_est_ratio[valid_mask_ratio] = np.where(
            safe_mask, np.exp(log_ratio[safe_mask]), np.inf
        )
    plt.plot(
        L_values_from_1,
        mu_est_count,
        label="$\hat{\mu}_L = \exp(\log(\hat{c}_L)/L)$",
        alpha=0.8,
        markersize=3,
        marker=".",
    )
    plt.plot(
        L_values_from_1,
        mu_est_ratio,
        label="$\hat{\mu}_L = \hat{r}_L$",
        alpha=0.6,
        markersize=3,
        marker=".",
    )
    mu_known = 2.63815853
    plt.axhline(
        mu_known,
        color="r",
        linestyle="--",
        label=f"Known $\mu \\approx {mu_known:.4f}$",
    )
    valid_mus = np.concatenate(
        (
            mu_est_count[np.isfinite(mu_est_count)],
            mu_est_ratio[np.isfinite(mu_est_ratio)],
        )
    )
    if len(valid_mus) > 0:
        median_mu = np.median(valid_mus[-max(1, len(valid_mus) // 2) :])
        if np.isfinite(median_mu):
            plt.ylim(max(0, median_mu - 0.5), median_mu + 0.5)
    plt.title("Estimates of Connective Constant $\mu$ vs L")
    plt.xlabel("Length L")
    plt.ylabel("Estimated $\mu$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === Figure 3: Plot ESS vs L ===
    plt.figure(figsize=(8, 6))
    # ... (Plotting code from previous answer) ...
    if "ess_history" in locals() and len(ess_history) == actual_L_max_reached:
        L_values_ess = np.arange(1, actual_L_max_reached + 1)
        plt.plot(L_values_ess, ess_history, marker=".", linestyle="-", markersize=3)
        plt.axhline(
            N_PARTICLES * RESAMPLE_THRESH,
            color="r",
            linestyle="--",
            label=f"Resample Threshold ({N_PARTICLES * RESAMPLE_THRESH:.0f})",
        )
        plt.title("Effective Sample Size (ESS) vs L")
        plt.xlabel("Length L")
        plt.ylabel("ESS")
        plt.ylim(0, N_PARTICLES * 1.1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "ESS history not available.")  # Fallback

    # === Figure 4: Plot for Regression (log(c_L)/L vs 1/L) ===
    plt.figure(figsize=(8, 6))
    # ... (Plotting code from previous answer, including plotting fit line) ...
    L_vals_regr = np.arange(
        max(2, int(actual_L_max_reached * 0.1)), actual_L_max_reached + 1
    )
    valid_indices_regr = [
        L for L in L_vals_regr if L < len(log_c_L) and np.isfinite(log_c_L[L])
    ]
    if len(valid_indices_regr) > 1:
        L_valid_regr = np.array(valid_indices_regr)
        y_regr = np.array([log_c_L[L] / L for L in L_valid_regr])
        x_regr = 1.0 / L_valid_regr
        finite_y_mask = np.isfinite(y_regr)
        x_regr, y_regr = x_regr[finite_y_mask], y_regr[finite_y_mask]
        L_valid_regr = L_valid_regr[finite_y_mask]
        if len(y_regr) > 1:
            plt.plot(
                x_regr,
                y_regr,
                "o",
                markersize=3,
                label="Simulation Data ($\log(\hat{c}_L)/L$)",
            )
            if (
                mu_est_regr is not None and len(y_regr) >= 3
            ):  # Need >=3 points for regression
                try:
                    x1_fit = 1.0 / L_valid_regr
                    x2_fit = np.log(L_valid_regr) / L_valid_regr
                    A_fit = np.vstack([np.ones(len(L_valid_regr)), x1_fit, x2_fit]).T
                    beta, _, _, _ = np.linalg.lstsq(A_fit, y_regr, rcond=None)
                    log_mu_fit, log_A_fit, gamma_m1_fit = beta[0], beta[1], beta[2]
                    if np.isfinite(log_mu_fit):
                        y_fit = log_mu_fit + log_A_fit * x1_fit + gamma_m1_fit * x2_fit
                        sort_indices = np.argsort(x_regr)
                        plt.plot(
                            x_regr[sort_indices],
                            y_fit[sort_indices],
                            "r-",
                            label=f"Regression Fit ($\log \mu \\approx {log_mu_fit:.4f}$)",
                        )
                        plt.axhline(
                            log_mu_fit,
                            color="g",
                            linestyle=":",
                            label=f"Intercept ($\hat{{\log \mu}}_{{regr}}$)",
                        )
                except Exception as e:
                    print(f"Could not plot regression line: {e}")
            plt.title("Regression Plot: $\log(\hat{c}_L)/L$ vs $1/L$")
            plt.xlabel("$1/L$")
            plt.ylabel("$\log(\hat{c}_L)/L$")
            plt.legend()
            plt.grid(True)
            plt.xlim(left=-0.005, right=max(x_regr) * 1.1 if len(x_regr) > 0 else 0.1)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "Not enough finite data for regression plot.")
            plt.title("Regression Plot")
    else:
        plt.text(0.5, 0.5, "Not enough valid L values for regression plot.")
        plt.title("Regression Plot")

    # --- Show all created figures ---
    plt.show()

else:
    print("Simulation did not run long enough to generate plots.")
