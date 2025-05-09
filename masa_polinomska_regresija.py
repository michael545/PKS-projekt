import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from scipy.interpolate import UnivariateSpline
from typing import Tuple, List, Dict, Callable, Any, NamedTuple
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class AnalysisConfig:
    file_path: str
    label: str  # For plot titles and window titles
    closing_phase_spline_subtract_value: float
    max_delta_mass_close: float
    opening_linear_slope: float  # Corresponds to the max flow rate for this dataset (e.g., 200 for nominal, 300 for the other)

# Define configurations for each dataset
CONFIGS: List[AnalysisConfig] = [
    AnalysisConfig(
        file_path=r"C:\Users\micha\Automation\vnetil_karakterirstika.txt",
        label="Dataset 1 (Nominal)",
        closing_phase_spline_subtract_value=3278.0, # Original value
        max_delta_mass_close=1190.0,              # Original value
        opening_linear_slope=200.0                # Assuming nominal flow rate of 200
    ),
    AnalysisConfig(
        file_path=r"C:\Users\micha\Automation\ventil_karakteristika_300.txt",
        label="Dataset 2 (300 Flow Rate)",
        closing_phase_spline_subtract_value=3278.0 * 1.5,
        max_delta_mass_close=1190.0 * 1.5,
        opening_linear_slope=300.0 # Assuming this file corresponds to a 300 g/s (or similar unit) flow rate
    ),
]

# --- Global Constants (assumed common for all datasets unless in AnalysisConfig) ---
# Data processing constants for load_mass_data
ARRAY_NAME_FILTER: str = "masa_func_VENTILA_A"
TIME_SCALING_FACTOR: float = 0.1
MASS_OFFSET: float = 200.0
FILL_VALUE_INDEX_THRESHOLD: int = 294
FILL_VALUE_SOURCE_INDEX: int = 262

# Regression and modeling constants
OPENING_PHASE_TIME_RANGE: Tuple[float, float] = (0.0, 12.0)
OPENING_PHASE_POLY_DEGREE: int = 5

CLOSING_PHASE_TIME_RANGE: Tuple[float, float] = (20.0, 29.0)
CLOSING_PHASE_ADJUSTED_DURATION: float = 9.0

# Test case generation
SIMULATION_TIME_STEP: float = 0.1
SIMULATION_NUM_STEPS: int = 400

# Inverse characteristics fitting
INVERSE_TIME_LIMIT_POLY: float = 9.0
INVERSE_POLY_DEGREE: int = 4

# Plotting defaults
DEFAULT_FIG_SIZE: Tuple[int, int] = (10, 6)
DEFAULT_SCATTER_ALPHA: float = 0.5
FIT_LINE_WIDTH: float = 2.0

class AnalysisResults(NamedTuple):
    config_label: str
    raw_mass_data: pd.DataFrame
    opening_poly_model: Callable[[np.ndarray], np.ndarray]
    open_t_fit: np.ndarray
    open_m_fit: np.ndarray
    open_coeffs: np.ndarray
    closing_spline_model: Callable[[np.ndarray], np.ndarray]
    close_t_norm_fit: np.ndarray
    close_m_fit: np.ndarray
    simulated_results_df: pd.DataFrame
    inv_poly_total_model: Callable[[np.ndarray], np.ndarray]
    inv_poly_coeffs: np.ndarray
    inv_linear_total_model: Callable[[np.ndarray], np.ndarray]
    inv_linear_coeffs: List[float]
    mass_transition_inv: float

# --- Data Loading and Preprocessing ---

def load_mass_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses mass data from a specified file."""
    try:
        mass_data_raw = pd.read_csv(
            file_path, sep=r"\s+", header=None,
            names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2",
                   "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
            engine="python"
        )
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

    mass_data_filtered = mass_data_raw[
        mass_data_raw["ArrayName"].str.contains(ARRAY_NAME_FILTER, na=False)
    ].copy()

    if mass_data_filtered.empty:
        print(f"Warning: No data matching filter '{ARRAY_NAME_FILTER}' in {file_path}.")
        return pd.DataFrame()
        
    mass_data_filtered["ArrayIndex"] = mass_data_filtered["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)
    mass_values = mass_data_filtered[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
    mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")

    source_fill_data = mass_values[mass_values["ArrayIndex"] == FILL_VALUE_SOURCE_INDEX]
    if not source_fill_data.empty:
        fill_value = source_fill_data["ActualValue"].values[0]
        mass_values.loc[mass_values["ArrayIndex"] > FILL_VALUE_INDEX_THRESHOLD, "ActualValue"] = fill_value
    else:
        print(f"Warning: Source index {FILL_VALUE_SOURCE_INDEX} for fill value not found in {file_path}.")

    mass_values["ActualValue"] = mass_values["ActualValue"] - MASS_OFFSET
    mass_values["Time"] = mass_values["ArrayIndex"] * TIME_SCALING_FACTOR
    return mass_values.sort_values(by="Time").reset_index(drop=True)

# --- Regression Models ---

def perform_polynomial_regression(
    data: pd.DataFrame, degree: int, time_range: Tuple[float, float],
    x_col: str = "Time", y_col: str = "ActualValue"
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Performs polynomial regression."""
    time_mask = (data[x_col] >= time_range[0]) & (data[x_col] <= time_range[1])
    filtered_data = data[time_mask].dropna(subset=[x_col, y_col])
    x_values, y_values = filtered_data[x_col].values, filtered_data[y_col].values

    if len(x_values) < degree + 1:
        print(f"Warning: Not enough data points ({len(x_values)}) for polynomial degree {degree}. Need {degree + 1}.")
        coeffs = np.zeros(degree + 1)
        model = np.poly1d(coeffs)
        return model, np.array([]), np.array([]), coeffs

    coefficients = np.polyfit(x_values, y_values, degree)
    poly_model = np.poly1d(coefficients)
    x_fit = np.linspace(time_range[0], time_range[1], 200)
    y_fit = poly_model(x_fit)
    return poly_model, x_fit, y_fit, coefficients

def perform_spline_regression(
    data: pd.DataFrame, time_range: Tuple[float, float], subtract_value: float = 0.0,
    x_col: str = "Time", y_col: str = "ActualValue"
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
    """Performs spline regression with normalized time."""
    time_mask = (data[x_col] >= time_range[0]) & (data[x_col] <= time_range[1])
    filtered_data = data[time_mask].dropna(subset=[x_col, y_col]).copy()

    if filtered_data.empty:
        print("Warning: No data for spline regression in the given time range.")
        return lambda x: np.zeros_like(x), np.array([]), np.array([])

    min_time_in_range = filtered_data[x_col].min()
    filtered_data["NormalizedTime"] = filtered_data[x_col] - min_time_in_range
    filtered_data["AdjustedValue"] = filtered_data[y_col] - subtract_value
    filtered_data = filtered_data.sort_values(by="NormalizedTime")

    x_values = filtered_data["NormalizedTime"].values
    y_values = filtered_data["AdjustedValue"].values
    unique_x, unique_idx = np.unique(x_values, return_index=True)
    unique_y = y_values[unique_idx]

    if len(unique_x) < 4: # k=3 (cubic spline) requires at least 4 points
        print(f"Warning: Not enough unique points ({len(unique_x)}) for spline (k=3). Need 4.")
        return lambda x: np.zeros_like(x), np.array([]), np.array([])

    spline_model = UnivariateSpline(unique_x, unique_y, k=3, s=0)
    spline_duration = time_range[1] - time_range[0]
    x_spline_fit = np.linspace(0, spline_duration, 200)
    y_spline_fit = spline_model(x_spline_fit)
    return spline_model, x_spline_fit, y_spline_fit

# --- Simulation of Mass Components ---

def simulate_mass_components(
    opening_poly_model: Callable[[np.ndarray], np.ndarray],
    closing_spline_model: Callable[[np.ndarray], np.ndarray],
    simulation_times: np.ndarray,
    opening_linear_slope: float, # Added parameter
    max_delta_mass_close: float  # Added parameter
) -> pd.DataFrame:
    """Simulates mass components over time."""
    results = []
    for t_current in simulation_times:
        if t_current <= INVERSE_TIME_LIMIT_POLY:
            mass_open_val = opening_poly_model(t_current)
        else:
            mass_open_val = opening_poly_model(INVERSE_TIME_LIMIT_POLY) + \
                            opening_linear_slope * (t_current - INVERSE_TIME_LIMIT_POLY)

        if t_current <= CLOSING_PHASE_ADJUSTED_DURATION:
            time_for_spline_input = CLOSING_PHASE_ADJUSTED_DURATION - t_current
            delta_m_close_val = max_delta_mass_close - closing_spline_model(time_for_spline_input)
        else:
            delta_m_close_val = max_delta_mass_close
        
        total_mass_val = mass_open_val + delta_m_close_val
        results.append({
            "Time (s)": t_current, "Mass Open (g)": mass_open_val,
            "Delta Mass Close (g)": delta_m_close_val, "Total Mass (g)": total_mass_val
        })
    return pd.DataFrame(results)

# --- Inverse Characteristics Fitting (Time from Mass) ---

def fit_inverse_polynomial_total(
    simulation_results: pd.DataFrame, time_limit: float = INVERSE_TIME_LIMIT_POLY, degree: int = INVERSE_POLY_DEGREE
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Fits an inverse polynomial model (Time = f(Total_Mass))."""
    mask = simulation_results["Time (s)"] <= time_limit
    filtered = simulation_results[mask]
    total_mass, time_values = filtered["Total Mass (g)"].values, filtered["Time (s)"].values
    
    sort_idx = np.argsort(total_mass)
    mass_sorted, time_sorted = total_mass[sort_idx], time_values[sort_idx]
    unique_mass, unique_idx = np.unique(mass_sorted, return_index=True)
    unique_time = time_sorted[unique_idx]

    if len(unique_mass) < degree + 1:
        print(f"Warning: Not enough unique points for inverse poly (degree {degree}). Have {len(unique_mass)}, need {degree+1}.")
        coeffs = np.zeros(degree + 1)
        return np.poly1d(coeffs), coeffs
        
    coefficients = np.polyfit(unique_mass, unique_time, deg=degree)
    return np.poly1d(coefficients), coefficients

def fit_inverse_linear_total(
    simulation_results: pd.DataFrame, time_limit: float = INVERSE_TIME_LIMIT_POLY
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[float], float]:
    """Fits an inverse linear model (Time = f(Total_Mass))."""
    mask_linear_part = simulation_results["Time (s)"] > time_limit
    filtered_linear_part = simulation_results[mask_linear_part]
    
    mass_at_time_limit_poly_end = simulation_results[simulation_results["Time (s)"] <= time_limit]["Total Mass (g)"]
    if mass_at_time_limit_poly_end.empty:
        print(f"Warning: No data at or before time_limit {time_limit}s to determine transition mass.")
        mass_transition = 0 # Fallback
    else:
        mass_transition = mass_at_time_limit_poly_end.max()
    time_transition = time_limit

    if filtered_linear_part.empty:
        print("Warning: No data for inverse linear fit beyond time_limit.")
        slope, intercept = 0.005, time_transition - 0.005 * mass_transition # Default slope
        return np.poly1d([slope, intercept]), [slope, intercept], mass_transition

    final_mass = filtered_linear_part["Total Mass (g)"].max()
    # Find the time corresponding to this final_mass in the filtered_linear_part
    final_time_series = filtered_linear_part[filtered_linear_part["Total Mass (g)"] == final_mass]["Time (s)"]
    final_time = final_time_series.max() if not final_time_series.empty else time_transition


    if final_mass <= mass_transition:
        slope = 0.005 # Default slope if no valid range
    else:
        slope = (final_time - time_transition) / (final_mass - mass_transition)
    
    intercept = time_transition - slope * mass_transition
    return np.poly1d([slope, intercept]), [slope, intercept], mass_transition

# --- Plotting Utilities ---

def _setup_plot_interactive(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend: bool = True):
    """Helper for plot setup and making axes interactive with mplcursors."""
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.4)
    if legend:
        ax.legend()
    return mplcursors.cursor(ax, hover=True) # Return cursor for specific annotations if needed

def plot_opening_and_closing_phases(
    fig_title: str, raw_mass_data: pd.DataFrame,
    opening_fit_time: np.ndarray, opening_fit_mass: np.ndarray,
    closing_fit_time_normalized: np.ndarray, closing_fit_mass: np.ndarray
) -> None:
    """Plots opening and closing phase regressions in a new figure."""
    fig, (ax_opening, ax_closing) = plt.subplots(2, 1, figsize=(10, 12))
    fig.canvas.manager.set_window_title(fig_title) # type: ignore
    fig.suptitle(fig_title, fontsize=16)

    ax_opening.scatter(raw_mass_data["Time"], raw_mass_data["ActualValue"], alpha=DEFAULT_SCATTER_ALPHA, label='Izmerjeno')
    ax_opening.plot(opening_fit_time, opening_fit_mass, 'r-', lw=FIT_LINE_WIDTH, label='Polinomska prilagoditev 5. reda')
    cursor_open = _setup_plot_interactive(ax_opening, 'Faza odpiranja: Masa m(t) (0-12 sekund)', 'Čas (s)', 'Masa (g)')
    @cursor_open.connect("add")
    def _(sel): sel.annotation.set_text(f"Čas: {sel.target[0]:.2f}s\nMasa: {sel.target[1]:.2f}g")

    ax_closing.plot(closing_fit_time_normalized, closing_fit_mass, 'b-', lw=FIT_LINE_WIDTH, label='Spline fit')
    cursor_close = _setup_plot_interactive(ax_closing, f'Faza zapiranja: Sprememba mase m(t) (prilagojeno na 0-{int(CLOSING_PHASE_ADJUSTED_DURATION)}s)',
                                          'Prilagojen čas v fazi zapiranja (s)', 'Sprememba mase (g)')
    @cursor_close.connect("add")
    def _(sel): sel.annotation.set_text(f"Pril. čas: {sel.target[0]:.2f}s\nΔMasa: {sel.target[1]:.2f}g")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle


def plot_simulated_mass_components(fig_title: str, simulation_results: pd.DataFrame) -> None:
    """Plots simulated mass components in a new figure."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    fig.canvas.manager.set_window_title(fig_title) # type: ignore

    ax.plot(simulation_results["Time (s)"], simulation_results["Total Mass (g)"], label="Skupna masa", color='b', lw=FIT_LINE_WIDTH)
    ax.plot(simulation_results["Time (s)"], simulation_results["Mass Open (g)"], label="Masa odpiranja", color='g', linestyle='--')
    ax.plot(simulation_results["Time (s)"], simulation_results["Delta Mass Close (g)"], label="Sprememba mase zapiranja", color='r', linestyle='--')
    
    cursor = _setup_plot_interactive(ax, fig_title, "Čas (s)", "Masa (g)")
    @cursor.connect("add")
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = sel.target
        label_map = {
            "Skupna masa": f"Čas: {x:.2f}s\nSkupna masa: {y:.2f}g",
            "Masa odpiranja": f"Čas: {x:.2f}s\nMasa odpiranja: {y:.2f}g",
            "Sprememba mase zapiranja": f"Čas: {x:.2f}s\nSprememba mase zapiranja: {y:.2f}g"
        }
        sel.annotation.set_text(label_map.get(sel.artist.get_label(), f"{x=:.2f}, {y=:.2f}"))
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    plt.tight_layout()

def plot_inverse_total_mass_characteristics(
    fig_title: str, simulation_results: pd.DataFrame,
    inv_poly_model: Callable[[np.ndarray], np.ndarray],
    inv_linear_model: Callable[[np.ndarray], np.ndarray], mass_transition: float
) -> None:
    """Plots inverse characteristics (Time vs. Total Mass) in a new figure."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    fig.canvas.manager.set_window_title(fig_title) # type: ignore

    ax.scatter(simulation_results["Total Mass (g)"], simulation_results["Time (s)"],
               alpha=DEFAULT_SCATTER_ALPHA - 0.2, label='Izračunana skupna masa', color='#1f77b4')
    
    mass_min = simulation_results["Total Mass (g)"].min()
    mass_max = simulation_results["Total Mass (g)"].max()

    mass_poly_range = np.linspace(mass_min if not np.isnan(mass_min) else 0, mass_transition, 200)
    time_poly_fit = inv_poly_model(mass_poly_range)
    ax.plot(mass_poly_range, time_poly_fit, 'r-', lw=FIT_LINE_WIDTH, label=f'Polinomska regresija (do {mass_transition:.0f}g)')

    mass_linear_range = np.linspace(mass_transition, mass_max if not np.isnan(mass_max) else mass_transition + 1000 , 200)
    time_linear_fit = inv_linear_model(mass_linear_range)
    ax.plot(mass_linear_range, time_linear_fit, 'g--', lw=FIT_LINE_WIDTH, label=f'Linearna regresija (nad {mass_transition:.0f}g)')
    
    ax.axvline(mass_transition, color='k', linestyle=':', label=f'Prehodna masa: {mass_transition:.1f}g')
    
    cursor = _setup_plot_interactive(ax, fig_title, "Skupna masa (g)", "Čas (s)")
    @cursor.connect("add")
    def on_add(sel: mplcursors.Selection) -> None:
        m, t = sel.target
        text = f"Masa: {m:.1f}g\nČas: {t:.2f}s" if sel.artist == ax.collections[0] \
               else f"Model:\nMasa: {m:.1f}g\nPredviden čas: {t:.2f}s"
        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    plt.tight_layout()

# --- Main Application Logic ---

def run_analysis_for_config(config: AnalysisConfig) -> AnalysisResults | None:
    """Runs the full analysis pipeline for a given configuration."""
    print(f"\n--- Processing: {config.label} ---")
    print(f"File: {config.file_path}")
    print(f"Spline Subtract: {config.closing_phase_spline_subtract_value}, Max Delta Close: {config.max_delta_mass_close}, Opening Slope: {config.opening_linear_slope}")

    raw_mass_data = load_mass_data(config.file_path)
    if raw_mass_data.empty:
        print(f"Failed to load or process data for {config.label}. Skipping.")
        return None

    opening_poly_model, open_t_fit, open_m_fit, open_coeffs = \
        perform_polynomial_regression(raw_mass_data, OPENING_PHASE_POLY_DEGREE, OPENING_PHASE_TIME_RANGE)

    closing_spline_model, close_t_norm_fit, close_m_fit = \
        perform_spline_regression(raw_mass_data, CLOSING_PHASE_TIME_RANGE, config.closing_phase_spline_subtract_value)

    simulation_times = np.arange(0.0, SIMULATION_NUM_STEPS * SIMULATION_TIME_STEP, SIMULATION_TIME_STEP)
    simulated_results_df = simulate_mass_components(
        opening_poly_model, closing_spline_model, simulation_times,
        config.opening_linear_slope, config.max_delta_mass_close
    )

    inv_poly_total_model, inv_poly_coeffs = fit_inverse_polynomial_total(simulated_results_df)
    inv_linear_total_model, inv_linear_coeffs, mass_transition_inv = \
        fit_inverse_linear_total(simulated_results_df)
        
    # Print model details
    print(f"\n--- {config.label}: Opening Phase Polynomial Model (Mass = f(Time)) ---")
    print(f"Coefficients (degree {OPENING_PHASE_POLY_DEGREE}): {open_coeffs}")
    print(f"\n--- {config.label}: INVERZNE KARAKTERISTIKE (TOTAL MASS) ---")
    print(f"Transition Mass for Inverse Model: {mass_transition_inv:.2f}g (at Time = {INVERSE_TIME_LIMIT_POLY}s)")
    poly_terms = [f"{inv_poly_coeffs[i]:.6e}·m^{INVERSE_POLY_DEGREE-i}" for i in range(INVERSE_POLY_DEGREE + 1)]
    print(f"Inverse Polynomial Model (Mass <= {mass_transition_inv:.2f}g): t(m) = " + " + ".join(poly_terms).replace("+ -", "- "))
    print(f"Inverse Linear Model (Mass > {mass_transition_inv:.2f}g): t(m) = {inv_linear_coeffs[0]:.6e}·m + {inv_linear_coeffs[1]:.3e}".replace("+ -", "- "))

    return AnalysisResults(
        config_label=config.label, raw_mass_data=raw_mass_data,
        opening_poly_model=opening_poly_model, open_t_fit=open_t_fit, open_m_fit=open_m_fit, open_coeffs=open_coeffs,
        closing_spline_model=closing_spline_model, close_t_norm_fit=close_t_norm_fit, close_m_fit=close_m_fit,
        simulated_results_df=simulated_results_df,
        inv_poly_total_model=inv_poly_total_model, inv_poly_coeffs=inv_poly_coeffs,
        inv_linear_total_model=inv_linear_total_model, inv_linear_coeffs=inv_linear_coeffs,
        mass_transition_inv=mass_transition_inv
    )

def main() -> None:
    """Main function to run valve characterization for multiple configurations."""
    plt.close('all')
    all_results: List[AnalysisResults] = []

    for config in CONFIGS:
        results = run_analysis_for_config(config)
        if results:
            all_results.append(results)

    if not all_results:
        print("No data processed. Exiting.")
        return

    for res in all_results:
        plot_inverse_total_mass_characteristics(
            fig_title=f"Inverzne karakteristike- {res.config_label}",
            simulation_results=res.simulated_results_df,
            inv_poly_model=res.inv_poly_total_model,
            inv_linear_model=res.inv_linear_total_model,
            mass_transition=res.mass_transition_inv
        )
        plot_opening_and_closing_phases(
            fig_title=f"Regresija za posamezne faze - {res.config_label}",
            raw_mass_data=res.raw_mass_data,
            opening_fit_time=res.open_t_fit, opening_fit_mass=res.open_m_fit,
            closing_fit_time_normalized=res.close_t_norm_fit, closing_fit_mass=res.close_m_fit
        )
        plot_simulated_mass_components(
            fig_title=f"Simulirane komponente mase - {res.config_label}",
            simulation_results=res.simulated_results_df
        )

    plt.show() # Display all figures simultaneously

if __name__ == "__main__":
    main()