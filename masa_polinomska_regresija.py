# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import mplcursors
# from scipy.interpolate import UnivariateSpline

# mass_file_path = r"C:\Users\micha\Automation\vnetil_karakterirstika.txt"

# mass_data = pd.read_csv(
#     mass_file_path,
#     sep=r"\s+",
#     header=None,
#     names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2",
#            "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
#     engine="python"
# )

# mass_data = mass_data[mass_data["ArrayName"].str.contains("masa_func_VENTILA_A", na=False)]
# mass_data["ArrayIndex"] = mass_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)

# mass_values = mass_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
# mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")
# mass_values.loc[mass_values["ArrayIndex"] > 294, "ActualValue"] = mass_values.loc[mass_values["ArrayIndex"] == 262, "ActualValue"].values[0]

# mass_values["ActualValue"] = mass_values["ActualValue"] - 200

# mass_values["Time"] = mass_values["ArrayIndex"] * 0.1  # Convert to seconds
# mass_values = mass_values.sort_values(by="Time")

# time_mask = (mass_values["Time"] >= 0) & (mass_values["Time"] <= 12)
# filtered_data = mass_values[time_mask].dropna()

# degree = 5
# x = filtered_data["Time"]
# y = filtered_data["ActualValue"]
# coefficients = np.polyfit(x, y, degree)
# poly = np.poly1d(coefficients)

# # fittane vrednosti med 0 in 12
# x_fit = np.linspace(0, 12, 120)
# y_fit = poly(x_fit)

# # po 20s se ventil zacne zapirat
# time_mask2 = (mass_values["Time"] >= 20) & (mass_values["Time"] <= 29)
# filtered_data2 = mass_values[time_mask2].dropna().copy()

# # zacni od 0s
# filtered_data2["Time"] = filtered_data2["Time"] - filtered_data2["Time"].min()
# # - 3278( ker se tam zacne ventil zapirat) od ActualValue za 2. graf
# filtered_data2["ActualValue"] = filtered_data2["ActualValue"] - 3278

# # Perregrejisa na novem casu
# x2 = filtered_data2["Time"]  # nova obmocja 0-9
# y2 = filtered_data2["ActualValue"]
# # coefficients2 = np.polyfit(x2, y2, degree)
# # poly2 = np.poly1d(coefficients2)

# # # fit
# # x_fit2 = np.linspace(0, 9, 100)  # 0-9 namesto 20-29
# # y_fit2 = poly2(x_fit2)

# spline = UnivariateSpline(x2, y2, k=3)  # k je stopnja za spline
# xs = np.linspace(0, 9, 90000)
# ys = spline(xs)


# print("\n\n--- Koeficienti polinoma ---")
# print("\n0-12s regresija (original time):")
# print("masa(t) = ")
# for i, coeff in enumerate(coefficients):
#     print(f"  {coeff:.4e} * t^{degree-i}")

# # print("\n0-9s regressija:")
# # print("Mass(t') = ")
# # for i, coeff in enumerate(coefficients2):
# #     print(f"  {coeff:.4e} * t'^{degree-i}")



# test_times = [1.0 + 0.2 * i for i in range(71)]

# print("\n\n--- Test Cases ---")
# for t in test_times:
#     mass_open_val = poly(t)
#     delta_m_close_val = 1190 - spline(9 - t)
#     total_mass_val = mass_open_val + delta_m_close_val
#     print(f"At t = {t}s:")
#     print(f"  Mass Open: {mass_open_val:.2f} g")
#     print(f"  Delta Mass Close: {delta_m_close_val:.2f} g")
#     print(f"  Total Mass: {total_mass_val:.2f} g")
#     print("-" * 20)

# # Create vertical subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# # First regression plot (0-12s)
# scatter1 = ax1.scatter(x, y, alpha=0.5, label='Measured')
# line1, = ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'5th Order Fit')
# ax1.set_title('masa(t) polinomska regresija (0-12 seconds)')
# ax1.set_ylabel('masa (g)')
# ax1.grid(True)
# ax1.legend()

# # Second regression plot (0-9s adjusted)
# scatter2 = ax2.scatter(x2, y2, alpha=0.5, label='Adjusted data')
# line2, = ax2.plot(xs, ys, 'b-', lw=2, label=f'spline fit')
# ax2.set_title('mass(t) polinomska regresija- spline fit (0-9 seconds)')
# ax2.set_xlabel('čas(s)')
# ax2.set_ylabel('masa (g)')
# ax2.grid(True)
# ax2.legend()

# plt.tight_layout()

# # super cursors/ matlab funkcije
# cursor1 = mplcursors.cursor([scatter1, line1], hover=True)
# cursor2 = mplcursors.cursor([scatter2, line2], hover=True)

# @cursor1.connect("add")
# def on_add1(sel):
#     if sel.artist == scatter1:
#         x, y = sel.target
#         sel.annotation.set(text=f"čas: {x:.2f}s\n masa: {y:.2f}g")
#     else:
#         x, y = sel.target
#         sel.annotation.set(text=f"čas: {x:.2f}s\nfittana maasa: {y:.2f}g")
#     sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

# @cursor2.connect("add")
# def on_add2(sel):
#     if sel.artist == scatter2:
#         x, y = sel.target
#         sel.annotation.set(text=f"čas: {x:.2f}s\nmasa: {y:.2f}g")
#     else:
#         x, y = sel.target
#         sel.annotation.set(text=f"čas: {x:.2f}s\nfittana Mass: {y:.2f}g")

# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from scipy.interpolate import UnivariateSpline

def plot_inverse_characteristics(data, poly_model, linear_model, m_transition):
    """Visualizes the inverse characteristics with transition point."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original data
    mask = data["Time"] <= 12
    opening_data = data[mask]
    ax.scatter(opening_data["ActualValue"], opening_data["Time"], 
               alpha=0.5, label='Measured Data')
    
    # Plot polynomial fit
    m_range_poly = np.linspace(opening_data["ActualValue"].min(), m_transition, 100)
    t_poly = poly_model(m_range_poly)
    ax.plot(m_range_poly, t_poly, 'r-', lw=2, 
            label='4th Order Polynomial Fit (0-9s)')
    
    # Plot linear extension
    m_range_lin = np.linspace(m_transition, opening_data["ActualValue"].max(), 50)
    t_lin = linear_model(m_range_lin)
    ax.plot(m_range_lin, t_lin, 'g--', lw=2, 
            label='Linear Extension (>9s)')
    
    # Highlight transition point
    ax.axvline(m_transition, color='k', linestyle=':', 
               label=f'Transition Mass: {m_transition:.1f}g')
    
    ax.set_title("Inverse Characteristic: Time vs. Mass")
    ax.set_xlabel("Mass (g)")
    ax.set_ylabel("Time (s)")
    ax.grid(True)
    ax.legend()
    
    # Add interactive cursor
    cursor = mplcursors.cursor(ax, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        if sel.artist == ax.collections[0]:  # Original data points
            m, t = sel.target
            sel.annotation.set(text=f"Mass: {m:.1f}g\nTime: {t:.2f}s")
        else:  # Fit lines
            m, t = sel.target
            sel.annotation.set(text=f"Mass: {m:.1f}g\nPredicted Time: {t:.2f}s")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    
    plt.show()

def fit_inverse_polynomial_total(rezultati, time_limit=9.0, degree=4):
    """Fits polynomial to total mass data for t ≤ 9s."""
    mask = rezultati["Time (s)"] <= time_limit
    filtered = rezultati[mask]
    
    # Extract total mass (m) and time (t)
    m = filtered["Total Mass (g)"].values
    t = filtered["Time (s)"].values
    
    # Sort by mass and remove duplicates
    sort_idx = np.argsort(m)
    m_sorted = m[sort_idx]
    t_sorted = t[sort_idx]
    unique_m, unique_idx = np.unique(m_sorted, return_index=True)
    
    # Fit polynomial
    coefficients = np.polyfit(unique_m, t_sorted[unique_idx], deg=degree)
    return np.poly1d(coefficients), coefficients

def fit_inverse_linear_total(rezultati, time_limit=9.0):
    """Fits linear model to total mass data for t > 9s."""
    mask = rezultati["Time (s)"] > time_limit
    filtered = rezultati[mask]
    
    # Get transition parameters
    m_trans = rezultati[rezultati["Time (s)"] <= time_limit]["Total Mass (g)"].max()
    t_trans = time_limit
    
    # Calculate linear slope
    final_m = filtered["Total Mass (g)"].max()
    final_t = filtered["Time (s)"].max()
    slope = (final_t - t_trans) / (final_m - m_trans)
    
    # Create linear function
    linear_fn = np.poly1d([slope, t_trans - slope*m_trans])
    return linear_fn, [slope, t_trans - slope*m_trans], m_trans

def load_mass_data(file_path):
    """Load and preprocess mass data from a file."""
    mass_data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2",
               "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
        engine="python"
    )
    mass_data = mass_data[mass_data["ArrayName"].str.contains("masa_func_VENTILA_A", na=False)]
    mass_data["ArrayIndex"] = mass_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)

    mass_values = mass_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
    mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")
    mass_values.loc[mass_values["ArrayIndex"] > 294, "ActualValue"] = mass_values.loc[mass_values["ArrayIndex"] == 262, "ActualValue"].values[0]
    mass_values["ActualValue"] = mass_values["ActualValue"] - 200
    mass_values["Time"] = mass_values["ArrayIndex"] * 0.1
    return mass_values.sort_values(by="Time")

def perform_polynomial_regression(data, degree, time_range):
    """Perform polynomial regression on the given data."""
    time_mask = (data["Time"] >= time_range[0]) & (data["Time"] <= time_range[1])
    filtered_data = data[time_mask].dropna()
    x = filtered_data["Time"]
    y = filtered_data["ActualValue"]
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    x_fit = np.linspace(time_range[0], time_range[1], 120)
    y_fit = poly(x_fit)
    return poly, x_fit, y_fit, coefficients

def perform_spline_regression(data, time_range, subtract_value=0):
    """Perform spline regression on the given data."""
    time_mask = (data["Time"] >= time_range[0]) & (data["Time"] <= time_range[1])
    filtered_data = data[time_mask].dropna().copy()
    filtered_data["Time"] = filtered_data["Time"] - filtered_data["Time"].min()
    filtered_data["ActualValue"] = filtered_data["ActualValue"] - subtract_value
    x = filtered_data["Time"]
    y = filtered_data["ActualValue"]
    spline = UnivariateSpline(x, y, k=3)
    xs = np.linspace(0, time_range[1] - time_range[0], 90000)
    ys = spline(xs)
    return spline, xs, ys

def plot_regression_results(ax, x, y, x_fit, y_fit, title, label):
    """Plot the regression results on the given axis."""
    # scatter is already an artist object
    scatter = ax.scatter(x, y, alpha=0.5, label=label) 
    
    # ax.plot returns a list of Line2D objects. Unpack the first (and only) one.
    # Note the comma after 'line' for unpacking the single-element list.
    line, = ax.plot(x_fit, y_fit, 'r-', lw=2, label=f'5th Order Fit')     
    ax.set_title(title)
    ax.set_ylabel('Mass (g)')
    ax.grid(True)
    ax.legend()
    # Now return the individual artist objects
    return scatter, line

def add_interactive_cursor(cursor, scatter, line):
    """Add interactive cursor to the plot."""
    @cursor.connect("add")
    def on_add(sel):
        if sel.artist == scatter:
            x, y = sel.target
            sel.annotation.set(text=f"Time: {x:.2f}s\nMass: {y:.2f}g")
        else:
            x, y = sel.target
            sel.annotation.set(text=f"Time: {x:.2f}s\nFitted Mass: {y:.2f}g")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

def run_test_cases(poly, spline, test_times, linear_slope=200):
    """Run test cases for specific times and store results in a DataFrame."""
    results = []

    for t in test_times:
        # Calculate mass open with linear continuation after 12s
        if t <= 12:
            mass_open_val = poly(t)
        else:
            mass_open_val = poly(12) + linear_slope * (t - 12)

        # Calculate delta mass close with adjustment for times greater than 9s
        if t <= 9:
            delta_m_close_val = 1190 - spline(9 - t)
        else:
            delta_m_close_val = 1190  # or another logic if needed

        total_mass_val = mass_open_val + delta_m_close_val

        # Append the results to the list
        results.append({
            "Time (s)": t,
            "Mass Open (g)": mass_open_val,
            "Delta Mass Close (g)": delta_m_close_val,
            "Total Mass (g)": total_mass_val
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Print the DataFrame
    print("\n--- Test Case Results ---")
    print(results_df)

    return results_df

def plot_results(rezultati):
    """Plot the results stored in the DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the total mass over time
    ax.plot(rezultati["Time (s)"], rezultati["Total Mass (g)"], label="Total Mass", color='b', linewidth=2)
    ax.plot(rezultati["Time (s)"], rezultati["Mass Open (g)"], label="Mass Open", color='g', linestyle='--')
    ax.plot(rezultati["Time (s)"], rezultati["Delta Mass Close (g)"], label="Delta Mass Close", color='r', linestyle='--')

    ax.set_title("Calculated Mass Components vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (g)")
    ax.legend()
    ax.grid(True)

    # Add interactive cursor
    cursor = mplcursors.cursor(ax.get_lines(), hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        if sel.artist.get_label() == "Total Mass":
            sel.annotation.set(text=f"Time: {x:.2f}s\nTotal Mass: {y:.2f}g")
        elif sel.artist.get_label() == "Mass Open":
            sel.annotation.set(text=f"Time: {x:.2f}s\nMass Open: {y:.2f}g")
        elif sel.artist.get_label() == "Delta Mass Close":
            sel.annotation.set(text=f"Time: {x:.2f}s\nDelta Mass Close: {y:.2f}g")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    plt.show()

def main():
    # Load and preprocess data
    file_path = r"C:\Users\micha\Automation\vnetil_karakterirstika.txt"
    mass_values = load_mass_data(file_path)

    # Perform polynomial regression for opening phase (0-12s)
    poly, x_fit, y_fit, coefficients = perform_polynomial_regression(mass_values, 5, [0, 12])

    # Perform spline regression for closing phase (20-29s adjusted to 0-9s)
    spline, xs, ys = perform_spline_regression(mass_values, [20, 29], subtract_value=3278)

    # Run test cases and store results
    test_times = [0.0 + 0.1 * i for i in range(400)]
    rezultati = run_test_cases(poly, spline, test_times)

    # Fit inverse models to TOTAL MASS
    inv_poly_total, coeffs_poly = fit_inverse_polynomial_total(rezultati)
    inv_linear_total, coeffs_linear, m_trans = fit_inverse_linear_total(rezultati)
    
    # Unified inverse function
    def t_on_total(m_total):
        if m_total <= m_trans:
            return inv_poly_total(m_total)
        else:
            return inv_linear_total(m_total)

    # Print models
    print("\n=== INVERSE CHARACTERISTICS (TOTAL MASS) ===")
    print(f"Transition Mass: {m_trans:.2f}g")
    print("\nPolynomial Model (m ≤ {:.2f}g):".format(m_trans))
    print("t(m) = " + " + ".join(f"{coeffs_poly[i]:.3e}·m^{4-i}" for i in range(5)))
    
    print("\nLinear Model (m > {:.2f}g):".format(m_trans))
    print(f"t(m) = {coeffs_linear[0]:.3e}·m + {coeffs_linear[1]:.3e}")

    # Example calculations
    print("\n--- Example Inverse Calculations ---")
    test_masses = [500, 3000, m_trans, 8000, 10000]
    for m in test_masses:
        t = t_on_total(m)
        model_type = "Poly" if m <= m_trans else "Linear"
        print(f"Total Mass: {m:6.1f}g → Time: {t:.2f}s ({model_type} model)")

    # Plotting
    plot_total_mass_characteristics(rezultati, inv_poly_total, inv_linear_total, m_trans)
    plt.show()

    # Create all plots
    plt.close('all')
    
    # Original regression plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # First regression plot (0-12s)
    scatter1 = ax1.scatter(mass_values["Time"], mass_values["ActualValue"], alpha=0.5, label='Measured')
    ax1.plot(x_fit, y_fit, 'r-', lw=2, label='5th Order Polynomial Fit')
    ax1.set_title('Opening Phase: Mass vs Time (0-12 seconds)')
    ax1.set_ylabel('Mass (g)')
    ax1.grid(True)
    ax1.legend()

    # Second regression plot (0-9s adjusted)
    ax2.plot(xs, ys, 'b-', lw=2, label='Spline Fit')
    ax2.set_title('Closing Phase: Mass Change vs Time (0-9 seconds adjusted)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass Change (g)')
    ax2.grid(True)
    ax2.legend()

    plot_results(rezultati)
    plt.show()

def plot_total_mass_characteristics(rezultati, poly_model, linear_model, m_trans):
    """Plots inverse characteristics using total mass data."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot raw total mass data
    ax.scatter(rezultati["Total Mass (g)"], rezultati["Time (s)"], 
               alpha=0.3, label='Calculated Total Mass', color='#1f77b4')
    
    # Plot polynomial fit
    m_poly = np.linspace(rezultati["Total Mass (g)"].min(), m_trans, 100)
    t_poly = poly_model(m_poly)
    ax.plot(m_poly, t_poly, 'r-', lw=2, label='4th Order Polynomial Fit (t ≤ 9s)')
    
    # Plot linear extension
    m_lin = np.linspace(m_trans, rezultati["Total Mass (g)"].max(), 100)
    t_lin = linear_model(m_lin)
    ax.plot(m_lin, t_lin, 'g--', lw=2, label='Linear Extension (t > 9s)')
    
    # Highlight transition
    ax.axvline(m_trans, color='k', linestyle=':', 
               label=f'Transition Mass: {m_trans:.1f}g')
    
    ax.set_title("Inverse Time Characteristics vs. Total Mass", fontsize=14)
    ax.set_xlabel("Total Mass (g)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Interactive annotations
    cursor = mplcursors.cursor(ax, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        if sel.artist == ax.collections[0]:  # Scatter points
            m, t = sel.target
            sel.annotation.set(text=f"Mass: {m:.1f}g\nTime: {t:.2f}s", 
                              fontfamily='monospace')
        else:  # Fit lines
            m, t = sel.target
            sel.annotation.set(text=f"Model Prediction:\n{m:.1f}g → {t:.2f}s",
                              bbox={"facecolor": "white", "alpha": 0.9})

if __name__ == "__main__":
    main()
