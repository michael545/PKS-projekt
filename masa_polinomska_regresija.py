import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from scipy.interpolate import UnivariateSpline

mass_file_path = r"C:\Users\micha\Automation\vnetil_karakterirstika.txt"

# Read mass data
mass_data = pd.read_csv(
    mass_file_path,
    sep=r"\s+",
    header=None,
    names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2",
           "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
    engine="python"
)

mass_data = mass_data[mass_data["ArrayName"].str.contains("masa_func_VENTILA_A", na=False)]
mass_data["ArrayIndex"] = mass_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)

# Keep only relevant columns and sort by ArrayIndex
mass_values = mass_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
# Convert ActualValue to numeric, handling potential missing values
mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")
mass_values.loc[mass_values["ArrayIndex"] > 294, "ActualValue"] = mass_values.loc[mass_values["ArrayIndex"] == 262, "ActualValue"].values[0]

mass_values["ActualValue"] = mass_values["ActualValue"] - 200

mass_values["Time"] = mass_values["ArrayIndex"] * 0.1  # Convert to seconds

mass_values = mass_values.sort_values(by="Time")

# First regression (0-12s)
time_mask = (mass_values["Time"] >= 0) & (mass_values["Time"] <= 12)
filtered_data = mass_values[time_mask].dropna()

# Perform 5th-order polynomial regression
degree = 5
x = filtered_data["Time"]
y = filtered_data["ActualValue"]
coefficients = np.polyfit(x, y, degree)
poly = np.poly1d(coefficients)

# Generate fitted values for the range 0 to 12 seconds
x_fit = np.linspace(0, 12, 100)
y_fit = poly(x_fit)

# Second regression (0-10s) with direct time adjustment
time_mask2 = (mass_values["Time"] >= 20) & (mass_values["Time"] <= 29)
filtered_data2 = mass_values[time_mask2].dropna().copy()

# Reset time to start from 0 for the second dataset
filtered_data2["Time"] = filtered_data2["Time"] - filtered_data2["Time"].min()

# Subtract 3278 from ActualValue for the second dataset
filtered_data2["ActualValue"] = filtered_data2["ActualValue"] - 3278

# Perform regression on adjusted time
x2 = filtered_data2["Time"]  # Now ranges 0-9
y2 = filtered_data2["ActualValue"]
coefficients2 = np.polyfit(x2, y2, degree)
poly2 = np.poly1d(coefficients2)

# Generate fitted values for adjusted time
x_fit2 = np.linspace(0, 9, 100)  # 0-9 instead of 20-29
y_fit2 = poly2(x_fit2)

# Assuming x2 and y2 are your data points for the second dataset
spline = UnivariateSpline(x2, y2, k=3)  # k is the degree of the spline
xs = np.linspace(0, 9, 90000)
ys = spline(xs)

# Print regression coefficients
print("\n\n--- Regression Coefficients ---")
print("\n0-12s Regression (original time):")
print("Mass(t) = ")
for i, coeff in enumerate(coefficients):
    print(f"  {coeff:.4e} * t^{degree-i}")

print("\n0-9s Regression:")
print("Mass(t') = ")
for i, coeff in enumerate(coefficients2):
    print(f"  {coeff:.4e} * t'^{degree-i}")



test_times = [1, 1.8, 2, 2.8, 3,4,5,6,7]


print("\n\n--- Test Cases ---")
for t in test_times:
    mass_open_val = poly(t)
    delta_m_close_val = 1190 - spline(9 - t)
    total_mass_val = mass_open_val + delta_m_close_val
    print(f"At t = {t}s:")
    print(f"  Mass Open: {mass_open_val:.2f} g")
    print(f"  Delta Mass Close: {delta_m_close_val:.2f} g")
    print(f"  Total Mass: {total_mass_val:.2f} g")
    print("-" * 20)

# Create vertical subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First regression plot (0-12s)
scatter1 = ax1.scatter(x, y, alpha=0.5, label='Measured Data')
line1, = ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'5th Order Fit')
ax1.set_title('masa(t) polinomska regresija (0-12 seconds)')
ax1.set_ylabel('masa (g)')
ax1.grid(True)
ax1.legend()

# Second regression plot (0-9s adjusted)
scatter2 = ax2.scatter(x2, y2, alpha=0.5, label='Adjusted Data')
line2, = ax2.plot(xs, ys, 'b-', lw=2, label=f'spline fit')
ax2.set_title('mass(t) polinomska regresija- spline fit (0-9 seconds)')
ax2.set_xlabel('čas(s)')
ax2.set_ylabel('masa (g)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Add interactive cursor to both subplots
cursor1 = mplcursors.cursor([scatter1, line1], hover=True)
cursor2 = mplcursors.cursor([scatter2, line2], hover=True)

# Custom annotation formatting
@cursor1.connect("add")
def on_add1(sel):
    if sel.artist == scatter1:
        x, y = sel.target
        sel.annotation.set(text=f"čas: {x:.2f}s\n masa: {y:.2f}g")
    else:
        x, y = sel.target
        sel.annotation.set(text=f"čas: {x:.2f}s\nfittana maasa: {y:.2f}g")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

@cursor2.connect("add")
def on_add2(sel):
    if sel.artist == scatter2:
        x, y = sel.target
        sel.annotation.set(text=f"čas: {x:.2f}s\nMass: {y:.2f}g")
    else:
        x, y = sel.target
        sel.annotation.set(text=f"čas: {x:.2f}s\nFitted Mass: {y:.2f}g")

plt.tight_layout()
plt.show()
