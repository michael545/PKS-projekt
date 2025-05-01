import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
mass_file_path = r"C:\Users\micha\Automation\vnetil_karakterirstika.txt"
angle_file_path = r"C:\Users\micha\Automation\ventil_karakteristika2.txt"

# Read mass data
mass_data = pd.read_csv(
    mass_file_path,
    sep=r"\s+",
    header=None,
    names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2", 
           "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
    engine="python"
)

# Extract array index from ArrayName and keep only masa_func_VENTILA_A rows
mass_data = mass_data[mass_data["ArrayName"].str.contains("masa_func_VENTILA_A", na=False)]
mass_data["ArrayIndex"] = mass_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)

# Keep only relevant columns and sort by ArrayIndex
mass_values = mass_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
# Convert ActualValue to numeric, handling potential missing values
mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")
mass_values.loc[mass_values["ArrayIndex"] > 262, "ActualValue"] = mass_values.loc[mass_values["ArrayIndex"] == 262, "ActualValue"].values[0]
# Read angle data
angle_data = pd.read_csv(
    angle_file_path,
    sep=r"\s+",
    header=None,
    names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2", 
           "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
    engine="python"
)

# Extract array index from ArrayName and keep only ventilA_stopinje_cas rows
angle_data = angle_data[angle_data["ArrayName"].str.contains("ventilA_stopinje_cas", na=False)]
angle_data["ArrayIndex"] = angle_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)

# Keep only relevant columns and sort by ArrayIndex
angle_values = angle_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
# Convert ActualValue to numeric, handling potential missing values
angle_values["ActualValue"] = pd.to_numeric(angle_values["ActualValue"], errors="coerce")

# Create time axis (100ms per index)
mass_values["Time"] = mass_values["ArrayIndex"] * 0.1  # pretvori v sec
angle_values["Time"] = angle_values["ArrayIndex"] * 0.1  

# Povezetek 
print(f"Mass data points: {len(mass_values)}")
print(f"Angle data points: {len(angle_values)}")
print("\nMass data sample:")
print(mass_values.head())
print("\nAngle data sample:")
print(angle_values.head())

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot mass vs time
ax1.plot(mass_values["Time"], mass_values["ActualValue"], 'b-', linewidth=2)
ax1.set_ylabel('masa [g]', fontsize=12)
ax1.set_title('masa m(t)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot angle vs time
ax2.plot(angle_values["Time"], angle_values["ActualValue"], 'r-', linewidth=2)
ax2.set_xlabel('cas (sec)', fontsize=12)
ax2.set_ylabel('Kot (deg)', fontsize=12)
ax2.set_title('Kot phi(t)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Save the plot
plt.savefig('valve_time_series.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("\n over copmlete'")