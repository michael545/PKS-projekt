import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mplcursors 

mass_file_path = r"C:\Users\micha\Automation\vnetil_karakterirstika.txt"
angle_file_path = r"C:\Users\micha\Automation\ventil_karakteristika2.txt"


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


mass_values = mass_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
# Convert ActualValue v num, kaj ce ni vrednosti?
mass_values["ActualValue"] = pd.to_numeric(mass_values["ActualValue"], errors="coerce")
mass_values.loc[mass_values["ArrayIndex"] > 262, "ActualValue"] = mass_values.loc[mass_values["ArrayIndex"] == 262, "ActualValue"].values[0]

mass_values["ActualValue"] = mass_values["ActualValue"] - 200


angle_data = pd.read_csv(
    angle_file_path,
    sep=r"\s+",
    header=None,
    names=["ArrayName", "DataType", "InitialValue", "ActualValue", "ActualValue2", 
           "Extra1", "Extra2", "Extra3", "Extra4", "Extra5"],
    engine="python"
)

angle_data = angle_data[angle_data["ArrayName"].str.contains("ventilA_stopinje_cas", na=False)]
angle_data["ArrayIndex"] = angle_data["ArrayName"].str.extract(r'\[(\d+)\]')[0].astype(int)


angle_values = angle_data[["ArrayIndex", "ActualValue"]].sort_values(by="ArrayIndex")
angle_values["ActualValue"] = pd.to_numeric(angle_values["ActualValue"], errors="coerce")

 #(100ms za index)
mass_values["Time"] = mass_values["ArrayIndex"] * 0.1  # pretvori v sec
angle_values["Time"] = angle_values["ArrayIndex"] * 0.1  

mass_values = mass_values.sort_values(by="Time")

# Izracunaj dm-je in dt-je
mass_values["dt"] = mass_values["Time"].diff()
mass_values["dm"] = mass_values["ActualValue"].diff()

#Numericno izracunaj odvod
mass_values["dm_dt"] = np.where(
    mass_values["dt"] != 0,
    mass_values["dm"] / mass_values["dt"],
    0
)
mass_values["dm_dt"] = mass_values["dm_dt"].fillna(0)  # Handle initial NaN

mass_values.loc[mass_values["Time"] <= 2.1, "dm_dt"] = 0

#Glajenje ostrine/ nepravilnosti
mass_values["dm_dt_smooth"] = gaussian_filter1d(
    mass_values["dm_dt"], 
    sigma=1.5, 
    mode='nearest'  #  boundary artifacti
)

angle_values["d_angle"] = angle_values["ActualValue"].diff()
angle_values["dt_angle"] = angle_values["Time"].diff()

# Calculate angular velocity (deg/s)
angle_values["angle_vel"] = angle_values["d_angle"] / angle_values["dt_angle"]

# Smooth angular velocity
sigma_angle = 2
angle_values["angle_vel_smooth"] = gaussian_filter1d(angle_values["angle_vel"], sigma_angle)

angle_values["actuator_signal"] = 0  # Default state

# Detect za ODPIORANJE/ZAPIRANJE 2s
VELOCITY_THRESHOLD = 0.5
opening_mask = (angle_values["angle_vel_smooth"] > VELOCITY_THRESHOLD) & (angle_values["Time"] > 2.2)
closing_mask = (angle_values["angle_vel_smooth"] < -VELOCITY_THRESHOLD) & (angle_values["Time"] > 2.2)

# Signali za aktuator
angle_values.loc[opening_mask, "actuator_signal"] = 1
angle_values.loc[closing_mask, "actuator_signal"] = -1

# Forceiraj  1 za 0-2.2s (override)
angle_values.loc[angle_values["Time"] <= 2.2, "actuator_signal"] = 1


opening_data = mass_values[(angle_values["actuator_signal"] == 1) & (mass_values["Time"] > 2.2)]
closing_data = mass_values[angle_values["actuator_signal"] == -1]


print(f"Mass data points: {len(mass_values)}")
print(f"Angle data points: {len(angle_values)}")
print("\nMass data sample:")
print(mass_values.head())
print("\nAngle data sample:")
print(angle_values.head())


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot masa
ax1.plot(mass_values["Time"], mass_values["ActualValue"], 'b-', linewidth=2)
ax1.set_ylabel('masa [g]')
ax1.set_title('masa m(t)')
ax1.grid(True)

# Plot odvod mase dm/dt
ax2.plot(mass_values["Time"], mass_values["dm_dt_smooth"], 'g-', linewidth=2)
ax2.set_ylabel('dm/dt [g/s]')
ax2.set_title('Odvod mase dm/dt(t)')
ax2.grid(True)

# Plot kot phi (t)
ax3.plot(angle_values["Time"], angle_values["ActualValue"], 'r-', linewidth=2)
ax3.set_xlabel('cas (sec)')
ax3.set_ylabel('Kot (deg)')
ax3.set_title('Kot phi(t)')
ax3.grid(True)

# Add cursor
cursor1 = mplcursors.cursor(fig1, hover=True)
@cursor1.connect("add")
def on_add1(sel):
    x, y = sel.target
    sel.annotation.set(text=f"X: {x:.2f} s\nY: {y:.2f}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------
# Figure 2: 2x2 detailed view
# ----------------------------------------------------------------------------------
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Mass plot
ax1.plot(mass_values["Time"], mass_values["ActualValue"], 'b-')
ax1.set(xlabel='čas (s)', ylabel='masa [g]', title='masa m(t)')
ax1.grid(True)

# Derivative plot
ax2.plot(mass_values["Time"], mass_values["dm_dt_smooth"], 'g-')
ax2.set(xlabel='čas (s)', ylabel='dm/dt [g/s]', title='masni pretok dm/dt(t)')
ax2.grid(True)

# Angle plot
ax3.plot(angle_values["Time"], angle_values["ActualValue"], 'r-')
ax3.set(xlabel='čas (s)', ylabel='Kot (deg)', title='Kot phi(t)')
ax3.grid(True)

# Actuator plot
ax4.step(angle_values["Time"], angle_values["actuator_signal"], 'k-', where='post')
ax4.set(xlabel='čas (s)', ylabel='State', title='Aktuatorji ventila',
       yticks=[-1, 0, 1], yticklabels=['Zapiranje (-1)', 'prosti tek (0)', 'Odpiranje (+1)'], 
       ylim=(-1.5, 1.5))
ax4.grid(True)

# Add cursor to Figure 2
cursor2 = mplcursors.cursor(fig2, hover=True)
@cursor2.connect("add")
def on_add2(sel):
    x, y = sel.target
    sel.annotation.set(text=f"X: {x:.2f} s\nY: {y:.2f}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.tight_layout()
plt.show()


# Add cursor to Figure 3
cursor3 = mplcursors.cursor(fig3, hover=True)
@cursor3.connect("add")
def on_add3(sel):
    x, y = sel.target
    sel.annotation.set(text=f"X: {x:.2f} s\nY: {y:.2f}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.tight_layout()
plt.show()