#Project Overview: Create a Model that trains engineering parameters (Synthetic Data) and predicts Failure modes. 

#Parameters to Train:
#1. Max Stress
#2. Max Strain
#3. Displacement
#4. Temperature
#5. Material Young's Modulus
#6. Load Magnitude
#7. Thickness

#Output Classes:
#0 - Safe
#1 - Yield Failure
#2 - Buckling
#3 - Fatigue Risk

import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

# ----------------------------
# 1. Generate input features
# ----------------------------
temperature = np.random.uniform(20, 300, n)              # °C
youngs_modulus = np.random.uniform(70e9, 210e9, n)       # Pa
load = np.random.uniform(1e3, 1e5, n)                    # N
thickness = np.random.uniform(0.001, 0.02, n)            # m

# Optional geometry assumptions
width = np.random.uniform(0.02, 0.1, n)                  # m
length = np.random.uniform(0.1, 1.0, n)                  # m

# Cross-sectional area
area = width * thickness

# ----------------------------
# 2. Material behavior
# ----------------------------
# Base yield strength at room temperature
base_yield_strength = np.random.uniform(180e6, 450e6, n)  # Pa

# Temperature reduces yield strength slightly
yield_strength = base_yield_strength * (1 - 0.0008 * (temperature - 20))
yield_strength = np.clip(yield_strength, 50e6, None)

# ----------------------------
# 3. Derived outputs
# ----------------------------
# Max stress
max_stress = load / area

# Max strain (Hooke's law)
max_strain = max_stress / youngs_modulus

# Simplified bending/axial displacement relation
# displacement grows with load and length, decreases with E and thickness
displacement = (load * length**3) / (youngs_modulus * width * thickness**3)
displacement = displacement * 1e-2  # scaling factor to keep values reasonable

# ----------------------------
# 4. Class labeling
# ----------------------------
displacement_threshold = 0.005  # 5 mm
fatigue_lower_ratio = 0.6
fatigue_upper_ratio = 1.0

output_class = np.zeros(n, dtype=int)  # default Safe

for i in range(n):
    if max_stress[i] > yield_strength[i]:
        output_class[i] = 1  # Yield Failure
    elif displacement[i] > displacement_threshold:
        output_class[i] = 2  # Buckling
    elif fatigue_lower_ratio * yield_strength[i] < max_stress[i] <= fatigue_upper_ratio * yield_strength[i]:
        output_class[i] = 3  # Fatigue Risk
    else:
        output_class[i] = 0  # Safe

# ----------------------------
# 5. Build dataframe
# ----------------------------
df = pd.DataFrame({
    "Temperature": temperature,
    "Material_Youngs_Modulus": youngs_modulus,
    "Load_Magnitude": load,
    "Thickness": thickness,
    "Max_Stress": max_stress,
    "Max_Strain": max_strain,
    "Displacement": displacement,
    "Yield_Strength": yield_strength,
    "Output_Class": output_class
})

print(df.head())
print(df["Output_Class"].value_counts())

df.to_csv("data/synthetic_to_train.csv", index=False)