import matplotlib.pyplot as plt

from models.darzi import calculate_darzi_pga


# =========================================================
# 1. Distance Validation
# =========================================================

magnitude = 6.0
site_class = 2
mechanism = "reverse"

distances = [5, 10, 20, 50, 100, 150, 200]

pga_values = []

for distance in distances:

    result = calculate_darzi_pga(
        magnitude=magnitude,
        hypocentral_distance_km=distance,
        site_class=site_class,
        mechanism=mechanism
    )

    pga_values.append(result.pga_g)


print("\n=== Distance Validation ===")
print("-" * 45)

for r, pga in zip(distances, pga_values):
    print(
        f"R = {r:6.1f} km | "
        f"PGA = {pga:.6f} g"
    )


plt.figure()

plt.plot(
    distances,
    pga_values,
    marker="o"
)

plt.xlabel("Hypocentral Distance (km)")
plt.ylabel("PGA (g)")
plt.title("Darzi PGA vs Hypocentral Distance")
plt.grid(True)

plt.show()

# =========================================================
# 2. Magnitude Validation
# =========================================================

magnitudes = [
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.4
]

distance = 30.0

pga_values = []

for magnitude in magnitudes:

    result = calculate_darzi_pga(
        magnitude=magnitude,
        hypocentral_distance_km=distance,
        site_class=2,
        mechanism="reverse"
    )

    pga_values.append(result.pga_g)


print("\n=== Magnitude Validation ===")
print("-" * 45)

for mw, pga in zip(magnitudes, pga_values):
    print(
        f"Mw = {mw:.1f} | "
        f"PGA = {pga:.6f} g"
    )


plt.figure()

plt.plot(
    magnitudes,
    pga_values,
    marker="o"
)

plt.xlabel("Moment Magnitude (Mw)")
plt.ylabel("PGA (g)")
plt.title("Darzi PGA vs Magnitude")
plt.grid(True)

plt.show()

# =========================================================
# 3. Site Class Validation
# =========================================================

site_classes = [1, 2, 3]

pga_values = []

for site_class in site_classes:

    result = calculate_darzi_pga(
        magnitude=6.0,
        hypocentral_distance_km=30.0,
        site_class=site_class,
        mechanism="reverse"
    )

    pga_values.append(result.pga_g)


print("\n=== Site Class Validation ===")
print("-" * 45)

for site_class, pga in zip(site_classes, pga_values):
    print(
        f"Site Class = {site_class} | "
        f"PGA = {pga:.6f} g"
    )


plt.figure()

plt.bar(
    ["I", "II", "III"],
    pga_values
)

plt.xlabel("Site Class")
plt.ylabel("PGA (g)")
plt.title("Darzi PGA by Site Class")

plt.show()

# =========================================================
# 4. Style of Faulting Validation
# =========================================================

mechanisms = [
    "reverse",
    "strike-slip",
    "normal"
]

pga_values = []

for mechanism in mechanisms:

    result = calculate_darzi_pga(
        magnitude=6.0,
        hypocentral_distance_km=30.0,
        site_class=2,
        mechanism=mechanism
    )

    pga_values.append(result.pga_g)


print("\n=== SoF Validation ===")
print("-" * 45)

for mechanism, pga in zip(mechanisms, pga_values):
    print(
        f"{mechanism:12s} | "
        f"PGA = {pga:.6f} g"
    )


plt.figure()

plt.bar(
    ["Reverse", "Strike-slip", "Normal"],
    pga_values
)

plt.xlabel("Style of Faulting")
plt.ylabel("PGA (g)")
plt.title("Darzi PGA by Style of Faulting")

plt.show()