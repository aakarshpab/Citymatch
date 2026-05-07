import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────
df = pd.read_csv("master_dataset_v4.csv")
print(f"Loaded {len(df)} neighborhoods.")
df = df[df['neighborhood'] != 'Najafgarh'].reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP 2: Ensure Rent_Price_scaled exists
# ─────────────────────────────────────────────
# master_dataset_v2.csv has Rent_Price (filled in the previous step) but
# may not have Rent_Price_scaled if it was missing from the saved output.
# We regenerate all scaled columns here to guarantee consistency.

features_to_scale = [
    'PM25_Winter', 'PM25_Monsoon', 'Traffic_Index',
    'num_schools', 'num_hospitals', 'num_parks',
    'Rent_Price'
]

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[features_to_scale])

for i, col in enumerate(features_to_scale):
    df[f'{col}_scaled'] = scaled_values[:, i]

print("Scaled columns (re)generated.")

# ─────────────────────────────────────────────
# STEP 3: Select Features for Clustering
# ─────────────────────────────────────────────
# Using scaled columns so no single feature dominates due to its unit/magnitude.
features = [
    'PM25_Winter_scaled',
    'PM25_Monsoon_scaled',
    'Traffic_Index_scaled',
    'Rent_Price_scaled',       # scaled version of the filled Rent_Price column
    'num_schools_scaled',
    'num_hospitals_scaled'
]

X = df[features]

# ─────────────────────────────────────────────
# STEP 4: Find Optimal K (Elbow Method)
# ─────────────────────────────────────────────
print("\n--- FINDING OPTIMAL K (Elbow Method) ---")
inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette Score = {round(score, 3)}")

# ─────────────────────────────────────────────
# STEP 5: Apply K-Means with Best K
# ─────────────────────────────────────────────
# Change BEST_K based on the highest Silhouette Score printed above.
BEST_K = 4
print(f"\n--- RUNNING K-MEANS with K={BEST_K} ---")

kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X)

# ─────────────────────────────────────────────
# STEP 6: Analyse Clusters using original (unscaled) columns
# ─────────────────────────────────────────────
# NOTE: analysis_cols uses Rent_Price (the original ₹ values), NOT Rent_Price_scaled.
# This makes the cluster profiles human-readable (e.g. avg rent ₹35,000 vs ₹1,20,000).
analysis_cols = ['PM25_Winter', 'Rent_Price', 'Traffic_Index', 'num_schools', 'num_hospitals']
cluster_summary = df.groupby('Cluster_ID')[analysis_cols].mean().round(1)

print("\nCluster Profiles (original scale — human readable):")
print(cluster_summary)

# ─────────────────────────────────────────────
# STEP 7: Save Results
# ─────────────────────────────────────────────
df.to_csv("clustered_results_v4.csv", index=False)
print("\nSaved to 'clustered_results_v4.csv'.")

# ─────────────────────────────────────────────
# STEP 8: Plot Elbow Curve
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method — Optimal K')
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150)
plt.show()
print("Elbow curve saved to 'elbow_curve.png'.")