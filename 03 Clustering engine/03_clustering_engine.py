import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# 1. Load Data
df = pd.read_csv("master_dataset_v1.csv")
print(f"Loaded {len(df)} neighborhoods.")

# 2. Select Features for Clustering
# We use the SCALED columns we created in Step 2.
# These are the factors that determine "similarity".
features = [
    'PM25_Winter_scaled', 
    'PM25_Monsoon_scaled', 
    'Traffic_Index_scaled', 
    'Rent_Price_scaled',
    'num_schools_scaled',
    'num_hospitals_scaled'
]

X = df[features]

# 3. Determine Optimal Clusters (The "Elbow Method")
# We test K=2 to K=10 to see which fits best.
print("\n--- FINDING OPTIMAL K (Elbow Method) ---")
inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    print(f"K={k}: Silhouette Score = {round(silhouette_score(X, kmeans.labels_), 3)}")

# 4. Apply K-Means (Let's assume K=4 is best based on typical city data)
# You can change n_clusters based on the scores above (look for highest Silhouette)
BEST_K = 4
print(f"\n--- RUNNING K-MEANS with K={BEST_K} ---")

kmeans = KMeans(n_clusters=BEST_K, random_state=42)
df['Cluster_ID'] = kmeans.fit_predict(X)

# 5. Analyze the Clusters (The "Interpretation")
# We calculate the average of the ORIGINAL (non-scaled) columns to understand what the groups mean.
analysis_cols = ['PM25_Winter', 'Rent_Price', 'Traffic_Index', 'num_schools', 'num_hospitals']
cluster_summary = df.groupby('Cluster_ID')[analysis_cols].mean().round(1)

print("\nCluster Profiles (What do these groups represent?):")
print(cluster_summary)

# 6. Save Results
df.to_csv("clustered_results.csv", index=False)
print("\nSaved to 'clustered_results.csv'.")
print("Next Step: Visualization.")

# Optional: Plot the Elbow Curve to show the examiner
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()