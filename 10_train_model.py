import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# STEP 1: Load Data
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("clustered_results_v5.csv")

# Exclude known peri-urban outliers
EXCLUDE = ['Najafgarh', 'Narela', 'Bawana']
df = df[~df['neighborhood'].isin(EXCLUDE)].reset_index(drop=True)
print(f"Training on {len(df)} neighbourhoods (excluded: {EXCLUDE})\n")

# ══════════════════════════════════════════════════════════════
# STEP 2: Define Features and Target
# ══════════════════════════════════════════════════════════════
# These are the ORIGINAL (non-scaled) features.
# The model learns which raw factors drive rent.
FEATURES = [
    'PM25_Winter',          # air quality
    'PM25_Monsoon',
    'Traffic_Index',        # congestion
    'num_schools',          # amenities
    'num_hospitals',
    'num_metro_stations',
    'num_parks',
    'distance_to_station_km',
    'latitude',             # location acts as a proxy for centrality
    'longitude',
]
TARGET = 'Rent_Price'

X = df[FEATURES]
y = df[TARGET]

print("=== Feature Summary ===")
for f in FEATURES:
    print(f"  {f:<30} mean={X[f].mean():.2f}  std={X[f].std():.2f}")

# ══════════════════════════════════════════════════════════════
# STEP 3: Compare Multiple Models
# ══════════════════════════════════════════════════════════════
# With only 70 rows, we use 5-fold cross-validation instead of
# a train/test split to get reliable performance estimates.

print("\n=== Model Comparison (5-Fold CV) ===")

models = {
    'Ridge Regression':        Ridge(alpha=10),
    'Random Forest':           RandomForestRegressor(n_estimators=200, max_depth=5,
                                                     min_samples_leaf=3, random_state=42),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                          learning_rate=0.05, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    r2_scores  = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mse_scores = cross_val_score(model, X, y, cv=kf,
                                 scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    results[name] = {
        'R2_mean':   r2_scores.mean(),
        'R2_std':    r2_scores.std(),
        'RMSE_mean': rmse_scores.mean(),
        'RMSE_std':  rmse_scores.std(),
    }
    print(f"\n  {name}")
    print(f"    R²   = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
    print(f"    RMSE = ₹{rmse_scores.mean():,.0f} ± ₹{rmse_scores.std():,.0f}")

# ══════════════════════════════════════════════════════════════
# STEP 4: Train Best Model on Full Data
# ══════════════════════════════════════════════════════════════
# Random Forest generally wins on tabular data with mixed features.
# Change this if Ridge or GB scores higher in your run.
BEST_MODEL_NAME = 'Random Forest'
best_model = models[BEST_MODEL_NAME]
best_model.fit(X, y)

# Full-data predictions for dataset enrichment
df['Predicted_Rent'] = best_model.predict(X).round(0)

# ── Value Score ────────────────────────────────────────────
# How much more (or less) do you get vs what you pay?
# Positive = undervalued (hidden gem)
# Negative = overpriced
df['Rent_Gap']    = df['Predicted_Rent'] - df['Rent_Price']
df['Value_Score'] = (df['Rent_Gap'] / df['Rent_Price'] * 100).round(1)

print(f"\n\n=== Full-Data Model Performance ({BEST_MODEL_NAME}) ===")
y_pred = df['Predicted_Rent']
print(f"  R²   = {r2_score(y, y_pred):.3f}")
print(f"  RMSE = ₹{np.sqrt(mean_squared_error(y, y_pred)):,.0f}")

# ══════════════════════════════════════════════════════════════
# STEP 5: Feature Importance
# ══════════════════════════════════════════════════════════════
print("\n=== Feature Importance (What drives rent?) ===")
importance_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<30} {bar} {row['Importance']:.3f}")

# ══════════════════════════════════════════════════════════════
# STEP 6: Hidden Gems — Undervalued Neighbourhoods
# ══════════════════════════════════════════════════════════════
print("\n=== Top Hidden Gems (Undervalued Neighbourhoods) ===")
print("(Predicted rent much higher than actual → you get more than you pay for)\n")

gems = df[df['Value_Score'] > 0][
    ['neighborhood', 'Rent_Price', 'Predicted_Rent', 'Value_Score', 'Cluster_ID']
].sort_values('Value_Score', ascending=False).head(10)

gems['Rent_Price']      = gems['Rent_Price'].apply(lambda x: f"₹{int(x):,}")
gems['Predicted_Rent']  = gems['Predicted_Rent'].apply(lambda x: f"₹{int(x):,}")
gems['Value_Score']     = gems['Value_Score'].apply(lambda x: f"+{x:.0f}%")
print(gems.to_string(index=False))

print("\n=== Most Overpriced Neighbourhoods ===")
print("(Actual rent higher than predicted → paying a premium)\n")

overpriced = df[df['Value_Score'] < 0][
    ['neighborhood', 'Rent_Price', 'Predicted_Rent', 'Value_Score', 'Cluster_ID']
].sort_values('Value_Score').head(5)

overpriced['Rent_Price']     = overpriced['Rent_Price'].apply(lambda x: f"₹{int(x):,}")
overpriced['Predicted_Rent'] = overpriced['Predicted_Rent'].apply(lambda x: f"₹{int(x):,}")
overpriced['Value_Score']    = overpriced['Value_Score'].apply(lambda x: f"{x:.0f}%")
print(overpriced.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# STEP 7: Label Clusters Meaningfully
# ══════════════════════════════════════════════════════════════
cluster_profiles = df.groupby('Cluster_ID').agg(
    Avg_Rent        = ('Rent_Price',         'mean'),
    Avg_PM25        = ('PM25_Winter',        'mean'),
    Avg_Traffic     = ('Traffic_Index',      'mean'),
    Avg_Schools     = ('num_schools',        'mean'),
    Avg_Hospitals   = ('num_hospitals',      'mean'),
    Avg_Metro       = ('num_metro_stations', 'mean'),
    Count           = ('neighborhood',       'count')
).round(1)

print("\n=== Cluster Profiles ===")
print(cluster_profiles.to_string())

# Auto-label clusters based on rent level
# You can manually override these labels after reviewing
rent_order = cluster_profiles['Avg_Rent'].rank()
label_map  = {}
for cid, rank in rent_order.items():
    if rank == 1:   label_map[cid] = "Budget Peripheral"
    elif rank == 2: label_map[cid] = "Mid-range Suburban"
    elif rank == 3: label_map[cid] = "Well-connected Mid-tier"
    else:           label_map[cid] = "Premium Central"

df['Cluster_Label'] = df['Cluster_ID'].map(label_map)

print("\n=== Cluster Labels ===")
for cid, label in label_map.items():
    members = df[df['Cluster_ID'] == cid]['neighborhood'].tolist()
    print(f"\n  Cluster {cid} — '{label}'")
    print(f"  Members: {', '.join(members)}")

# ══════════════════════════════════════════════════════════════
# STEP 8: Save model + enriched dataset
# ══════════════════════════════════════════════════════════════
with open("citymatch_model.pkl", "wb") as f:
    pickle.dump({
        'model':        best_model,
        'features':     FEATURES,
        'model_name':   BEST_MODEL_NAME,
        'cluster_labels': label_map,
        'r2':           r2_score(y, y_pred),
        'rmse':         np.sqrt(mean_squared_error(y, y_pred)),
    }, f)

df.to_csv("master_dataset_v5.csv", index=False, encoding='utf-8')

print(f"\n{'='*55}")
print(f"  ✅ Model saved     → citymatch_model.pkl")
print(f"  ✅ Dataset saved   → master_dataset_v5.csv")
print(f"  New columns added  → Predicted_Rent, Rent_Gap,")
print(f"                        Value_Score, Cluster_Label")
print(f"\n  Next: Re-run clustering → clustered_results_v5.csv")
print(f"  Then: Update citymatch_app.py to use v5 + model.pkl")
print(f"{'='*55}")

# ══════════════════════════════════════════════════════════════
# STEP 9: Visualise Predicted vs Actual Rent
# ══════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')

colors = ['#22c55e' if v > 0 else '#ef4444'
          for v in (df['Predicted_Rent'] - df['Rent_Price'])]

plt.scatter(df['Rent_Price'], df['Predicted_Rent'],
            c=colors, alpha=0.8, s=80, edgecolors='#1f2937', linewidth=0.5)

# Perfect prediction line
max_rent = max(df['Rent_Price'].max(), df['Predicted_Rent'].max())
plt.plot([0, max_rent], [0, max_rent], '--', color='#6b7280',
         linewidth=1.5, label='Perfect Prediction')

# Label top gems
for _, row in df.nlargest(5, 'Value_Score').iterrows():
    plt.annotate(row['neighborhood'],
                 (row['Rent_Price'], row['Predicted_Rent']),
                 fontsize=7, color='#86efac',
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Actual Rent (₹)', color='#9ca3af')
plt.ylabel('Predicted Rent (₹)', color='#9ca3af')
plt.title('CityMatch — Rent Prediction Model\nGreen = Undervalued | Red = Overpriced',
          color='#f3f4f6', pad=15)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("rent_prediction_plot.png", dpi=150, bbox_inches='tight',
            facecolor='#0d0f14')
plt.show()
print("  Plot saved → rent_prediction_plot.png")