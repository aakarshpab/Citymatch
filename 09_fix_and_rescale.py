import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("master_dataset_v3.csv", encoding='latin1')
print(f"Loaded {len(df)} neighbourhoods.\n")

# ══════════════════════════════════════════════════════════════
# FIX 1 — Percentile cap on amenity counts
# ══════════════════════════════════════════════════════════════
# OSMnx counts are noisy — small clinics, dispensaries, and
# health posts all get tagged as "hospitals" in OpenStreetMap.
# Capping at the 95th percentile prevents a single outlier
# (Najafgarh's 17 hospitals) from dominating the scaled score.
# This is standard practice for count-based geospatial features.

cap_cols = ['num_hospitals', 'num_schools', 'num_parks', 'num_metro_stations']
print("=== Percentile Caps Applied ===")
for col in cap_cols:
    cap = df[col].quantile(0.95)
    before_max = df[col].max()
    df[col] = df[col].clip(upper=cap)
    after_max = df[col].max()
    print(f"  {col:<25} 95th pct cap = {cap:.1f}  |  max before: {before_max} → after: {after_max:.1f}")

# ══════════════════════════════════════════════════════════════
# FIX 2 — Compress PM25_Winter variance
# ══════════════════════════════════════════════════════════════
# PM25_Winter only ranges from 93.1 to 106.2 — a spread of 13 µg/m³.
# After MinMax scaling this becomes 0.0 to 1.0, but the real-world
# difference is tiny (all neighbourhoods are in the "Very Poor" AQI band).
# Using log scaling compresses the dominance of this near-uniform feature
# so it doesn't give perfect scores to the lowest-pollution outlier.

df['PM25_Winter_log'] = np.log1p(df['PM25_Winter'])
df['PM25_Monsoon_log'] = np.log1p(df['PM25_Monsoon'])

print("\n=== PM25 Variance Check (log-transformed) ===")
print(f"  PM25_Winter  — original std: {df['PM25_Winter'].std():.3f} | "
      f"log std: {df['PM25_Winter_log'].std():.4f}")

# ══════════════════════════════════════════════════════════════
# FIX 3 — Regenerate ALL scaled columns fresh
# ══════════════════════════════════════════════════════════════
# Drop all old scaled columns to avoid stale values
df = df.drop(columns=[c for c in df.columns if c.endswith('_scaled')], errors='ignore')

# Scale using log-transformed PM25, capped amenities, and real Traffic_Index
features_to_scale = {
    'PM25_Winter_log':      'PM25_Winter_scaled',
    'PM25_Monsoon_log':     'PM25_Monsoon_scaled',
    'Traffic_Index':        'Traffic_Index_scaled',
    'num_schools':          'num_schools_scaled',
    'num_hospitals':        'num_hospitals_scaled',
    'num_parks':            'num_parks_scaled',
    'num_metro_stations':   'num_metro_stations_scaled',
    'Rent_Price':           'Rent_Price_scaled',
}

scaler = MinMaxScaler()
raw_cols    = list(features_to_scale.keys())
scaled_cols = list(features_to_scale.values())

scaled_vals = scaler.fit_transform(df[raw_cols])
for i, scaled_name in enumerate(scaled_cols):
    df[scaled_name] = scaled_vals[:, i].round(6)

print("\n=== Scaled columns regenerated ===")
for raw, scaled in features_to_scale.items():
    print(f"  {raw:<28} → {scaled}")

# ══════════════════════════════════════════════════════════════
# SANITY CHECK — Najafgarh profile after fixes
# ══════════════════════════════════════════════════════════════
print("\n=== Najafgarh Profile After Fixes ===")
check_cols = ['neighborhood'] + scaled_cols
naj = df[df['neighborhood'] == 'Najafgarh'][check_cols].T
print(naj.to_string())

print("\n=== Top 5 by hospitals_scaled (should no longer be Najafgarh #1) ===")
print(df[['neighborhood','num_hospitals','num_hospitals_scaled']]
      .sort_values('num_hospitals_scaled', ascending=False)
      .head(5).to_string(index=False))

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
# Drop the temporary log columns before saving
df = df.drop(columns=['PM25_Winter_log', 'PM25_Monsoon_log'], errors='ignore')

df.to_csv("master_dataset_v4.csv", index=False, encoding='utf-8')
print(f"\n✅ Saved → master_dataset_v4.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
print("\nNext step: Re-run clustering on master_dataset_v4.csv → clustered_results_v4.csv")
print("Then update citymatch_app.py to load clustered_results_v4.csv")