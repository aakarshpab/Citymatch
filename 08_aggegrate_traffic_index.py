import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# STEP 1: Load the traffic log
# ==========================================
LOG_FILE    = "traffic_log.csv"
MASTER_FILE = "master_dataset_v2.csv"
OUTPUT_FILE = "master_dataset_v3.csv"

print("Loading traffic log...")
log = pd.read_csv(LOG_FILE)

total_runs  = log['Run_ID'].nunique()
total_reads = len(log)
print(f"  Runs completed : {total_runs}/16")
print(f"  Total readings : {total_reads}")

# Warn if any runs are missing
if total_runs < 16:
    print(f"\n  ⚠ WARNING: Only {total_runs}/16 runs found.")
    print(  "    Averaging will still work but Traffic_Index will be")
    print(  "    less representative. Ideally complete all 16 runs.\n")

# ==========================================
# STEP 2: Check coverage per slot
# ==========================================
print("\nReadings per time slot:")
print(log.groupby('Slot')['Neighborhood'].count().to_string())

# ==========================================
# STEP 3: Average Traffic_Index per neighbourhood
# ==========================================
print("\nAggregating Traffic_Index (mean across all runs)...")

traffic_final = (
    log.groupby('Neighborhood')
       .agg(
           Traffic_Index        = ('Traffic_Index',         'mean'),
           Morning_Peak_Avg     = ('Traffic_Index',         lambda x: x[log.loc[x.index, 'Slot'] == 'Morning_Peak'].mean()),
           Afternoon_Avg        = ('Traffic_Index',         lambda x: x[log.loc[x.index, 'Slot'] == 'Afternoon'].mean()),
           Evening_Peak_Avg     = ('Traffic_Index',         lambda x: x[log.loc[x.index, 'Slot'] == 'Evening_Peak'].mean()),
           Night_Avg            = ('Traffic_Index',         lambda x: x[log.loc[x.index, 'Slot'] == 'Night'].mean()),
           Avg_Confidence       = ('Congestion_Confidence', 'mean'),
           Readings_Count       = ('Traffic_Index',         'count'),
       )
       .round(3)
       .reset_index()
)

print(f"\n  Neighbourhoods aggregated: {len(traffic_final)}")
print("\nSample (top 10 by Traffic_Index):")
print(traffic_final.nlargest(10, 'Traffic_Index')[
    ['Neighborhood', 'Traffic_Index', 'Morning_Peak_Avg', 'Evening_Peak_Avg']
].to_string(index=False))

# ==========================================
# STEP 4: Merge into master_dataset_v2
# ==========================================
print(f"\nMerging into {MASTER_FILE}...")
master = pd.read_csv(MASTER_FILE, encoding='utf-8')

# Drop old Traffic_Index and its scaled column before replacing
master = master.drop(columns=[c for c in master.columns if 'Traffic_Index' in c], errors='ignore')

master = pd.merge(
    master,
    traffic_final[['Neighborhood', 'Traffic_Index',
                   'Morning_Peak_Avg', 'Afternoon_Avg',
                   'Evening_Peak_Avg', 'Night_Avg',
                   'Avg_Confidence', 'Readings_Count']],
    left_on='neighborhood',
    right_on='Neighborhood',
    how='left'
).drop(columns=['Neighborhood'])

# Warn about any unmatched neighbourhoods
unmatched = master[master['Traffic_Index'].isna()]['neighborhood'].tolist()
if unmatched:
    print(f"\n  ⚠ WARNING: {len(unmatched)} neighbourhoods had no traffic data:")
    for n in unmatched:
        print(f"    - {n}")
    print("  Filling with column mean.")
    master['Traffic_Index'] = master['Traffic_Index'].fillna(master['Traffic_Index'].mean())

# ==========================================
# STEP 5: Re-scale all features consistently
# ==========================================
print("\nRe-scaling all features...")

features_to_scale = [
    'PM25_Winter', 'PM25_Monsoon', 'Traffic_Index',
    'num_schools', 'num_hospitals', 'num_parks',
    'Rent_Price'
]

# Drop old scaled columns and regenerate fresh
master = master.drop(columns=[c for c in master.columns if c.endswith('_scaled')], errors='ignore')

scaler      = MinMaxScaler()
scaled_vals = scaler.fit_transform(master[features_to_scale])

for i, col in enumerate(features_to_scale):
    master[f'{col}_scaled'] = scaled_vals[:, i].round(6)

# ==========================================
# STEP 6: Save master_dataset_v3.csv
# ==========================================
master.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\n{'='*55}")
print(f"  ✅ SUCCESS: '{OUTPUT_FILE}' created.")
print(f"  Rows    : {len(master)}")
print(f"  Columns : {len(master.columns)}")
print(f"\n  Traffic_Index stats:")
print(f"    Min    : {master['Traffic_Index'].min():.2f}")
print(f"    Median : {master['Traffic_Index'].median():.2f}")
print(f"    Max    : {master['Traffic_Index'].max():.2f}")
print(f"{'='*55}")
print(f"\n  Next step: Re-run clustering on master_dataset_v3.csv")