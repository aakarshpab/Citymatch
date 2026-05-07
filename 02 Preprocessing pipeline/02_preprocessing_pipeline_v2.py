import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# STEP 1: Load Files
# ─────────────────────────────────────────────
print("Loading data...")
neighborhoods = pd.read_csv("data_geospatial_amenities.csv")   # From OSMnx script
stations       = pd.read_csv("data_pollution_profiles.csv")    # From Step 1 script

# ─────────────────────────────────────────────
# STEP 2: Merge
# ─────────────────────────────────────────────
print("Merging data...")
merged_df = pd.merge(
    neighborhoods,
    stations,
    left_on="nearest_station",
    right_on="Station_Name",
    how="left"
)

# ─────────────────────────────────────────────
# STEP 3: Fill Rent_Price from 2 BHK midpoint
# ─────────────────────────────────────────────
# Each rent column contains a range like "25,000 – 38,000" or "1.5L – 2.5L"
# Strategy: Use 2 BHK midpoint (min + max) / 2 as the representative rent price.
# Rationale: 2 BHK is the most commonly rented flat type in Delhi, making it
# the best single proxy for neighbourhood-level rent. The midpoint is used to
# avoid bias toward either the lower or upper bound of the range.

def to_number(s: str) -> float:
    """Convert a rent string like '25,000' or '1.5L' to a float."""
    s = s.strip().replace(',', '')
    s = re.sub(r'^[^0-9.]+', '', s)   # strip any stray non-numeric prefix
    if 'L' in s.upper():
        return float(s.upper().replace('L', '')) * 100_000
    return float(s)

def parse_rent_midpoint(val: str) -> float | None:
    """
    Parse a rent range string and return the midpoint.
    Separator is the latin1 en-dash character (0x96).
    Returns None if parsing fails so nulls are visible and fixable.
    """
    try:
        parts = str(val).strip().split('\x96')   # en-dash used in source data
        if len(parts) == 2:
            return (to_number(parts[0]) + to_number(parts[1])) / 2
    except (ValueError, IndexError):
        pass
    return None

print("Filling Rent_Price from 2 BHK midpoint...")
merged_df['Rent_Price'] = merged_df['Rent 2 BHK'].apply(parse_rent_midpoint)

# Warn if any rows failed to parse
null_rents = merged_df['Rent_Price'].isnull().sum()
if null_rents > 0:
    print(f"  WARNING: {null_rents} rows could not be parsed — check 'Rent 2 BHK' column:")
    print(merged_df.loc[merged_df['Rent_Price'].isnull(), ['neighborhood', 'Rent 2 BHK']])
else:
    print(f"  All {len(merged_df)} rows filled successfully.")

# ─────────────────────────────────────────────
# STEP 4: Fill missing numeric values with column means
# ─────────────────────────────────────────────
print("Filling missing numeric values...")
numeric_cols = ['PM25_Annual', 'PM25_Winter', 'PM25_Monsoon', 'Traffic_Index']
for col in numeric_cols:
    merged_df[col].fillna(merged_df[col].mean(), inplace=True)

# ─────────────────────────────────────────────
# STEP 5: Normalize features (scale 0–1) for the model
# ─────────────────────────────────────────────
print("Scaling features...")
features_to_scale = [
    'PM25_Winter', 'PM25_Monsoon', 'Traffic_Index',
    'num_schools', 'num_hospitals', 'num_parks',
    'Rent_Price'
]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_df[features_to_scale])

for i, col in enumerate(features_to_scale):
    merged_df[f'{col}_scaled'] = scaled_data[:, i]

# ─────────────────────────────────────────────
# STEP 6: Save
# ─────────────────────────────────────────────
output_path = "master_dataset_v2.csv"
merged_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nSUCCESS: '{output_path}' created.")
print(f"  Rows: {len(merged_df)}")
print(f"  Rent_Price — min: ₹{merged_df['Rent_Price'].min():,.0f} | "
      f"median: ₹{merged_df['Rent_Price'].median():,.0f} | "
      f"max: ₹{merged_df['Rent_Price'].max():,.0f}")