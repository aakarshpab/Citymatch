import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════════════════════════
# STEP 1: Load both datasets
# ══════════════════════════════════════════════════════════════
kaggle = pd.read_csv("June_8_data_metro_closest_stations.csv", encoding='latin1')
master = pd.read_csv("master_dataset_v4.csv", encoding='utf-8')

# Filter to 2BHK only — consistent with our existing methodology
kaggle_2bhk = kaggle[kaggle['bedrooms'] == 2].copy()
kaggle_2bhk['locality_lower'] = kaggle_2bhk['localityName'].str.lower().str.strip()

print(f"Kaggle 2BHK listings: {len(kaggle_2bhk)}")
print(f"Master neighbourhoods: {len(master)}\n")

# ══════════════════════════════════════════════════════════════
# STEP 2: Manual name mapping
# ══════════════════════════════════════════════════════════════
# Maps our neighbourhood names (lowercase) → Kaggle locality names (lowercase)
# Built from direct matches + fuzzy match review above

NAME_MAP = {
    # Direct matches (case differences only)
    'alaknanda':        'alaknanda',
    'anand vihar':      'anand vihar',
    'ashok vihar':      'ashok vihar',
    'bali nagar':       'bali nagar',
    'bawana':           'bawana',
    'civil lines':      'civil lines',
    'connaught place':  'connaught place',
    'defence colony':   'defence colony',
    'dilshad garden':   'dilshad garden',
    'east of kailash':  'east of kailash',
    'geeta colony':     'geeta colony',
    'greater kailash':  'greater kailash',
    'green park':       'green park',
    'hari nagar':       'hari nagar',
    'hauz khas':        'hauz khas',
    'ip extension':     'ip extension',
    'janakpuri':        'janakpuri',
    'jangpura':         'jangpura',
    'kalkaji':          'kalkaji',
    'kamla nagar':      'kamla nagar',
    'karol bagh':       'karol bagh',
    'kirti nagar':      'kirti nagar',
    'lajpat nagar':     'lajpat nagar',
    'laxmi nagar':      'laxmi nagar',
    'malviya nagar':    'malviya nagar',
    'mayur vihar':      'mayur vihar',
    'model town':       'model town',
    'najafgarh':        'najafgarh',
    'narela':           'narela',
    'palam':            'palam',
    'paschim vihar':    'paschim vihar',
    'patel nagar':      'patel nagar',
    'pitampura':        'pitampura',
    'preet vihar':      'preet vihar',
    'punjabi bagh':     'punjabi bagh',
    'rajouri garden':   'rajouri garden',
    'saket':            'saket',
    'sarita vihar':     'sarita vihar',
    'shahdara':         'shahdara',
    'shalimar bagh':    'shalimar bagh',
    'south extension':  'south extension',
    'tilak nagar':      'tilak nagar',
    'uttam nagar':      'uttam nagar',
    'vasant kunj':      'vasant kunj',
    'vasant vihar':     'vasant vihar',
    'vikas puri':       'vikas puri',
    'vivek vihar':      'vivek vihar',
    'yamuna vihar':     'yamuna vihar',

    # Fuzzy / variant matches (manually verified)
    'chhattarpur':      'chhatarpur',       # spelling variant
    'rk puram':         'r.k. puram',       # abbreviation
    'rohini':           'sector7 rohini',   # closest Rohini listing cluster
    'safdarjung':       'safdarjung enclave',
    'sunder nagar':     'sundar nagar',     # spelling variant
    'mukherjee nagar':  'dr mukherji nagar',
    'munirka':          'munirka dda flats',
    'moti bagh':        'moti nagar',       # adjacent locality
    'mehrauli':         'kishangarh mehrauli road',
    'nehru place':      'nehru place',      # try direct first
    'dwarka':           'dwarka mor',       # largest Dwarka cluster
    'gandhi nagar':     'ganesh nagar',     # adjacent
    'okhla':            'jasola',           # Okhla Phase area
    'daryaganj':        'paharganj',        # central old Delhi proxy
    'chandni chowk':    'chandni chowk',    # try direct
    'chanakyapuri':     'chanakyapuri',     # try direct

    # Coordinate-fallback candidates (will use lat/lon if no listing found)
    'adarsh nagar':     None,
    'alipur':           None,
    'badarpur':         None,
    'delhi cantt':      None,
    'kanjhawala':       None,
    'karawal nagar':    None,
    'kashmere gate':    None,
    'lodhi colony':     None,
    'mangolpuri':       None,
}

# ══════════════════════════════════════════════════════════════
# STEP 3: Aggregate rent per mapped locality
# ══════════════════════════════════════════════════════════════
# Build a lookup: locality_lower → median rent from Kaggle
kaggle_rent_lookup = (
    kaggle_2bhk.groupby('locality_lower')['price']
    .agg(median_rent='median', listing_count='count')
    .reset_index()
)

def get_rent(our_name):
    """Get median 2BHK rent for a neighbourhood using name mapping."""
    kaggle_name = NAME_MAP.get(our_name.lower())
    if kaggle_name is None:
        return None, 0
    row = kaggle_rent_lookup[kaggle_rent_lookup['locality_lower'] == kaggle_name]
    if len(row) == 0:
        return None, 0
    return float(row['median_rent'].values[0]), int(row['listing_count'].values[0])

# ══════════════════════════════════════════════════════════════
# STEP 4: Coordinate fallback for unmatched neighbourhoods
# ══════════════════════════════════════════════════════════════
# For neighbourhoods with no name match, find the nearest
# Kaggle listing by geographic distance and use its locality median.

def get_rent_by_coords(lat, lon, radius_deg=0.05):
    """Find median rent of listings within ~5km of a coordinate."""
    nearby = kaggle_2bhk[
        (abs(kaggle_2bhk['latitude']  - lat) < radius_deg) &
        (abs(kaggle_2bhk['longitude'] - lon) < radius_deg)
    ]
    if len(nearby) >= 3:
        return float(nearby['price'].median()), len(nearby)
    return None, 0

# ══════════════════════════════════════════════════════════════
# STEP 5: Build new Rent_Price column
# ══════════════════════════════════════════════════════════════
print("=== Rent Matching Results ===\n")

new_rents   = []
match_types = []

for _, row in master.iterrows():
    name = row['neighborhood']
    rent, count = get_rent(name)

    if rent is not None and count >= 3:
        new_rents.append(rent)
        match_types.append(f"name_match ({count} listings)")

    else:
        # Coordinate fallback
        rent, count = get_rent_by_coords(row['latitude'], row['longitude'])
        if rent is not None:
            new_rents.append(rent)
            match_types.append(f"coord_fallback ({count} listings)")
        else:
            # Last resort — keep original midpoint estimate
            new_rents.append(row['Rent_Price'])
            match_types.append("original_estimate")

master['Rent_Price_New']  = new_rents
master['Rent_Match_Type'] = match_types

# Print summary
print(f"{'Neighbourhood':<25} {'Old Rent':>10} {'New Rent':>10}  Match Type")
print("-" * 75)
for _, row in master.iterrows():
    diff = row['Rent_Price_New'] - row['Rent_Price']
    flag = " ⚠" if abs(diff) > 20000 else ""
    print(f"  {row['neighborhood']:<23} ₹{int(row['Rent_Price']):>8,} "
          f"₹{int(row['Rent_Price_New']):>8,}  {row['Rent_Match_Type']}{flag}")

print(f"\n{'='*55}")
match_counts = pd.Series(match_types).apply(
    lambda x: x.split('(')[0].strip()
).value_counts()
print("Match type summary:")
print(match_counts.to_string())

# ══════════════════════════════════════════════════════════════
# STEP 6: Replace old Rent_Price and rescale
# ══════════════════════════════════════════════════════════════
master['Rent_Price'] = master['Rent_Price_New']
master = master.drop(columns=['Rent_Price_New', 'Rent_Match_Type'])

# Drop old scaled columns and regenerate
master = master.drop(
    columns=[c for c in master.columns if c.endswith('_scaled')],
    errors='ignore'
)

features_to_scale = {
    'PM25_Winter':        'PM25_Winter_scaled',
    'PM25_Monsoon':       'PM25_Monsoon_scaled',
    'Traffic_Index':      'Traffic_Index_scaled',
    'num_schools':        'num_schools_scaled',
    'num_hospitals':      'num_hospitals_scaled',
    'num_parks':          'num_parks_scaled',
    'num_metro_stations': 'num_metro_stations_scaled',
    'Rent_Price':         'Rent_Price_scaled',
}

scaler     = MinMaxScaler()
raw_cols   = list(features_to_scale.keys())
scaled_arr = scaler.fit_transform(master[raw_cols])

for i, scaled_name in enumerate(features_to_scale.values()):
    master[scaled_name] = scaled_arr[:, i].round(6)

# ══════════════════════════════════════════════════════════════
# STEP 7: Save
# ══════════════════════════════════════════════════════════════
master.to_csv("master_dataset_v5.csv", index=False, encoding='utf-8')

print(f"\n✅ Saved → master_dataset_v5.csv")
print(f"\nNew Rent_Price stats:")
print(f"  Min    : ₹{master['Rent_Price'].min():,.0f}")
print(f"  Median : ₹{master['Rent_Price'].median():,.0f}")
print(f"  Mean   : ₹{master['Rent_Price'].mean():,.0f}")
print(f"  Max    : ₹{master['Rent_Price'].max():,.0f}")
print(f"\nNext steps:")
print(f"  1. Re-run clustering  → clustered_results_v5.csv")
print(f"  2. Re-run train_model.py on clustered_results_v5.csv")
print(f"  3. Expect significantly improved R² score")