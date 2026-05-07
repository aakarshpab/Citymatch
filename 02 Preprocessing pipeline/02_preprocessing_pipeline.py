import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load Files
neighborhoods = pd.read_csv("data_geospatial_amenities.csv") # From OSMnx script
stations = pd.read_csv("data_pollution_profiles.csv")      # From Step 1 script

# 2. Merge
print("Merging data...")
merged_df = pd.merge(neighborhoods, stations, left_on="nearest_station", right_on="Station_Name", how="left")

# 3. Add Placeholder for Rent (You must fill this manually later!)
if 'Rent_Price' not in merged_df.columns:
    merged_df['Rent_Price'] = 0 

# 4. Fill missing values (if any API calls failed) with column averages
numeric_cols = ['PM25_Annual', 'PM25_Winter', 'PM25_Monsoon', 'Traffic_Index']
for col in numeric_cols:
    merged_df[col].fillna(merged_df[col].mean(), inplace=True)

# 5. Normalize (Scale 0-1) for the Model
features_to_scale = ['PM25_Winter', 'PM25_Monsoon', 'Traffic_Index', 
                     'num_schools', 'num_hospitals', 'num_parks', 'Rent_Price']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_df[features_to_scale])

# Create "_scaled" columns
for i, col in enumerate(features_to_scale):
    merged_df[f'{col}_scaled'] = scaled_data[:, i]

# 6. Save
merged_df.to_csv("final_model_data.csv", index=False)
print("SUCCESS: 'final_model_data.csv' created.")
print("NEXT STEP: Open this file in Excel and fill in the 'Rent_Price' column.")