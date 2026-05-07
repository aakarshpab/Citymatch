import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the Clustered Data (Output of Step 3)
df = pd.read_csv("clustered_results.csv")

# 2. Define the Input Features (The things a user cares about)
features = [
    'PM25_Winter',      # Lower is better
    'Traffic_Index',    # Lower is better
    'Rent_Price',       # Lower is better
    'num_schools',      # Higher is better
    'num_hospitals',    # Higher is better
    'num_metro_stations'# Higher is better
]

# --- THE HARD PART: HANDLING "LOWER IS BETTER" ---
# Cosine Similarity thinks "Higher Number = Better Match".
# But for Rent/Pollution, "Higher Number = Worse".
# So we must INVERT these values (1 - normalized_value) before matching.

# Create a temporary normalized dataframe just for calculation
scaler = MinMaxScaler()
norm_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Invert "Bad" things (Pollution, Traffic, Rent) so 1.0 becomes "Best" (Low)
norm_df['PM25_Winter'] = 1 - norm_df['PM25_Winter']
norm_df['Traffic_Index'] = 1 - norm_df['Traffic_Index']
norm_df['Rent_Price'] = 1 - norm_df['Rent_Price']

# 3. The Recommendation Function
def get_recommendations(user_preferences, top_n=5):
    """
    user_preferences: A list of 6 numbers between 0 and 1 representing importance.
    [Clean_Air, Low_Traffic, Low_Rent, Schools, Hospitals, Metro]
    """
    
    # Convert user list to a 2D array (1 row)
    user_vector = np.array(user_preferences).reshape(1, -1)
    
    # Calculate Cosine Similarity against ALL neighborhoods
    # Result is a list of scores (0% to 100% match)
    similarity_scores = cosine_similarity(user_vector, norm_df)[0]
    
    # Add scores to the original dataframe
    df['Match_Score'] = similarity_scores
    
    # Sort by Score (Highest first) and return top N
    return df.sort_values(by='Match_Score', ascending=False).head(top_n)

# --- TEST AREA (Simulating a User) ---
print("--- TESTING CITYMATCH RECOMMENDER ---\n")

# Simulation 1: The "Broke Student"
# Wants: CHEAP Rent (High Priority), Metro (High Priority). Doesn't care about pollution.
# Vector: [Air=0.1, Traffic=0.1, Cheap_Rent=0.9, Schools=0, Hospitals=0.1, Metro=0.9]
student_pref = [0.1, 0.1, 0.9, 0.0, 0.1, 0.9]

print("CASE 1: The Broke Student (Prioritizes Cheap Rent + Metro)")
results = get_recommendations(student_pref)
print(results[['neighborhood', 'Rent_Price', 'PM25_Winter', 'Match_Score']])

print("\n" + "-"*50 + "\n")

# Simulation 2: The "Health-Conscious Family"
# Wants: CLEAN Air (Max Priority), Schools, Hospitals. Budget is flexible.
# Vector: [Air=1.0, Traffic=0.8, Cheap_Rent=0.2, Schools=0.9, Hospitals=0.9, Metro=0.5]
family_pref = [1.0, 0.8, 0.2, 0.9, 0.9, 0.5]

print("CASE 2: The Healthy Family (Prioritizes Clean Air + Schools)")
results = get_recommendations(family_pref)
print(results[['neighborhood', 'Rent_Price', 'PM25_Winter', 'Match_Score']])