import requests
import pandas as pd
import time
import sys
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 🔴 REPLACE THIS with your actual TomTom API Key
API_KEY = "8FYp2Btu2wX7ZQG4vaVOGywiRXQPpnH9" 

# List of Neighborhoods (Name: Lat, Lon)
# Ensure these match the coordinates in your Master Dataset
neighborhoods = {
    "Adarsh Nagar": (28.7161, 77.1706),
    "Alaknanda": (28.5255, 77.2471),
    "Alipur": (28.7973, 77.1331),
    "Anand Vihar": (28.6465, 77.3122),
    "Ashok Vihar": (28.6917, 77.1742),
    "Badarpur": (28.5034, 77.3047),
    "Bali Nagar": (28.6534, 77.1306),
    "Bawana": (28.7946, 77.0396),
    "Chanakyapuri": (28.5919, 77.1852),
    "Chandni Chowk": (28.6606, 77.2273),
    "Chhattarpur": (28.5038, 77.1818),
    "Civil Lines": (28.6814, 77.2229),
    "Connaught Place": (28.6315, 77.2167),
    "Daryaganj": (28.643, 77.2407),
    "Defence Colony": (28.5724, 77.2334),
    "Delhi Cantt": (28.5961, 77.1587),
    "Dilshad Garden": (28.6844, 77.3197),
    "Dwarka": (28.5823, 77.05),
    "East of Kailash": (28.5562, 77.2464),
    "Gandhi Nagar": (28.6652, 77.2711),
    "Geeta Colony": (28.6535, 77.2691),
    "Greater Kailash": (28.5482, 77.2326),
    "Green Park": (28.5589, 77.2028),
    "Hari Nagar": (28.6277, 77.1044),
    "Hauz Khas": (28.5494, 77.2001),
    "IP Extension": (28.6293, 77.3005),
    "Janakpuri": (28.6219, 77.0878),
    "Jangpura": (28.5828, 77.2435),
    "Kalkaji": (28.5414, 77.2515),
    "Kamla Nagar": (28.6806, 77.2023),
    "Kanjhawala": (28.7196, 77.0016),
    "Karawal Nagar": (28.7302, 77.2741),
    "Karol Bagh": (28.655, 77.1888),
    "Kashmere Gate": (28.6675, 77.2285),
    "Kirti Nagar": (28.6496, 77.1422),
    "Lajpat Nagar": (28.5677, 77.2433),
    "Laxmi Nagar": (28.6305, 77.2773),
    "Lodhi Colony": (28.5878, 77.2215),
    "Malviya Nagar": (28.5362, 77.2114),
    "Mangolpuri": (28.6946, 77.0722),
    "Mayur Vihar": (28.6046, 77.2911),
    "Mehrauli": (28.5204, 77.1804),
    "Model Town": (28.7027, 77.1939),
    "Moti Bagh": (28.5825, 77.1683),
    "Mukherjee Nagar": (28.7093, 77.2144),
    "Munirka": (28.5559, 77.1702),
    "Najafgarh": (28.6092, 76.9798),
    "Narela": (28.855, 77.091),
    "Nehru Place": (28.5492, 77.2517),
    "Okhla": (28.5649, 77.285),
    "Palam": (28.5866, 77.085),
    "Paschim Vihar": (28.6685, 77.0931),
    "Patel Nagar": (28.6508, 77.1652),
    "Pitampura": (28.7033, 77.1323),
    "Preet Vihar": (28.6387, 77.2974),
    "Punjabi Bagh": (28.6675, 77.1259),
    "Rajouri Garden": (28.6415, 77.1209),
    "RK Puram": (28.5653, 77.1747),
    "Rohini": (28.7041, 77.1025),
    "Safdarjung": (28.5619, 77.1969),
    "Saket": (28.5246, 77.2066),
    "Sarita Vihar": (28.529, 77.3027),
    "Shahdara": (28.6732, 77.286),
    "Shalimar Bagh": (28.7175, 77.1528),
    "South Extension": (28.5701, 77.2215),
    "Sunder Nagar": (28.6015, 77.2378),
    "Tilak Nagar": (28.6365, 77.0937),
    "Uttam Nagar": (28.6213, 77.056),
    "Vasant Kunj": (28.5293, 77.1519),
    "Vasant Vihar": (28.5606, 77.1627),
    "Vikas Puri": (28.6346, 77.0697),
    "Vivek Vihar": (28.6644, 77.3117),
    "Yamuna Vihar": (28.6931, 77.2694)
}

# ==========================================
# 2. THE FETCHING LOGIC
# ==========================================
print(f"--- STARTING TOMTOM TRAFFIC INGESTION ---")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Targets: {len(neighborhoods)} Neighborhoods\n")

results = []
count = 0

for name, (lat, lon) in neighborhoods.items():
    count += 1
    # Print progress bar
    sys.stdout.write(f"\rProcessing {count}/{len(neighborhoods)}: {name}...")
    sys.stdout.flush()
    
    try:
        # API Call: Flow Segment Data (Real-time speed on the nearest road)
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point={lat},{lon}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            flow = data['flowSegmentData']
            
            # Extract Core Metrics
            current_speed = flow['currentSpeed']      # Real-time speed (km/h)
            free_flow_speed = flow['freeFlowSpeed']   # Speed limit / Empty road speed (km/h)
            confidence = flow['confidence']           # Data reliability (0 to 1)
            
            # --- THE ALGORITHM: CONGESTION INDEX ---
            # Formula: How much speed is lost?
            # If Free Flow = 60km/h and Current = 15km/h -> 75% Congestion
            
            if free_flow_speed > 0:
                congestion_ratio = 1 - (current_speed / free_flow_speed)
                # Map 0-1 ratio to 0-10 Scale
                traffic_index = round(congestion_ratio * 10, 2)
            else:
                traffic_index = 0.0
            
            # Clamp value between 0 (Empty) and 10 (Gridlock)
            traffic_index = max(0.0, min(10.0, traffic_index))
            
            results.append({
                "Neighborhood": name,
                "Traffic_Index": traffic_index,
                "Current_Speed_kmh": current_speed,
                "Free_Flow_Speed_kmh": free_flow_speed,
                "Congestion_Confidence": confidence,
                "Timestamp": datetime.now().strftime('%H:%M')
            })
            
        else:
            print(f" [Error {response.status_code}]")
            # Append None so we know data is missing
            results.append({"Neighborhood": name, "Traffic_Index": None})

    except Exception as e:
        print(f" [Failed: {e}]")

    # Rate Limiting: Sleep 0.5s to respect Free Tier limits (5 calls/sec)
    time.sleep(0.5)

# ==========================================
# 3. SAVE DATA
# ==========================================
print(f"\n\n--- FETCH COMPLETE ---")
df = pd.DataFrame(results)

# Generate filename with timestamp (e.g., 'traffic_data_afternoon.csv')
filename = f"tomtom_traffic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
df.to_csv(filename, index=False)

print(f"Data saved to: {filename}")
print("\nSample Output:")
print(df.head())