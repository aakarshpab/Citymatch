import requests
import pandas as pd
import time
import sys

# 1. The 38 Stations
monitoring_stations = {
    "Mandir Marg": (28.6364, 77.1987), "ITO": (28.6285, 77.2410), "Pusa (DPCC)": (28.6396, 77.1463),
    "Pusa (IMD)": (28.6358, 77.1530), "Major Dhyan Chand Stadium": (28.6113, 77.2377),
    "Jawaharlal Nehru Stadium": (28.5802, 77.2338), "Dr. Karni Singh Shooting Range": (28.4975, 77.2642),
    "Alipur": (28.8153, 77.1530), "Narela": (28.8570, 77.1009), "Bawana": (28.7762, 77.0511),
    "DTU": (28.7501, 77.1113), "Ashok Vihar": (28.6954, 77.1816), "Jahangirpuri": (28.7328, 77.1706),
    "Rohini": (28.7325, 77.1199), "Sonia Vihar": (28.7105, 77.2495), "Wazirpur": (28.6998, 77.1654),
    "North Campus (DU)": (28.6877, 77.2058), "Burari Crossing": (28.7256, 77.2012),
    "Siri Fort": (28.5504, 77.2159), "R.K. Puram": (28.5632, 77.1869), "Okhla Phase-2": (28.5308, 77.2713),
    "Nehru Nagar": (28.5679, 77.2505), "Sri Aurobindo Marg": (28.5313, 77.1902),
    "Aya Nagar": (28.4707, 77.1099), "Lodhi Road": (28.5918, 77.2273), "CRRI Mathura Road": (28.5512, 77.2736),
    "IGI Airport (T3)": (28.5627, 77.1180), "Anand Vihar": (28.6469, 77.3160),
    "Vivek Vihar": (28.6723, 77.3153), "Patparganj": (28.6237, 77.2872),
    "IHBAS (Dilshad Garden)": (28.6811, 77.3025), "Punjabi Bagh": (28.6683, 77.1167),
    "Shadipur": (28.6515, 77.1473), "Mundka": (28.6847, 77.0766), "NSIT Dwarka": (28.6090, 77.0325),
    "Dwarka Sector-8": (28.5710, 77.0719), "Najafgarh": (28.6148, 76.9858)
}

results = []
URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
print("--- EXTRACTING SEASONAL DATA (2023-2025) ---")

count = 0
for station_name, (lat, lon) in monitoring_stations.items():
    count += 1
    sys.stdout.write(f"\rStation {count}/38: {station_name}")
    sys.stdout.flush()
    try:
        # Fetch 2 Years of Hourly Data
        r = requests.get(URL, params={
            "latitude": lat, "longitude": lon, "hourly": "pm2_5,nitrogen_dioxide",
            "start_date": "2023-01-01", "end_date": "2024-12-31" 
        })
        data = r.json()
        df_temp = pd.DataFrame({'time': pd.to_datetime(data['hourly']['time']),
                                'pm25': data['hourly']['pm2_5'], 'no2': data['hourly']['nitrogen_dioxide']})
        
        # Calculate Seasons
        df_temp['month'] = df_temp['time'].dt.month
        winter_avg = df_temp[df_temp['month'].isin([11, 12, 1, 2])]['pm25'].mean()
        monsoon_avg = df_temp[df_temp['month'].isin([7, 8, 9])]['pm25'].mean()
        annual_avg = df_temp['pm25'].mean()
        traffic_idx = max(2.0, min(9.5, (df_temp['no2'].mean() / 60) * 10)) # NO2 Proxy for Traffic

        results.append({
            "Station_Name": station_name,
            "PM25_Annual": round(annual_avg, 1),
            "PM25_Winter": round(winter_avg, 1),
            "PM25_Monsoon": round(monsoon_avg, 1),
            "Traffic_Index": round(traffic_idx, 1)
        })
    except:
        print(f" - Error fetching {station_name}")
    time.sleep(0.5)

pd.DataFrame(results).to_csv("station_seasonal_stats.csv", index=False)
print("\nDone. File 'station_seasonal_stats.csv' created.")