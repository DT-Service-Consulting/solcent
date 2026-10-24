#!/usr/bin/env python3
"""
fetch_mos_data.py
=================
Fetches historical day-ahead weather forecasts (ECMWF) and ERA5 ground truth 
for each house to build the MOS training dataset safely.

Outputs: data/interim/mos_training_data.csv
"""

import pandas as pd
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "shortlist.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = OUTPUT_DIR / "mos_training_data.csv"

# Configuration
START_DATE = '2024-04-01'
END_DATE = '2026-05-31'
NWP_MODEL = 'ecmwf_ifs025' # The undisputed champion from our bake-off!

def fetch_openmeteo_json(url, params, max_retries=5):
    query_string = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query_string}"
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Solcent-MOS-Pipeline'})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sleep_time = 10 * (2 ** attempt)
                print(f"      [Rate Limit] HTTP 429. Sleeping for {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"      [HTTP Error] {e.code}: {e.reason}. Retrying...")
                time.sleep(5)
        except Exception as e:
            print(f"      [Network Error] {e}. Retrying...")
            time.sleep(5)
            
    return None

def main():
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    meta_df = pd.read_csv(METADATA_FILE)
    system_coords = meta_df[['system_id', 'latitude', 'longitude']].drop_duplicates()

    print(f"Preparing to fetch ECMWF data for {len(system_coords)} locations...")
    all_house_data = []

    for index, row in system_coords.iterrows():
        sys_id = row['system_id']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"Processing System ID: {sys_id} ({index+1}/{len(system_coords)})...")
        
        try:
            # 1. Fetch ERA5 Ground Truth
            truth_params = {
                "latitude": lat, "longitude": lon,
                "start_date": START_DATE, "end_date": END_DATE,
                "hourly": ["shortwave_radiation", "cloud_cover"],
                "timezone": "UTC"
            }
            truth_data = fetch_openmeteo_json("https://archive-api.open-meteo.com/v1/archive", truth_params)
            if not truth_data or 'hourly' not in truth_data:
                print("  -> Skipped: Failed to fetch ground truth.")
                continue
            
            truth_df = pd.DataFrame({
                "timestamp": pd.to_datetime(truth_data['hourly']['time'], utc=True),
                "actual_ghi": truth_data['hourly']['shortwave_radiation'],
                "actual_cloud": truth_data['hourly']['cloud_cover']
            })

            # 2. Fetch ECMWF Forecast Data
            fc_params = {
                "latitude": lat, "longitude": lon,
                "start_date": START_DATE, "end_date": END_DATE,
                "hourly": ["shortwave_radiation", "cloud_cover", "temperature_2m"],
                "models": NWP_MODEL,
                "timezone": "UTC"
            }
            fc_data = fetch_openmeteo_json("https://historical-forecast-api.open-meteo.com/v1/forecast", fc_params)
            if not fc_data or 'hourly' not in fc_data:
                print("  -> Skipped: Failed to fetch forecast data.")
                continue
            
            hourly_dict = fc_data['hourly']
            
            # Safely extract keys using dynamic lookup to bypass Open-Meteo's naming quirks
            ghi_key = next((k for k in hourly_dict.keys() if 'shortwave_radiation' in k), None)
            cloud_key = next((k for k in hourly_dict.keys() if 'cloud_cover' in k), None)
            temp_key = next((k for k in hourly_dict.keys() if 'temperature_2m' in k), None)
            
            if not ghi_key:
                print("  -> Skipped: Irradiance key not found in response.")
                continue

            fc_df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly_dict['time'], utc=True),
                "ecmwf_ghi": hourly_dict[ghi_key],
                "ecmwf_cloud": hourly_dict[cloud_key] if cloud_key else 0,
                "ecmwf_temp": hourly_dict[temp_key] if temp_key else 0
            })

            # 3. Merge Truth and Forecast
            merged_df = pd.merge_asof(
                truth_df.sort_values('timestamp'),
                fc_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            merged_df.insert(0, 'system_id', sys_id)
            merged_df = merged_df[merged_df['actual_ghi'] > 10].dropna()
            
            all_house_data.append(merged_df)
            print(f"  -> Success: Added {len(merged_df)} daylight records.")
            time.sleep(1.5)
            
        except Exception as e:
            print(f"  -> Error processing data for {sys_id}: {e}")

    if all_house_data:
        master_df = pd.concat(all_house_data, ignore_index=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully generated MOS dataset! Saved {len(master_df):,} total records to {OUTPUT_FILE.name}")
    else:
        print("Failed to generate dataset.")

if __name__ == "__main__":
    main()