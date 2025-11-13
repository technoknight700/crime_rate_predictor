import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union, voronoi_polygons  # <-- This is the fixed import
from shapely.geometry import Polygon
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, jsonify, request
from flask_cors import CORS
from fuzzywuzzy import process, fuzz
import datetime

# --- Configuration ---
# These are the absolute paths the script will use.
COORDS_FILE = r'C:\Users\nitaa\Downloads\foml_crime_predictor\Police_Station_Coords.csv'
HISTORY_FILE = r'C:\Users\nitaa\Downloads\foml_crime_predictor\mock_crime_history.csv'

# Mumbai Bounding Box for clipping Voronoi zones
MUMBAI_BBOX = (72.77, 18.89, 73.0, 19.28) # (min_lon, min_lat, max_lon, max_lat)

# --- Global Variables ---
app = Flask(__name__)
CORS(app) # Allow frontend to call backend
station_models = {}
gdf_zones = None
crime_history = None
BASE_START_DATE = datetime.date(2022, 1, 1)

# --- Helper Functions ---

def clean_station_name(name):
    """Standardizes station names for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    return (
        name.lower()
        .replace(" police station", "")
        .replace(" (w)", "")
        .replace(" (e)", "")
        .replace(" road", "")
        .replace(" marg", "")
        .replace("t.t.", "truck terminal")
        .replace(".", "")
        .replace(" ", "")
    )

# ---
# --- THIS IS THE NEW, CORRECTED FUNCTION ---
# ---
def create_voronoi_zones(gdf_stations):
    """Creates GeoDataFrame of Voronoi zones (jurisdictions) from station points."""
    print("Creating Voronoi zones...")
    
    # 1. Create a bounding box polygon for Mumbai
    min_lon, min_lat, max_lon, max_lat = MUMBAI_BBOX
    bbox_polygon = Polygon([
        (min_lon, min_lat), (max_lon, min_lat), 
        (max_lon, max_lat), (min_lon, max_lat)
    ])
    
    # 2. Generate Voronoi polygons
    try:
        # ---
        # --- THIS IS THE FIX ---
        # ---
        # Call shapely.ops.voronoi_polygons directly, bypassing the broken gpd wrapper
        # We also convert gdf_stations.geometry to a MultiPoint object for robustness
        multipoint = unary_union(gdf_stations.geometry)
        polygons_collection = voronoi_polygons(multipoint, extend_to=bbox_polygon)
        # ---
        # ---
        # ---
    except Exception as e:
        print(f"Error in shapely.ops.voronoi_polygons: {e}")
        print("Falling back to non-extended Voronoi.")
        polygons_collection = voronoi_polygons(unary_union(gdf_stations.geometry))

    # 3. Create a new GeoDataFrame from these polygons
    # We get the list of polygons from the GeometryCollection
    gdf_voronoi = gpd.GeoDataFrame(geometry=list(polygons_collection.geoms), crs="EPSG:4326")
    
    # 4. Attach the station names.
    # The polygons in 'gdf_voronoi' are in the *exact same order*
    # as the stations in 'gdf_stations'. We can join them by their index.
    
    # Reset index on stations to be safe
    gdf_stations = gdf_stations.reset_index(drop=True)
    gdf_voronoi = gdf_voronoi.reset_index(drop=True)
    
    # Join based on index
    gdf_voronoi['Police_Station_Coords'] = gdf_stations['Police_Station_Coords']
    
    # 5. Clip all geometries to the bounding box to ensure clean edges
    gdf_voronoi['geometry'] = gdf_voronoi.geometry.intersection(bbox_polygon)

    print(f"Voronoi zones created successfully. Found {len(gdf_voronoi)} zones.")
    return gdf_voronoi
# ---
# --- END OF THE UPDATED FUNCTION ---
# ---


def train_all_models(df_history, all_stations):
    """Trains and stores a linear regression model for each police station."""
    print(f"Training {len(all_stations)} ML models...")
    models = {}
    
    # Get list of unique station names from the crime history
    historical_stations = df_history['Police_Station_History'].unique()
    
    for station_coords_name in all_stations:
        # Find the best match in the crime history
        clean_coords_name = clean_station_name(station_coords_name)
        best_match, score = process.extractOne(
            clean_coords_name, 
            historical_stations, 
            scorer=fuzz.token_sort_ratio
        )
        
        if score > 80: # Using a threshold of 80 for a good match
            # Get this station's data
            station_df = df_history[df_history['Police_Station_History'] == best_match].copy()
            station_df['Date'] = pd.to_datetime(df_history['Date'])
            station_df = station_df.sort_values(by='Date')
            
            if len(station_df) > 1:
                # Create 'time_index' (e.g., Jan 2022 = 0, Feb 2022 = 1, ...)
                station_df['time_index'] = (
                    (station_df['Date'].dt.year - BASE_START_DATE.year) * 12 +
                    (station_df['Date'].dt.month - BASE_START_DATE.month)
                )
                
                X = station_df[['time_index']]
                y = station_df['Total_Crimes']
                
                # Train the model
                model = LinearRegression()
                model.fit(X, y)
                
                # Store the model and its matched name
                models[station_coords_name] = {
                    'model': model,
                    'history_name': best_match
                }
    
    print(f"Training complete. {len(models)} models trained.")
    return models

def predict_crime(station_name, target_date_str):
    """Predicts crime for a station and a future month, or gets historical data."""
    global crime_history, station_models, BASE_START_DATE

    target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    
    # 1. Check if we have a model for this station
    if station_name not in station_models:
        return 0, 'N/A'
        
    model_data = station_models[station_name]
    history_name = model_data['history_name']

    # 2. Check if this date is in our historical data
    historical_record = crime_history[
        (crime_history['Police_Station_History'] == history_name) &
        (crime_history['Date_str'] == target_date_str)
    ]
    
    if not historical_record.empty:
        return int(historical_record['Total_Crimes'].values[0]), 'Historical'

    # 3. If not, predict it
    model = model_data['model']
    
    # Calculate the time index for the target date
    time_index = (
        (target_date.year - BASE_START_DATE.year) * 12 +
        (target_date.month - BASE_START_DATE.month)
    )
    
    prediction = model.predict(np.array([[time_index]]))
    
    # Ensure prediction isn't negative
    return int(max(0, prediction[0])), 'Predicted'

def get_color(crime, low_thresh, mid_thresh):
    """Assigns a color based on crime thresholds."""
    if crime <= 0:
        return '#9CA3AF' # Gray
    elif crime <= low_thresh:
        return '#22C55E' # Green
    elif crime <= mid_thresh:
        return '#FACC15' # Yellow
    else:
        return '#F97316' # Orange/Red
        
# --- Main Application Logic ---

@app.before_request
def load_data_and_models():
    """Run once on startup: Load data, train models, create zones."""
    global gdf_zones, station_models, crime_history
    
    # Only run this once
    if gdf_zones is not None:
        return

    print("Loading data for the first time...")
    
    # 1. Load Crime History
    df_history = pd.read_csv(HISTORY_FILE, encoding='utf-8-sig')
    df_history['Date_str'] = pd.to_datetime(df_history['Date']).dt.strftime('%Y-%m-%d')
    df_history['Police_Station_History'] = df_history['Police_Station'].apply(clean_station_name)
    crime_history = df_history
    
    # 2. Load Station Coordinates
    df_coords = pd.read_csv(COORDS_FILE, encoding='utf-8-sig')
    df_coords = df_coords.dropna(subset=['Latitude', 'Longitude'])
    
    # --- THIS IS THE FIX ---
    # Forcefully rename the first column to 'Police_Station'
    # This solves issues with 'ï»¿Police_Station' (BOM) or 'Police Station' (space)
    df_coords = df_coords.rename(columns={df_coords.columns[0]: 'Police_Station'})
    # --- END OF FIX ---
    
    # Create a GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        df_coords,
        geometry=gpd.points_from_xy(df_coords.Longitude, df_coords.Latitude),
        crs="EPSG:4326"
    )
    # Store the original, clean name from this file
    gdf_stations['Police_Station_Coords'] = gdf_stations['Police_Station']

    # 3. Create Voronoi Zones (Jurisdictions)
    gdf_zones = create_voronoi_zones(gdf_stations)

    # 4. Train all ML Models
    station_models = train_all_models(df_history, gdf_stations['Police_Station_Coords'].unique())
    
    # 5. Link models and names to the zones
    gdf_zones = gdf_zones.merge(
        pd.DataFrame([
            {'Police_Station_Coords': k, 'history_name': v['history_name']} 
            for k, v in station_models.items()
        ]),
        on='Police_Station_Coords',
        how='left'
    )
    print("--- Backend is ready ---")


@app.route('/get_crime_map')
def get_crime_map():
    """API endpoint to get the GeoJSON map for a specific month."""
    selected_month = request.args.get('month') # e.g., '2025-10-31'
    if not selected_month:
        return jsonify({"error": "Missing 'month' parameter"}), 400

    results = []
    
    # Calculate crime for each zone
    for _, row in gdf_zones.iterrows():
        station_name_coords = row['Police_Station_Coords']
        crime, data_type = predict_crime(station_name_coords, selected_month)
        results.append({
            # ---
            # --- THIS IS THE FIX ---
            # ---
            # The key must match the column name in gdf_zones, which is 'Police_Station_Coords'
            'Police_Station_Coords': station_name_coords,
            # ---
            # ---
            # ---
            'history_name': row.get('history_name', 'N/A'),
            'crime': crime,
            'data_type': data_type
        })
    
    df_results = pd.DataFrame(results)
    
    # Calculate color thresholds for this month
    crime_values = df_results[df_results['crime'] > 0]['crime']
    if not crime_values.empty:
        low_thresh = crime_values.quantile(0.33)
        mid_thresh = crime_values.quantile(0.66)
    else:
        low_thresh, mid_thresh = 0, 0
        
    df_results['color_zone'] = df_results['crime'].apply(get_color, args=(low_thresh, mid_thresh))
    
    # Merge results back into the GeoDataFrame
    gdf_map_data = gdf_zones.merge(df_results, on='Police_Station_Coords')
    
    # Return as GeoJSON
    return gdf_map_data.to_json()


if __name__ == '__main__':
    load_data_and_models() # Load and train on start
    app.run(debug=True, port=5000)