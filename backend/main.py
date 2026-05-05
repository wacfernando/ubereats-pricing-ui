from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODELS_DIR = "models/"
DATA_PATH  = "models/ubereats_engineered.csv"
DB_PATH    = "recommendations.db"

GUARD_RAILS = {
    'Budget':      (180,  1320),
    'Mid-Range':   (540,  2800),
    'Fine Dining': (1095, 5900),
}

FEATURES_VERSION_A = [
    'restaurant_tier_encoded', 'dish_category_std_encoded',
    'area_encoded', 'cuisine_type_encoded',
    'restaurant_overall_rating', 'review_band_encoded',
    'comp_mean_price', 'comp_median_price', 'comp_price_stdev',
    'comp_n', 'price_percentile_25', 'price_percentile_75',
]

FEATURES_VERSION_B = [
    'restaurant_tier_encoded', 'dish_category_std_encoded',
    'area_encoded', 'cuisine_type_encoded',
    'restaurant_overall_rating', 'review_band_encoded',
    'is_top_liked', 'is_popular',
    'comp_mean_price', 'comp_median_price', 'comp_price_stdev',
    'comp_n', 'price_percentile_25', 'price_percentile_75',
]

# ── DATABASE SETUP ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            restaurant_tier     TEXT NOT NULL,
            area                TEXT NOT NULL,
            cuisine_type        TEXT NOT NULL,
            dish_category       TEXT NOT NULL,
            restaurant_rating   REAL,
            recommended_price   REAL NOT NULL,
            competitor_range_low  REAL NOT NULL,
            competitor_range_high REAL NOT NULL,
            competitor_count    INTEGER NOT NULL,
            positioning_percentile INTEGER NOT NULL,
            confidence_level    TEXT NOT NULL,
            new_entrant_advisory INTEGER NOT NULL,
            version_used        TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialised.")

def log_recommendation(req, result):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO recommendation_log (
                timestamp, restaurant_tier, area, cuisine_type,
                dish_category, restaurant_rating, recommended_price,
                competitor_range_low, competitor_range_high,
                competitor_count, positioning_percentile,
                confidence_level, new_entrant_advisory, version_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            req.restaurant_tier,
            req.area,
            req.cuisine_type,
            req.dish_category,
            req.restaurant_rating,
            result['recommended_price'],
            result['competitor_range_low'],
            result['competitor_range_high'],
            result['competitor_count'],
            result['positioning_percentile'],
            result['confidence_level'],
            1 if result['new_entrant_advisory'] else 0,
            result['version_used'],
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging error (non-fatal): {e}")

# ── LOAD MODELS ON STARTUP ────────────────────────────────────────────────────
print("Loading models...")
init_db()
final_models = joblib.load(f"{MODELS_DIR}final_models_meta.pkl")
le_dict      = joblib.load(f"{MODELS_DIR}label_encoders.pkl")
df_reference = pd.read_csv(DATA_PATH)

for tier in ['Budget', 'Mid-Range', 'Fine Dining']:
    fname = tier.lower().replace('-', '_').replace(' ', '_')
    final_models[tier]['model'] = joblib.load(f"{MODELS_DIR}prod_{fname}.pkl")

print("Models loaded successfully.")

# ── REQUEST SCHEMA ────────────────────────────────────────────────────────────
class PricingRequest(BaseModel):
    restaurant_tier:   str
    area:              str
    cuisine_type:      str
    dish_category:     str
    restaurant_rating: Optional[float] = None

# ── RECOMMENDATION LOGIC ──────────────────────────────────────────────────────
def recommend_price(req: PricingRequest):
    model_info   = final_models[req.restaurant_tier]
    model        = model_info['model']
    version      = model_info['version']
    features     = FEATURES_VERSION_A if version == 'Version A' else FEATURES_VERSION_B
    guard_low, guard_high = GUARD_RAILS[req.restaurant_tier]

    try:
        tier_enc     = int(le_dict['restaurant_tier'].transform([req.restaurant_tier])[0])
        area_enc     = int(le_dict['area'].transform([req.area])[0])
        cuisine_enc  = int(le_dict['cuisine_type'].transform([req.cuisine_type])[0])
        category_enc = int(le_dict['dish_category_std'].transform([req.dish_category])[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input value: {e}")

    if req.restaurant_rating is None or req.restaurant_rating < 1.0:
        tier_medians = {'Budget': 4.5, 'Mid-Range': 4.4, 'Fine Dining': 4.6}
        processed_rating = tier_medians[req.restaurant_tier]
        new_entrant = True
    else:
        processed_rating = req.restaurant_rating
        new_entrant = req.restaurant_rating < 3.0

    group = df_reference[
        (df_reference['restaurant_tier']   == req.restaurant_tier) &
        (df_reference['dish_category_std'] == req.dish_category)
    ]['selling_price_lkr']

    if len(group) == 0:
        raise HTTPException(status_code=400, detail="No competitor data for this tier and category combination.")

    comp_mean   = float(group.mean())
    comp_median = float(group.median())
    comp_stdev  = float(group.std()) if len(group) > 1 else 0.0
    comp_n      = int(len(group))
    p25         = float(group.quantile(0.25))
    p75         = float(group.quantile(0.75))

    input_dict = {
        'restaurant_tier_encoded':   tier_enc,
        'dish_category_std_encoded': category_enc,
        'area_encoded':              area_enc,
        'cuisine_type_encoded':      cuisine_enc,
        'restaurant_overall_rating': processed_rating,
        'review_band_encoded':       1,
        'comp_mean_price':           comp_mean,
        'comp_median_price':         comp_median,
        'comp_price_stdev':          comp_stdev,
        'comp_n':                    comp_n,
        'price_percentile_25':       p25,
        'price_percentile_75':       p75,
        'is_top_liked':              0,
        'is_popular':                0,
    }

    X_input = pd.DataFrame([input_dict])[features]

    log_pred        = model.predict(X_input)[0]
    predicted_price = np.expm1(log_pred)

    if version == 'Version A':
        predicted_price *= 0.935

    final_price = float(np.clip(predicted_price, guard_low, guard_high))
    final_price = round(final_price / 10) * 10

    all_prices = sorted(group.tolist())
    position   = sum(p <= final_price for p in all_prices) / len(all_prices) * 100

    if comp_n >= 50:
        confidence = 'High'
    elif comp_n >= 20:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    result = {
        "recommended_price":      final_price,
        "competitor_range_low":   round(p25),
        "competitor_range_high":  round(p75),
        "competitor_count":       comp_n,
        "positioning_percentile": round(position),
        "confidence_level":       confidence,
        "new_entrant_advisory":   new_entrant,
        "version_used":           version,
        "tier":                   req.restaurant_tier,
        "area":                   req.area,
        "dish_category":          req.dish_category,
    }

    # Log to database
    log_recommendation(req, result)

    return result

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "UberEATS Pricing API is running"}

@app.post("/recommend")
def get_recommendation(req: PricingRequest):
    return recommend_price(req)

@app.get("/options")
def get_options():
    return {
        "tiers":      ["Budget", "Mid-Range", "Fine Dining"],
        "areas":      [f"Colombo {i}" for i in range(1, 16)],
        "cuisines":   ["American", "Cafe / Bakery", "Chinese", "Indian",
                       "Italian", "Japanese", "Korean", "Middle Eastern",
                       "Other", "Sri Lankan", "Thai"],
        "categories": [
            "Appetizer / Sides / Salads", "Bakery / Breads / Pastries",
            "Beverage", "Biriyani", "Breakfast / Traditional",
            "Burger / Sandwich / Wrap", "Dessert / Bakery / Sweets",
            "Fried Chicken", "Fried Rice", "Japanese / Sushi",
            "Kottu / Roti / Indian Breads",
            "Meat & Veg Curries / Stir-fry / Grill",
            "Noodles / Pasta", "Nuts / Dried Fruits", "Pizza",
            "Rice & Curry", "Seafood", "Set Meal / Combo", "Soup / Stew"
        ]
    }

@app.get("/logs")
def get_logs():
    """View all recommendation logs — for research/admin use."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recommendation_log ORDER BY timestamp DESC LIMIT 100")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return {"count": len(rows), "logs": [dict(zip(columns, row)) for row in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))