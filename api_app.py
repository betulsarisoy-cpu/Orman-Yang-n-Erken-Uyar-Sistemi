"""Akdeniz yangın erken uyarı modeli için basit Flask API.

Bu dosya modeli yeniden eğitmez ve model dosyasını değiştirmez.
Yalnızca mevcut joblib paketini okuyarak tahmin üretir.
"""
import truststore

# Windows'un güvenilir sertifika deposunu Python'da kullanır.
truststore.inject_into_ssl()
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests as http_requests
from flask import Flask, jsonify, request


# ============================================================
# 1. MODEL DOSYASINI BUL VE SADECE OKU
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "mesogeos_erken_uyari_modeli.joblib"

model_from_environment = os.getenv("YANGIN_MODEL_PATH")

MODEL_CANDIDATES = [
    Path(model_from_environment) if model_from_environment else None,
    APP_DIR / MODEL_NAME,
    APP_DIR / "yangin_model_ciktilari_basit" / MODEL_NAME,
    APP_DIR / "final_test_outputs" / MODEL_NAME,
]

MODEL_FILE = next(
    (
        path
        for path in MODEL_CANDIDATES
        if path is not None and path.exists()
    ),
    None,
)

if MODEL_FILE is None:
    raise FileNotFoundError(
        f"{MODEL_NAME} bulunamadı. Modeli api_app.py ile aynı klasöre "
        "veya yangin_model_ciktilari_basit klasörüne koyun."
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
feature_columns = list(bundle["feature_columns"])
feature_names_tr = dict(bundle.get("feature_names_turkish", {}))
defaults = dict(bundle["streamlit_input_defaults"])
lower_bounds = dict(bundle.get("streamlit_input_lower_bounds", {}))
upper_bounds = dict(bundle.get("streamlit_input_upper_bounds", {}))

threshold_options = dict(
    bundle.get(
        "streamlit_threshold_options",
        {"Model eşiği": float(bundle["threshold"])},
    )
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# 2. FLASK UYGULAMASI
# ============================================================

app = Flask(__name__)
app.json.ensure_ascii = False


# API'de kabul edilen kullanıcı dostu alanlar.
INPUT_ALIASES = {
    "rh_percent": ("rh", lambda value: value / 100.0),
    "smi_percent": ("smi", lambda value: value / 100.0),
    "tp_mm": ("tp", lambda value: value / 1000.0),
    "lc_forest_percent": ("lc_forest", lambda value: value / 100.0),
    "lc_shrubland_percent": ("lc_shrubland", lambda value: value / 100.0),
    "lc_agriculture_percent": (
        "lc_agriculture",
        lambda value: value / 100.0,
    ),
    "lc_grassland_percent": ("lc_grassland", lambda value: value / 100.0),
    "lc_settlement_percent": (
        "lc_settlement",
        lambda value: value / 100.0,
    ),
}

DIRECT_INPUTS = {
    "x",
    "y",
    "t2m",
    "d2m",
    "wind_speed",
    "ssrd",
    "sp",
    "ndvi",
    "lai",
    "lst_day",
    "lst_night",
    "dem",
    "slope",
    "roads_distance",
    "population",
}

REQUIRED_INPUTS = {
    "x",
    "y",
    "t2m",
    "d2m",
    "rh_percent",
    "smi_percent",
    "tp_mm",
    "wind_speed",
    "ndvi",
    "lai",
    "lst_day",
    "lst_night",
    "lc_forest_percent",
    "lc_shrubland_percent",
}


def parse_risk_date(value):
    """YYYY-AA-GG biçimindeki tarihi date nesnesine dönüştürür."""
    if value is None:
        return date.today()

    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError(
            "risk_date YYYY-AA-GG biçiminde olmalıdır. Örnek: 2026-09-15"
        ) from error


def convert_api_inputs(api_inputs):
    """Kullanıcı dostu API alanlarını model birimlerine dönüştürür."""
    if not isinstance(api_inputs, dict):
        raise ValueError("inputs alanı JSON nesnesi olmalıdır.")

    missing_inputs = sorted(REQUIRED_INPUTS.difference(api_inputs))
    if missing_inputs:
        raise ValueError(
            "Eksik zorunlu girdiler: " + ", ".join(missing_inputs)
        )

    allowed_inputs = DIRECT_INPUTS.union(INPUT_ALIASES)
    unknown_inputs = sorted(set(api_inputs).difference(allowed_inputs))
    if unknown_inputs:
        raise ValueError(
            "Tanınmayan girdiler: " + ", ".join(unknown_inputs)
        )

    model_inputs = {}

    for name, raw_value in api_inputs.items():
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} sayısal bir değer olmalıdır.") from error

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} sonlu bir sayı olmalıdır.")

        if name in INPUT_ALIASES:
            model_name, converter = INPUT_ALIASES[name]
            model_inputs[model_name] = float(converter(numeric_value))
        else:
            model_inputs[name] = numeric_value

    percent_names = [
        "rh_percent",
        "smi_percent",
        "lc_forest_percent",
        "lc_shrubland_percent",
        "lc_agriculture_percent",
        "lc_grassland_percent",
        "lc_settlement_percent",
    ]

    for name in percent_names:
        if name in api_inputs and not 0 <= float(api_inputs[name]) <= 100:
            raise ValueError(f"{name} 0 ile 100 arasında olmalıdır.")

    if not -180 <= model_inputs["x"] <= 180:
        raise ValueError("x (boylam) -180 ile 180 arasında olmalıdır.")

    if not -90 <= model_inputs["y"] <= 90:
        raise ValueError("y (enlem) -90 ile 90 arasında olmalıdır.")

    if not -1 <= model_inputs["ndvi"] <= 1:
        raise ValueError("ndvi -1 ile 1 arasında olmalıdır.")

    return model_inputs


def calculate_user_features(row, risk_date):
    """Streamlit ekranındaki feature hesaplarını aynen uygular."""
    month = risk_date.month
    day_of_year = risk_date.timetuple().tm_yday

    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)
    row["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    row["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    row["is_fire_season"] = int(5 <= month <= 10)

    t2m = row.get("t2m", np.nan)
    d2m = row.get("d2m", np.nan)
    rh = row.get("rh", np.nan)
    wind_speed = row.get("wind_speed", np.nan)
    smi = row.get("smi", np.nan)
    ssrd = row.get("ssrd", np.nan)

    if np.isfinite(t2m) and np.isfinite(d2m):
        row["temp_dewpoint_difference"] = t2m - d2m

    if np.isfinite(t2m) and np.isfinite(rh):
        saturation_vapor_pressure = 0.6108 * np.exp(
            (17.27 * t2m) / (t2m + 237.3)
        )
        row["saturation_vapor_pressure"] = saturation_vapor_pressure
        row["vpd"] = saturation_vapor_pressure * (1 - np.clip(rh, 0, 1))
        row["dryness_index"] = max(t2m, 0) * (1 - np.clip(rh, 0, 1))

        if np.isfinite(wind_speed):
            row["wind_dryness_interaction"] = (
                max(wind_speed, 0) * (1 - np.clip(rh, 0, 1))
            )
            row["hot_dry_windy_index"] = (
                max(t2m, 0)
                * (1 - np.clip(rh, 0, 1))
                * np.log1p(max(wind_speed, 0))
            )

        if np.isfinite(ssrd):
            row["radiation_dryness"] = (
                np.log1p(max(ssrd, 0)) * (1 - np.clip(rh, 0, 1))
            )

    if np.isfinite(t2m) and np.isfinite(smi):
        row["heat_soil_dryness"] = max(t2m, 0) * (1 - np.clip(smi, 0, 1))

    if np.isfinite(rh) and np.isfinite(smi):
        row["atmosphere_soil_dryness"] = (
            (1 - np.clip(rh, 0, 1)) * (1 - np.clip(smi, 0, 1))
        )

    lst_day = row.get("lst_day", np.nan)
    lst_night = row.get("lst_night", np.nan)

    if np.isfinite(lst_day) and np.isfinite(lst_night):
        row["lst_day_night_difference"] = lst_day - lst_night
        row["lst_mean"] = (lst_day + lst_night) / 2

    if np.isfinite(lst_day) and np.isfinite(t2m):
        row["lst_air_temperature_difference"] = lst_day - t2m

    if np.isfinite(lst_night) and np.isfinite(t2m):
        row["lst_night_air_difference"] = lst_night - t2m

    land_cover_names = [
        "lc_agriculture",
        "lc_forest",
        "lc_grassland",
        "lc_settlement",
        "lc_shrubland",
        "lc_sparse_vegetation",
        "lc_water_bodies",
        "lc_wetland",
    ]

    land_cover_values = {
        name: max(float(row.get(name, 0.0)), 0.0)
        for name in land_cover_names
    }

    row["vegetation_fuel_fraction"] = np.clip(
        land_cover_values["lc_forest"]
        + land_cover_values["lc_grassland"]
        + land_cover_values["lc_shrubland"]
        + land_cover_values["lc_sparse_vegetation"],
        0,
        1,
    )

    total_lc = sum(land_cover_values.values())
    if total_lc > 0:
        probabilities = np.array(
            [value / total_lc for value in land_cover_values.values()]
        )
        valid_probabilities = probabilities[probabilities > 0]
        entropy = -(
            valid_probabilities * np.log(valid_probabilities)
        ).sum()
        row["land_cover_diversity"] = np.clip(
            entropy / np.log(len(land_cover_values)),
            0,
            1,
        )

    ndvi = row.get("ndvi", np.nan)

    if np.isfinite(smi):
        soil_dryness = 1 - np.clip(smi, 0, 1)
        row["forest_dryness"] = land_cover_values["lc_forest"] * soil_dryness
        row["shrubland_dryness"] = (
            land_cover_values["lc_shrubland"] * soil_dryness
        )
        row["grassland_dryness"] = (
            land_cover_values["lc_grassland"] * soil_dryness
        )
        row["agriculture_dryness"] = (
            land_cover_values["lc_agriculture"] * soil_dryness
        )
        row["dry_fuel_proxy"] = row["vegetation_fuel_fraction"] * soil_dryness

        if np.isfinite(ndvi):
            row["live_fuel_moisture_proxy"] = (
                max(ndvi, 0) * np.clip(smi, 0, 1)
            )

    if "vpd" in row and np.isfinite(row["vpd"]):
        row["vegetation_vpd_stress"] = (
            row["vegetation_fuel_fraction"] * row["vpd"]
        )

    if (
        "hot_dry_windy_index" in row
        and np.isfinite(row["hot_dry_windy_index"])
    ):
        row["forest_hot_dry_windy"] = (
            land_cover_values["lc_forest"] * row["hot_dry_windy_index"]
        )

    population = row.get("population", np.nan)
    roads_distance = row.get("roads_distance", np.nan)

    if np.isfinite(population):
        row["log_population"] = np.log1p(max(population, 0))

    if np.isfinite(roads_distance):
        row["road_accessibility"] = 1 / (max(roads_distance, 0) + 1)

    if "log_population" in row and "road_accessibility" in row:
        row["human_pressure"] = (
            row["log_population"] * row["road_accessibility"]
        )

    row["forest_settlement_interaction"] = (
        land_cover_values["lc_forest"]
        * land_cover_values["lc_settlement"]
    )

    if "human_pressure" in row:
        row["forest_human_pressure"] = (
            land_cover_values["lc_forest"] * row["human_pressure"]
        )

    return row


def build_model_row(user_inputs, risk_date):
    """186 feature'ı eğitimdeki sırayla hazırlar."""
    row = {
        feature: float(defaults.get(feature, 0.0))
        for feature in feature_columns
    }

    for key, value in user_inputs.items():
        if key in row:
            row[key] = float(value)

    row = calculate_user_features(row, risk_date)
    frame = pd.DataFrame([row]).reindex(columns=feature_columns)

    if list(frame.columns) != feature_columns:
        raise RuntimeError("Model feature sırası oluşturulamadı.")

    return frame


def find_out_of_range_inputs(user_inputs):
    """Eğitim aralığı dışındaki değerleri bilgilendirme için bulur."""
    result = []

    for key, value in user_inputs.items():
        lower = lower_bounds.get(key)
        upper = upper_bounds.get(key)

        if lower is None or upper is None:
            continue
        if pd.isna(lower) or pd.isna(upper):
            continue

        if value < lower or value > upper:
            result.append(
                {
                    "variable": feature_names_tr.get(key, key),
                    "value": round(float(value), 6),
                    "training_min": round(float(lower), 6),
                    "training_max": round(float(upper), 6),
                }
            )

    return result


def score_level(score_percent):
    if score_percent < 30:
        return "Düşük"
    if score_percent < 55:
        return "İzlenmeli"
    if score_percent < 75:
        return "Yüksek"
    return "Çok Yüksek"


# ============================================================
# 3. API UÇ NOKTALARI
# ============================================================

@app.get("/")
def home():
    return jsonify(
        {
            "application": "Akdeniz Yangın Erken Uyarı API",
            "status": "çalışıyor",
            "endpoints": ["GET /health", "GET /metadata", "POST /predict"],
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model_name": bundle.get("model_name", type(model).__name__),
            "feature_count": len(feature_columns),
            "model_file": MODEL_FILE.name,
        }
    )


@app.get("/metadata")
def metadata():
    return jsonify(
        {
            "target": bundle.get(
                "target_definition",
                "Son 59 günlük bilgiden ertesi gün yüksek tehlike tahmini",
            ),
            "history_days": bundle.get("history_days", 59),
            "feature_count": len(feature_columns),
            "threshold_policies": threshold_options,
            "required_inputs": sorted(REQUIRED_INPUTS),
            "optional_inputs": sorted(
                DIRECT_INPUTS.union(INPUT_ALIASES).difference(REQUIRED_INPUTS)
            ),
            "weather_endpoint": "GET /weather",
            "note": (
                "Tahmin API'si Streamlit girdilerini kullanır; /weather seçilen "
                "koordinata göre meteorolojiyi otomatik getirir. "
                "Kullanıcının sağlamadığı geçmiş özetleri eğitim medyanlarında kalır."
            ),
        }
    )


@app.get("/weather")
def weather():
    """Seçilen koordinat için hedef günden bir önceki günün verisini getirir."""
    try:
        latitude = float(request.args.get("latitude"))
        longitude = float(request.args.get("longitude"))
        risk_date = parse_risk_date(request.args.get("risk_date"))

        if not -90 <= latitude <= 90:
            raise ValueError("latitude -90 ile 90 arasında olmalıdır.")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude -180 ile 180 arasında olmalıdır.")

        # Model ertesi günü tahmin ettiği için hedef günün bir önceki günü alınır.
        observation_date = risk_date - timedelta(days=1)
        today = date.today()

        if observation_date < today - timedelta(days=92):
            raise ValueError(
                "Canlı hava servisi bu ekranda en fazla son 92 günü destekliyor."
            )
        if observation_date > today + timedelta(days=15):
            raise ValueError(
                "Canlı hava servisi en fazla 16 günlük tahmin sunuyor."
            )

        hourly_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "shortwave_radiation",
            "soil_moisture_0_to_1cm",
        ]

        response = http_requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(hourly_variables),
                "start_date": observation_date.isoformat(),
                "end_date": observation_date.isoformat(),
                "timezone": "auto",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            },
            timeout=25,
        )
        response.raise_for_status()
        weather_data = response.json()

        if "hourly" not in weather_data:
            raise ValueError("Hava servisinden saatlik veri alınamadı.")

        hourly_frame = pd.DataFrame(weather_data["hourly"])

        if hourly_frame.empty:
            raise ValueError("Seçilen tarih için hava verisi bulunamadı.")

        def mean_value(column):
            values = pd.to_numeric(hourly_frame[column], errors="coerce")
            result = values.mean()
            if pd.isna(result):
                raise ValueError(f"{column} verisi alınamadı.")
            return float(result)

        def sum_value(column):
            values = pd.to_numeric(hourly_frame[column], errors="coerce")
            result = values.sum(min_count=1)
            if pd.isna(result):
                raise ValueError(f"{column} verisi alınamadı.")
            return float(result)

        # Open-Meteo basıncı hPa verir; model Pa kullanır.
        surface_pressure_pa = mean_value("surface_pressure") * 100

        # Saatlik radyasyon W/m² ortalamasıdır. Her saat için 3600 ile
        # çarpılarak günlük J/m² toplamına dönüştürülür.
        shortwave_j_m2 = sum_value("shortwave_radiation") * 3600

        result = {
            "t2m": round(mean_value("temperature_2m"), 2),
            "d2m": round(mean_value("dew_point_2m"), 2),
            "rh_percent": round(mean_value("relative_humidity_2m")),
            "smi_percent": round(
                mean_value("soil_moisture_0_to_1cm") * 100
            ),
            "tp_mm": round(sum_value("precipitation"), 2),
            "wind_speed": round(mean_value("wind_speed_10m"), 2),
            "sp": round(surface_pressure_pa),
            "ssrd": round(shortwave_j_m2),
            "dem": round(float(weather_data.get("elevation", 0.0)), 1),
        }

        return jsonify(
            {
                "status": "success",
                "source": "Open-Meteo hava modeli",
                "risk_date": risk_date.isoformat(),
                "observation_date": observation_date.isoformat(),
                "coordinates": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "values": result,
                "not_updated": [
                    "NDVI",
                    "LAI",
                    "gündüz/gece uydu yüzey sıcaklığı",
                    "arazi örtüsü oranları",
                ],
                "note": (
                    "Toprak nemi, hava modelindeki 0-1 cm hacimsel nemin "
                    "yüzde gösterimidir; uydu SMI ile birebir aynı değildir."
                ),
            }
        ), 200

    except (TypeError, ValueError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    except http_requests.RequestException as error:
        return jsonify(
            {
                "status": "error",
                "message": "Güncel hava servisine ulaşılamadı.",
                "detail": str(error),
            }
        ), 503


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=False)

        if not isinstance(payload, dict):
            raise ValueError("İstek gövdesi JSON nesnesi olmalıdır.")

        risk_date = parse_risk_date(payload.get("risk_date"))
        api_inputs = payload.get("inputs")
        model_inputs = convert_api_inputs(api_inputs)

        selected_policy = payload.get(
            "threshold_policy",
            "Maliyet - Validation 2021",
        )

        if selected_policy not in threshold_options:
            raise ValueError(
                "Geçersiz threshold_policy. Kullanılabilir seçenekler: "
                + ", ".join(threshold_options)
            )

        selected_threshold = float(threshold_options[selected_policy])
        model_frame = build_model_row(model_inputs, risk_date)

        positive_index = list(model.classes_).index(1)
        raw_score = float(
            model.predict_proba(model_frame)[0, positive_index]
        )

        risk_score = float(np.clip(raw_score * 100, 0, 100))

        response = {
            "status": "success",
            "location": payload.get("location", "Belirtilmedi"),
            "risk_date": risk_date.isoformat(),
            "prediction_horizon": "Ertesi gün yüksek tehlike risk skoru",
            "model_name": bundle.get("model_name", type(model).__name__),
            "risk_score": round(risk_score, 2),
            "risk_level": score_level(risk_score),
            "threshold_policy": selected_policy,
            "threshold": round(selected_threshold, 4),
            "alarm": bool(raw_score >= selected_threshold),
            "out_of_training_range": find_out_of_range_inputs(model_inputs),
            "note": (
                "Risk skoru kalibre edilmiş yangın olasılığı değildir. "
                "Karar destek amacıyla saha ve resmî verilerle birlikte kullanılmalıdır."
            ),
        }

        return jsonify(response), 200

    except (ValueError, TypeError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": "Tahmin üretilemedi.",
                "detail": str(error),
            }
        ), 500


if __name__ == "__main__":
    # Bu geliştirme sunucusu yerel proje demosu içindir.
    app.run(host="127.0.0.1", port=5000, debug=False)
