# -*- coding: utf-8 -*-
# AKDENİZ YANGIN ERKEN UYARI DESTEK SİSTEMİ
# Streamlit portfolyo/demo uygulaması
# Çalıştırma: streamlit run streamlit_api_app.py
# Model dosyası bu dosya ile aynı klasörde veya
# yangin_model_ciktilari_basit klasöründe bulunabilir.

from pathlib import Path
from datetime import date, timedelta
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests

import os
import streamlit as st

API_URL = st.secrets.get(
    "YANGIN_API_URL",
    os.getenv(
        "YANGIN_API_URL",
        "http://127.0.0.1:5000",
    ),
).rstrip("/")


# ============================================================
# 1. SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Akdeniz Yangın Erken Uyarı",
    page_icon="🔥",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Ana sayfa genişliği ve başlık için güvenli üst boşluk */
        .block-container {
            padding-top: 2.50rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1450px !important;
        }

        /* Başlık sabit yükseklik kullanmaz; dar ekranda alt satıra geçer */
        h1 {
            font-size: clamp(1.85rem, 3.2vw, 3.10rem) !important;
            line-height: 1.35 !important;
            font-weight: 750 !important;
            letter-spacing: -0.02em !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: normal !important;
            margin-top: 0 !important;
            margin-bottom: 0.40rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.35rem !important;
        }

        /* Streamlit'in başlık kapsayıcısı da metni kırpmamalı */
        [data-testid="stHeadingWithActionElements"] {
            overflow: visible !important;
            height: auto !important;
            min-height: fit-content !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(242, 247, 245, 0.60);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 10px;
            padding: 12px;
            background: rgba(255,255,255,0.55);
        }

        .metric-card {
            border: 1px solid rgba(120,120,120,0.22);
            border-radius: 16px;
            padding: 18px;
            background: rgba(255,255,255,0.04);
            min-height: 132px;
        }

        .risk-number {
            font-size: 2.45rem;
            font-weight: 850;
            margin: 0;
        }

        .small-muted {
            font-size: 0.88rem;
            opacity: 0.7;
        }

        .status-safe {
            border-left: 7px solid #2a9d8f;
        }

        .status-watch {
            border-left: 7px solid #f4a261;
        }

        .status-alert {
            border-left: 7px solid #e76f51;
        }

        .status-high {
            border-left: 7px solid #d62828;
        }

        /* Tablet ve küçük bilgisayar ekranları */
        @media (max-width: 900px) {
            .block-container {
                padding-top: 2rem !important;
                padding-left: 1.20rem !important;
                padding-right: 1.20rem !important;
            }

            h1 {
                font-size: 2rem !important;
                line-height: 1.40 !important;
            }
        }

        /* Telefon ekranları */
        @media (max-width: 600px) {
            .block-container {
                padding-top: 1.75rem !important;
                padding-left: 0.90rem !important;
                padding-right: 0.90rem !important;
            }

            h1 {
                font-size: 1.65rem !important;
                line-height: 1.45 !important;
            }

            .risk-number {
                font-size: 2rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. MODEL DOSYASINI BULMA
# ============================================================

APP_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("YANGIN_API_URL", "http://127.0.0.1:5000").rstrip("/")
model_from_environment = os.getenv("YANGIN_MODEL_PATH")

MODEL_CANDIDATES = [
    Path(model_from_environment) if model_from_environment else None,
    APP_DIR / "mesogeos_erken_uyari_modeli.joblib",
    APP_DIR / "yangin_model_ciktilari_basit" / "mesogeos_erken_uyari_modeli.joblib",
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
    st.error(
        "Model dosyası bulunamadı. "
        "`mesogeos_erken_uyari_modeli.joblib` dosyasını "
        "bu Streamlit dosyasıyla aynı klasöre veya "
        "`yangin_model_ciktilari_basit` klasörüne koyun."
    )
    st.stop()


@st.cache_resource
def load_bundle(model_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(model_path)


bundle = load_bundle(MODEL_FILE)

model = bundle["model"]
feature_columns = list(bundle["feature_columns"])
feature_names_tr = dict(bundle.get("feature_names_turkish", {}))

defaults = dict(bundle["streamlit_input_defaults"])
lower_bounds = dict(bundle["streamlit_input_lower_bounds"])
upper_bounds = dict(bundle["streamlit_input_upper_bounds"])

threshold_options = dict(bundle["streamlit_threshold_options"])


# ============================================================
# 3. ÜLKE / ŞEHİR KOORDİNATLARI
# ============================================================

CITY_COORDINATES = {
    "Türkiye": {
        "Antalya": (30.7133, 36.8969),
        "Mersin": (34.6415, 36.8121),
        "Adana": (35.3213, 37.0000),
        "Hatay": (36.2020, 36.2021),
        "Osmaniye": (36.2478, 37.0742),
        "Kahramanmaraş": (36.9371, 37.5753),
        "Isparta": (30.5522, 37.7648),
        "Burdur": (30.2908, 37.7203),
        "Muğla": (28.3636, 37.2153),
        "İzmir": (27.1428, 38.4237),
    },
    "Yunanistan": {
        "Atina": (23.7275, 37.9838),
        "Selanik": (22.9444, 40.6401),
        "Patras": (21.7346, 38.2466),
        "Kalamata": (22.1142, 37.0389),
    },
    "İtalya": {
        "Roma": (12.4964, 41.9028),
        "Napoli": (14.2681, 40.8518),
        "Bari": (16.8719, 41.1171),
        "Palermo": (13.3614, 38.1157),
        "Catania": (15.0873, 37.5027),
    },
    "İspanya": {
        "Barselona": (2.1734, 41.3851),
        "Valensiya": (-0.3763, 39.4699),
        "Malaga": (-4.4214, 36.7213),
        "Alicante": (-0.4907, 38.3452),
    },
    "Fransa": {
        "Marsilya": (5.3698, 43.2965),
        "Nice": (7.2620, 43.7102),
        "Montpellier": (3.8767, 43.6108),
    },
    "Hırvatistan": {
        "Split": (16.4402, 43.5081),
        "Dubrovnik": (18.0944, 42.6507),
        "Zadar": (15.2314, 44.1194),
    },
    "Arnavutluk": {
        "Tiran": (19.8187, 41.3275),
        "Vlora": (19.4897, 40.4661),
        "Dıraç": (19.4565, 41.3231),
    },
    "Karadağ": {
        "Podgoritsa": (19.2629, 42.4304),
        "Bar": (19.1003, 42.0937),
    },
    "Bosna-Hersek": {
        "Mostar": (17.8078, 43.3438),
        "Neum": (17.6156, 42.9233),
    },
    "Slovenya": {
        "Koper": (13.7302, 45.5481),
    },
    "Malta": {
        "Valletta": (14.5146, 35.8989),
    },
    "Kıbrıs": {
        "Lefkoşa": (33.3823, 35.1856),
        "Limasol": (33.0226, 34.6786),
        "Baf": (32.4297, 34.7754),
    },
    "Fas": {
        "Tanca": (-5.8340, 35.7595),
        "Nador": (-2.9335, 35.1681),
    },
    "Cezayir": {
        "Cezayir": (3.0588, 36.7538),
        "Oran": (-0.6417, 35.6971),
    },
    "Tunus": {
        "Tunus": (10.1815, 36.8065),
        "Susa": (10.6346, 35.8256),
    },
    "Libya": {
        "Trablus": (13.1913, 32.8872),
    },
    "Mısır": {
        "İskenderiye": (29.9187, 31.2001),
    },
    "İsrail": {
        "Hayfa": (34.9896, 32.7940),
        "Tel Aviv": (34.7818, 32.0853),
    },
    "Lübnan": {
        "Beyrut": (35.5018, 33.8938),
    },
}

# Mevcut ülke ve şehirleri ÜLKE > İL/ŞEHİR > İLÇE/BÖLGE
# biçiminde kullanıyoruz. Diğer ülkelerde mevcut şehir merkezi korunur.
LOCATION_COORDINATES = {
    country_name: {
        province_name: {"Merkez": coordinates}
        for province_name, coordinates in province_dict.items()
    }
    for country_name, province_dict in CITY_COORDINATES.items()
}

# Türkiye için yaygın Akdeniz ilçe merkezleri.
# Daha kesin bir orman/kırsal noktası için manuel koordinat kullanılabilir.
LOCATION_COORDINATES["Türkiye"].update(
    {
        "Antalya": {
            "Muratpaşa": (30.7133, 36.8969),
            "Manavgat": (31.7249, 36.7867),
            "Alanya": (32.0025, 36.5444),
            "Gazipaşa": (32.3179, 36.2694),
            "Serik": (31.0988, 36.9177),
            "Kemer": (30.5606, 36.6020),
            "Kumluca": (30.2864, 36.3703),
            "Kaş": (29.6377, 36.1996),
        },
        "Mersin": {
            "Akdeniz": (34.6333, 36.8000),
            "Tarsus": (34.8951, 36.9177),
            "Erdemli": (34.3080, 36.6050),
            "Silifke": (33.9344, 36.3778),
            "Anamur": (32.8364, 36.0751),
            "Gülnar": (33.3992, 36.3415),
            "Mut": (33.4384, 36.6455),
        },
        "Adana": {
            "Seyhan": (35.3213, 37.0000),
            "Ceyhan": (35.8170, 37.0240),
            "Kozan": (35.8157, 37.4552),
            "Karaisalı": (35.0560, 37.2560),
            "Pozantı": (34.8710, 37.4280),
        },
        "Hatay": {
            "Antakya": (36.1613, 36.2021),
            "İskenderun": (36.1735, 36.5872),
            "Dörtyol": (36.2280, 36.8390),
            "Arsuz": (35.8900, 36.4130),
            "Samandağ": (35.9770, 36.0850),
            "Kırıkhan": (36.3570, 36.4990),
        },
        "Osmaniye": {
            "Merkez": (36.2478, 37.0742),
            "Kadirli": (36.0961, 37.3739),
            "Düziçi": (36.4546, 37.2427),
            "Bahçe": (36.5766, 37.2019),
            "Hasanbeyli": (36.5500, 37.1300),
        },
        "Kahramanmaraş": {
            "Onikişubat": (36.9371, 37.5753),
            "Dulkadiroğlu": (36.9500, 37.5800),
            "Andırın": (36.3540, 37.5770),
            "Göksun": (36.4970, 38.0210),
            "Pazarcık": (37.2990, 37.4870),
            "Türkoğlu": (36.8500, 37.3900),
        },
        "Isparta": {
            "Merkez": (30.5522, 37.7648),
            "Eğirdir": (30.8500, 37.8740),
            "Sütçüler": (30.9810, 37.4910),
            "Yalvaç": (31.1770, 38.2950),
            "Şarkikaraağaç": (31.3660, 38.0790),
        },
        "Burdur": {
            "Merkez": (30.2908, 37.7203),
            "Bucak": (30.5950, 37.4590),
            "Gölhisar": (29.5080, 37.1450),
            "Tefenni": (29.7750, 37.3090),
            "Yeşilova": (29.7540, 37.5080),
        },
        "Muğla": {
            "Menteşe": (28.3636, 37.2153),
            "Marmaris": (28.2740, 36.8550),
            "Bodrum": (27.4305, 37.0344),
            "Fethiye": (29.1263, 36.6217),
            "Milas": (27.7830, 37.3160),
            "Datça": (27.6910, 36.7280),
        },
        "İzmir": {
            "Konak": (27.1428, 38.4237),
            "Bornova": (27.2177, 38.4622),
            "Karşıyaka": (27.1097, 38.4550),
            "Seferihisar": (26.8370, 38.1970),
            "Çeşme": (26.3054, 38.3228),
            "Bergama": (27.1800, 39.1200),
        },
    }
)


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================

@st.cache_data(ttl=10)
def get_api_health():
    """Flask API'nin açık olup olmadığını kontrol eder."""
    response = requests.get(f"{API_URL}/health", timeout=4)
    response.raise_for_status()
    return response.json()


def fetch_current_weather(latitude, longitude, risk_date):
    """Seçilen koordinat için API üzerinden meteoroloji verisini getirir."""
    response = requests.get(
        f"{API_URL}/weather",
        params={
            "latitude": float(latitude),
            "longitude": float(longitude),
            "risk_date": risk_date.isoformat(),
        },
        timeout=35,
    )

    if response.status_code != 200:
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(message)

    return response.json()


def get_default(name, fallback=0.0):
    value = defaults.get(name, fallback)
    if pd.isna(value):
        return fallback
    return float(value)


def bounded_number_input(
    label,
    key,
    step=0.1,
    help_text=None,
    format_string="%.3f",
):
    default_value = get_default(key)

    lower = lower_bounds.get(key, None)
    upper = upper_bounds.get(key, None)

    min_value = float(lower) if lower is not None and np.isfinite(lower) else None
    max_value = float(upper) if upper is not None and np.isfinite(upper) else None

    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=float(np.clip(
            default_value,
            min_value if min_value is not None else default_value,
            max_value if max_value is not None else default_value,
        )),
        step=step,
        format=format_string,
        help=help_text,
    )


def calculate_user_features(row, forecast_date):
    """
    Kullanıcının girdiği temel değişkenlerden üretilebilen feature'ları
    yeniden hesaplar.

    3/7/30/59 günlük geçmiş özetleri kullanıcı tarafından sağlanmadığı
    için eğitim medyanlarında bırakılır.
    """

    # Zaman feature'ları
    month = forecast_date.month
    day_of_year = forecast_date.timetuple().tm_yday

    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)
    row["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    row["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    row["is_fire_season"] = int(5 <= month <= 10)

    # Meteorolojik türetmeler
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
                np.log1p(max(ssrd, 0))
                * (1 - np.clip(rh, 0, 1))
            )

    if np.isfinite(t2m) and np.isfinite(smi):
        row["heat_soil_dryness"] = max(t2m, 0) * (1 - np.clip(smi, 0, 1))

    if np.isfinite(rh) and np.isfinite(smi):
        row["atmosphere_soil_dryness"] = (
            (1 - np.clip(rh, 0, 1))
            * (1 - np.clip(smi, 0, 1))
        )

    # Kara yüzeyi sıcaklığı türetmeleri
    lst_day = row.get("lst_day", np.nan)
    lst_night = row.get("lst_night", np.nan)

    if np.isfinite(lst_day) and np.isfinite(lst_night):
        row["lst_day_night_difference"] = lst_day - lst_night
        row["lst_mean"] = (lst_day + lst_night) / 2

    if np.isfinite(lst_day) and np.isfinite(t2m):
        row["lst_air_temperature_difference"] = lst_day - t2m

    if np.isfinite(lst_night) and np.isfinite(t2m):
        row["lst_night_air_difference"] = lst_night - t2m

    # Arazi / bitki feature'ları
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
        valid_probs = probabilities[probabilities > 0]
        entropy = -(valid_probs * np.log(valid_probs)).sum()
        row["land_cover_diversity"] = np.clip(
            entropy / np.log(len(land_cover_values)),
            0,
            1,
        )

    ndvi = row.get("ndvi", np.nan)

    if np.isfinite(smi):
        soil_dryness = 1 - np.clip(smi, 0, 1)

        row["forest_dryness"] = land_cover_values["lc_forest"] * soil_dryness
        row["shrubland_dryness"] = land_cover_values["lc_shrubland"] * soil_dryness
        row["grassland_dryness"] = land_cover_values["lc_grassland"] * soil_dryness
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

    if "hot_dry_windy_index" in row and np.isfinite(row["hot_dry_windy_index"]):
        row["forest_hot_dry_windy"] = (
            land_cover_values["lc_forest"] * row["hot_dry_windy_index"]
        )

    # İnsan etkisi
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

    if np.isfinite(land_cover_values["lc_forest"]):
        row["forest_settlement_interaction"] = (
            land_cover_values["lc_forest"]
            * land_cover_values["lc_settlement"]
        )

        if "human_pressure" in row:
            row["forest_human_pressure"] = (
                land_cover_values["lc_forest"] * row["human_pressure"]
            )

    # Yamaç ve rüzgâr yönü kullanıcıya bu ilk sürümde açılmıyor.
    # Bu feature'lar model bundle'ındaki eğitim medyanlarında kalır.

    return row


def build_model_row(user_inputs, forecast_date):
    # Önce bütün feature'ları eğitim medyanlarıyla oluştur.
    row = {
        feature: float(defaults.get(feature, 0.0))
        for feature in feature_columns
    }

    # Kullanıcının verdiği değerleri medyanların üstüne yaz.
    for key, value in user_inputs.items():
        if key in row:
            row[key] = float(value)

    row = calculate_user_features(row, forecast_date)

    frame = pd.DataFrame([row])

    # Modelin beklediği kolon sırasını garanti et.
    frame = frame.reindex(columns=feature_columns)

    return frame


def get_out_of_range_inputs(user_inputs):
    warnings_list = []

    for key, value in user_inputs.items():
        if key not in lower_bounds or key not in upper_bounds:
            continue

        lower = lower_bounds[key]
        upper = upper_bounds[key]

        if pd.isna(lower) or pd.isna(upper):
            continue

        if value < lower or value > upper:
            warnings_list.append(
                {
                    "Değişken": feature_names_tr.get(key, key),
                    "Girilen Değer": value,
                    "Eğitim Alt Sınırı": lower,
                    "Eğitim Üst Sınırı": upper,
                }
            )

    return warnings_list


def risk_level(score):
    if score < 30:
        return "Düşük", "status-safe"
    if score < 55:
        return "İzlenmeli", "status-watch"
    if score < 75:
        return "Yüksek", "status-alert"
    return "Çok Yüksek", "status-high"


# ============================================================
# 5. BAŞLIK VE KISA AÇIKLAMA
# ============================================================

# Not: Özel HTML başlık yerine Streamlit'in kendi başlığı kullanılır.
# Emoji page_icon içinde kalır. Böylece satır yüksekliği bozulmaz.
st.title("Akdeniz Yangın Erken Uyarı Destek Sistemi")
st.caption(
    "Mesogeos verisiyle eğitilmiş LightGBM modeli kullanılarak, "
    "seçilen konum ve çevresel koşullar için ertesi gün yangın risk skoru üretilir."
)

st.info(
    "Bu uygulama bir **portfolyo / karar destek demosudur**. "
    "Çıktı, kalibre edilmiş bir yüzde olasılık değil; **0–100 arası yangın risk skorudur**."
)

with st.expander("📘 Modeli kısaca tanı", expanded=False):
    st.write(f"**Seçilen model:** {bundle.get('model_name', 'LightGBM')}")
    st.write(
        f"**Train / Validation / Test:** "
        f"{bundle.get('train_years', '2006-2020')} / "
        f"{bundle.get('validation_year', 2021)} / "
        f"{bundle.get('test_year', 2022)}"
    )
    st.write(f"**Geçmiş pencere:** {bundle.get('history_days', 59)} gün")
    st.write(f"**Model feature sayısı:** {len(feature_columns)}")
    st.write(
        "Model; meteoroloji, bitki-toprak, topoğrafya, insan etkisi, konum ve "
        "geçmiş dönem özetlerini birlikte değerlendirir."
    )


# ============================================================
# 6. SOL MENÜ: KONUM VE UYARI POLİTİKASI
# ============================================================

with st.sidebar:
    st.header("1️⃣ Konum Seçimi")

    try:
        api_health = get_api_health()
        st.success(
            "Flask API bağlı · "
            f"{api_health.get('model_name', 'Model hazır')}"
        )
    except requests.RequestException:
        st.warning(
            "Flask API bağlantısı bekleniyor. Önce ayrı terminalde "
            "`python api_app.py` komutunu çalıştırın."
        )

    country = st.selectbox(
        "Ülke",
        list(LOCATION_COORDINATES.keys()),
        index=0,
    )

    province_names = list(LOCATION_COORDINATES[country].keys())
    province = st.selectbox("İl / Şehir", province_names, index=0)

    district_names = list(
        LOCATION_COORDINATES[country][province].keys()
    )
    district = st.selectbox(
        "İlçe / Bölge",
        district_names,
        index=0,
    )

    city_lon, city_lat = (
        LOCATION_COORDINATES[country][province][district]
    )

    manual_location = st.checkbox(
        "Koordinatı elle girmek istiyorum",
        value=False,
    )

    if manual_location:
        longitude = st.number_input(
            "Boylam",
            value=float(city_lon),
            step=0.01,
            format="%.4f",
        )
        latitude = st.number_input(
            "Enlem",
            value=float(city_lat),
            step=0.01,
            format="%.4f",
        )
    else:
        longitude = float(city_lon)
        latitude = float(city_lat)

    st.caption(
        "Seçim ilçe merkezinin koordinatına dönüşür. "
        "Ülke, il ve ilçe adları model feature'ı değildir."
    )

    forecast_date = st.date_input(
        "Tahmin edilecek gün",
        value=date.today() + timedelta(days=1),
        help=(
            "Model ertesi günü tahmin eder. Bu nedenle canlı meteoroloji, "
            "seçilen tarihten bir önceki gün için alınır."
        ),
    )

    current_location_key = (country, province, district)
    previous_weather_location = st.session_state.get("weather_location_key")

    if (
        previous_weather_location is not None
        and previous_weather_location != current_location_key
    ):
        for state_key in [
            "api_t2m",
            "api_d2m",
            "api_rh_percent",
            "api_smi_percent",
            "api_tp_mm",
            "api_wind_speed",
            "api_ssrd",
            "api_sp",
            "api_dem",
            "weather_observation_date",
            "weather_location_key",
        ]:
            st.session_state.pop(state_key, None)

    fetch_weather_button = st.button(
        "🌤️ Güncel meteorolojiyi getir",
        use_container_width=True,
    )

    if fetch_weather_button:
        try:
            weather_result = fetch_current_weather(
                latitude=latitude,
                longitude=longitude,
                risk_date=forecast_date,
            )
            weather_values = weather_result["values"]

            st.session_state["api_t2m"] = float(weather_values["t2m"])
            st.session_state["api_d2m"] = float(weather_values["d2m"])
            st.session_state["api_rh_percent"] = int(
                weather_values["rh_percent"]
            )
            st.session_state["api_smi_percent"] = int(
                weather_values["smi_percent"]
            )
            st.session_state["api_tp_mm"] = float(weather_values["tp_mm"])
            st.session_state["api_wind_speed"] = float(
                weather_values["wind_speed"]
            )
            st.session_state["api_ssrd"] = float(weather_values["ssrd"])
            st.session_state["api_sp"] = float(weather_values["sp"])
            st.session_state["api_dem"] = float(weather_values["dem"])

            st.session_state["weather_observation_date"] = (
                weather_result["observation_date"]
            )
            st.session_state["weather_location_key"] = (
                country,
                province,
                district,
            )

            st.success(
                "Meteoroloji getirildi · Veri tarihi: "
                f"{weather_result['observation_date']}"
            )
        except (requests.RequestException, RuntimeError, KeyError) as error:
            st.error(f"Meteoroloji alınamadı: {error}")

    if "weather_observation_date" in st.session_state:
        st.caption(
            "Otomatik meteoroloji veri tarihi: "
            f"{st.session_state['weather_observation_date']}. "
            "NDVI, LAI, uydu yüzey sıcaklığı ve arazi oranları "
            "otomatik değiştirilmez."
        )

    st.divider()
    st.header("2️⃣ Uyarı Politikası")

    policy_labels = {
        "Maliyet - Validation 2021": "0.13 · Operasyonel (önerilen)",
        "F2 - Validation 2021": "0.16 · Dengeli",
        "Walk-forward 2017-2021": "0.11 · Daha hassas",
        "Varsayılan 0.50": "0.50 · Klasik eşik",
    }

    policy_guidance = {
        "Maliyet - Validation 2021": (
            "Kaçırılan yüksek tehlikeye, gereksiz alarmdan daha fazla "
            "önem verir.",
            "Günlük ve genel kullanım için bu seçeneği tercih edin. "
            "Ekip hem yangın kaçırmak istemiyor hem de alarm sayısını "
            "yönetilebilir tutmak istiyorsa uygundur.",
            "Yangınları yakalama ile yanlış alarm maliyeti arasında "
            "operasyonel denge kurar.",
        ),
        "F2 - Validation 2021": (
            "Yangınları yakalamaya, alarm doğruluğundan daha fazla "
            "önem veren dengeli model eşiğidir.",
            "Model sonuçlarını sunarken, modelleri karşılaştırırken "
            "ve tek bir dengeli karar eşiği istenirken seçin.",
            "Hassasiyet korunur; 0.11 ve 0.13'e göre daha az alarm "
            "üretme eğilimindedir.",
        ),
        "Walk-forward 2017-2021": (
            "Yıllar arasındaki sonuçlara göre daha hassas alarm "
            "üretmeyi amaçlar.",
            "Aşırı sıcak, kuvvetli rüzgâr veya ciddi kuraklık dönemlerinde; "
            "bir yangını kaçırmanın maliyeti çok yüksekse seçin.",
            "Daha fazla tehlikeli durumu yakalayabilir; ekip daha fazla "
            "yanlış alarmı incelemek zorunda kalabilir.",
        ),
        "Varsayılan 0.50": (
            "Yalnızca modelin en güçlü risk sinyallerinde alarm vermesine "
            "yakın klasik eşiktir.",
            "Saha ve inceleme kapasitesi çok kısıtlıysa ve yalnızca en "
            "güçlü alarmlar incelenebilecekse seçin.",
            "Yanlış alarmı azaltabilir; ancak yangın kaçırma riski daha "
            "yüksektir. Ana erken uyarı eşiği olarak önerilmez.",
        ),
    }

    policy_tradeoffs = {
        "Maliyet - Validation 2021": {
            "Yangın yakalama": "Çok yüksek — hassas seçeneğe yakın koruma sağlar.",
            "Alarm sayısı": "Yüksek fakat 0.11'e göre daha kontrollüdür.",
            "Yangın kaçırma riski": "Düşüktür.",
            "Operasyon maliyeti": "Dengelidir — günlük kullanım için önerilir.",
        },
        "F2 - Validation 2021": {
            "Yangın yakalama": "Yüksektir, ancak 0.11 ve 0.13'ten daha seçicidir.",
            "Alarm sayısı": "0.11 ve 0.13'e göre daha azdır.",
            "Yangın kaçırma riski": "Bir miktar artar.",
            "Operasyon maliyeti": "Orta düzeydedir — daha az alarm incelenir.",
        },
        "Walk-forward 2017-2021": {
            "Yangın yakalama": "En yüksektir — daha fazla riskli durum yakalanır.",
            "Alarm sayısı": "En fazladır.",
            "Yangın kaçırma riski": "En düşüktür.",
            "Operasyon maliyeti": "En yüksektir — daha fazla alarm incelenir.",
        },
        "Varsayılan 0.50": {
            "Yangın yakalama": "En düşüktür — yalnızca güçlü sinyaller seçilir.",
            "Alarm sayısı": "En azdır.",
            "Yangın kaçırma riski": "En yüksektir.",
            "Operasyon maliyeti": "Kısa vadede düşüktür; kaçırılan yangın riski yüksektir.",
        },
    }

    available_policy_keys = [
        key
        for key in [
            "Maliyet - Validation 2021",
            "F2 - Validation 2021",
            "Walk-forward 2017-2021",
            "Varsayılan 0.50",
        ]
        if key in threshold_options
    ]

    selected_policy = st.selectbox(
        "Karar yaklaşımı",
        available_policy_keys,
        index=0,
        format_func=lambda x: policy_labels.get(x, x),
    )

    selected_threshold = float(threshold_options[selected_policy])

    policy_meaning, choose_when, expected_result = policy_guidance[
        selected_policy
    ]

    st.caption(policy_meaning)
    st.markdown(
        f"**Ne zaman seçilmeli?**  \n{choose_when}"
    )
    st.markdown(
        f"**Beklenen sonuç:**  \n{expected_result}"
    )

    selected_tradeoffs = policy_tradeoffs[selected_policy]
    st.markdown("**Seçimin pratik etkisi:**")
    st.markdown(
        "\n".join(
            [
                f"- 🔥 **Yangın yakalama:** {selected_tradeoffs['Yangın yakalama']}",
                f"- 🚨 **Alarm sayısı:** {selected_tradeoffs['Alarm sayısı']}",
                f"- ⚠️ **Yangın kaçırma riski:** {selected_tradeoffs['Yangın kaçırma riski']}",
                f"- 👷 **Operasyon maliyeti:** {selected_tradeoffs['Operasyon maliyeti']}",
            ]
        )
    )


# ============================================================
# 7. ANA EKRAN: GİRDİLER
# ============================================================

st.subheader("3️⃣ Bugünkü Koşulları Gir")
st.write(
    "Aşağıdaki alanlar teknik olmayan bir kullanıcının da doldurabileceği şekilde sadeleştirildi. "
    "İstersen gelişmiş değerleri değiştirmeden bırakabilirsin."
)

# Kullanıcı dostu varsayılanlar
rh_percent_default = int(round(get_default("rh", 0.40) * 100))
smi_percent_default = int(round(get_default("smi", 0.40) * 100))
tp_mm_default = get_default("tp", 0.0) * 1000

widget_defaults = {
    "api_t2m": round(get_default("t2m", 25.0), 1),
    "api_d2m": round(get_default("d2m", 15.0), 1),
    "api_lst_day": round(get_default("lst_day", 30.0), 1),
    "api_lst_night": round(get_default("lst_night", 18.0), 1),
    "api_rh_percent": max(0, min(100, rh_percent_default)),
    "api_smi_percent": max(0, min(100, smi_percent_default)),
    "api_tp_mm": max(0.0, round(tp_mm_default, 2)),
    "api_wind_speed": max(0.0, round(get_default("wind_speed", 3.0), 1)),
    "api_ndvi": float(np.clip(get_default("ndvi", 0.45), -1, 1)),
    "api_lai": max(0.0, round(get_default("lai", 1.0), 2)),
    "api_forest_percent": int(
        np.clip(round(get_default("lc_forest", 0.35) * 100), 0, 100)
    ),
    "api_shrub_percent": int(
        np.clip(round(get_default("lc_shrubland", 0.10) * 100), 0, 100)
    ),
    "api_dem": round(get_default("dem", 600.0), 1),
    "api_slope": max(0.0, round(get_default("slope", 1.5), 2)),
    "api_roads_distance": max(
        0.0,
        round(get_default("roads_distance", 3.0), 2),
    ),
    "api_population": max(0.0, round(get_default("population", 40.0), 1)),
    "api_agriculture_percent": int(
        np.clip(round(get_default("lc_agriculture", 0.10) * 100), 0, 100)
    ),
    "api_grassland_percent": int(
        np.clip(round(get_default("lc_grassland", 0.10) * 100), 0, 100)
    ),
    "api_settlement_percent": int(
        np.clip(round(get_default("lc_settlement", 0.05) * 100), 0, 100)
    ),
    "api_ssrd": max(0.0, round(get_default("ssrd", 12000000.0), 0)),
    "api_sp": max(0.0, round(get_default("sp", 95000.0), 0)),
}

for widget_key, widget_value in widget_defaults.items():
    if widget_key not in st.session_state:
        st.session_state[widget_key] = widget_value

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🌡️ Sıcaklık")
    t2m = st.number_input(
        "Hava sıcaklığı (°C)",
        key="api_t2m",
        step=0.5,
        help="2 metre yükseklikte ölçülen hava sıcaklığı.",
    )
    d2m = st.number_input(
        "Çiğ noktası sıcaklığı (°C)",
        key="api_d2m",
        step=0.5,
        help="Havadaki nemi anlamaya yardımcı olan sıcaklık göstergesi.",
    )
    lst_day = st.number_input(
        "Gündüz yüzey sıcaklığı (°C)",
        key="api_lst_day",
        step=0.5,
        help="Uydu tarafından görülen yer yüzeyi sıcaklığı.",
    )
    lst_night = st.number_input(
        "Gece yüzey sıcaklığı (°C)",
        key="api_lst_night",
        step=0.5,
    )

with col2:
    st.markdown("#### 💧 Nem ve Yağış")
    rh_percent = st.slider(
        "Bağıl nem (%)",
        min_value=0,
        max_value=100,
        key="api_rh_percent",
        step=1,
        help="Düşük nem, yangın riskini artırabilecek koşullardan biridir.",
    )
    smi_percent = st.slider(
        "Toprak nemi (%)",
        min_value=0,
        max_value=100,
        key="api_smi_percent",
        step=1,
        help="Toprak kurudukça bitki ve yüzey koşulları yangına daha elverişli olabilir.",
    )
    tp_mm = st.number_input(
        "Yağış (mm)",
        min_value=0.0,
        key="api_tp_mm",
        step=0.1,
        help="Model yağışı metre cinsinden kullanır; uygulama otomatik dönüştürür.",
    )
    wind_speed = st.number_input(
        "Rüzgâr hızı (m/s)",
        min_value=0.0,
        key="api_wind_speed",
        step=0.2,
    )

with col3:
    st.markdown("#### 🌿 Bitki Örtüsü")
    ndvi = st.number_input(
        "NDVI / bitki canlılığı",
        min_value=-1.0,
        max_value=1.0,
        key="api_ndvi",
        step=0.01,
        format="%.2f",
        help="Bitkinin ne kadar canlı/yeşil olduğunu gösteren uydu temelli indeks.",
    )
    lai = st.number_input(
        "LAI / yaprak alanı",
        min_value=0.0,
        key="api_lai",
        step=0.1,
        help="Bitki örtüsünün yaprak yoğunluğunu temsil eder.",
    )
    forest_percent = st.slider(
        "Orman alanı (%)",
        min_value=0,
        max_value=100,
        key="api_forest_percent",
        step=1,
    )
    shrub_percent = st.slider(
        "Çalılık alan (%)",
        min_value=0,
        max_value=100,
        key="api_shrub_percent",
        step=1,
    )

# Model 0-1 aralığında kullanıyor.
rh = rh_percent / 100.0
smi = smi_percent / 100.0
tp = tp_mm / 1000.0
lc_forest = forest_percent / 100.0
lc_shrubland = shrub_percent / 100.0

with st.expander("⚙️ Gelişmiş arazi ve teknik değerler", expanded=False):
    st.caption(
        "Bu alanları bilmiyorsan değiştirmene gerek yok. Eğitim verisinin tipik değerleri ile başlatılır."
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        dem = st.number_input(
            "Yükseklik (m)",
            key="api_dem",
            step=10.0,
        )
        slope = st.number_input(
            "Eğim",
            min_value=0.0,
            key="api_slope",
            step=0.05,
        )
        roads_distance = st.number_input(
            "Yola uzaklık",
            min_value=0.0,
            key="api_roads_distance",
            step=0.1,
        )

    with a2:
        population = st.number_input(
            "Nüfus yoğunluğu",
            min_value=0.0,
            key="api_population",
            step=1.0,
        )
        agriculture_percent = st.slider(
            "Tarım alanı (%)",
            0, 100,
            key="api_agriculture_percent",
        )
        grassland_percent = st.slider(
            "Otlak alanı (%)",
            0, 100,
            key="api_grassland_percent",
        )

    with a3:
        settlement_percent = st.slider(
            "Yerleşim alanı (%)",
            0, 100,
            key="api_settlement_percent",
        )
        ssrd = st.number_input(
            "Güneş radyasyonu (J/m²)",
            min_value=0.0,
            key="api_ssrd",
            step=100000.0,
            format="%.0f",
        )
        sp = st.number_input(
            "Yüzey basıncı (Pa)",
            min_value=0.0,
            key="api_sp",
            step=100.0,
            format="%.0f",
        )

lc_agriculture = agriculture_percent / 100.0
lc_grassland = grassland_percent / 100.0
lc_settlement = settlement_percent / 100.0


# ============================================================
# 8. MODEL GİRDİSİ
# ============================================================

user_inputs = {
    "x": float(longitude),
    "y": float(latitude),
    "t2m": float(t2m),
    "d2m": float(d2m),
    "rh": float(rh),
    "tp": float(tp),
    "wind_speed": float(wind_speed),
    "ssrd": float(ssrd),
    "sp": float(sp),
    "smi": float(smi),
    "ndvi": float(ndvi),
    "lai": float(lai),
    "lst_day": float(lst_day),
    "lst_night": float(lst_night),
    "dem": float(dem),
    "slope": float(slope),
    "roads_distance": float(roads_distance),
    "population": float(population),
    "lc_forest": float(lc_forest),
    "lc_shrubland": float(lc_shrubland),
    "lc_agriculture": float(lc_agriculture),
    "lc_grassland": float(lc_grassland),
    "lc_settlement": float(lc_settlement),
}

out_of_range = get_out_of_range_inputs(user_inputs)


# ============================================================
# 9. KONUM HARİTASI
# ============================================================

st.subheader("4️⃣ Seçilen Konumu Kontrol Et")

map_df = pd.DataFrame(
    {
        "lon": [longitude],
        "lat": [latitude],
        "Konum": [f"{district}, {province}, {country}"],
    }
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[lon, lat]",
    get_radius=11000,
    get_fill_color=[230, 70, 30, 210],
    pickable=True,
)

view_state = pdk.ViewState(
    longitude=float(longitude),
    latitude=float(latitude),
    zoom=6.0,
    pitch=0,
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{Konum}"},
    ),
    use_container_width=True,
)


# ============================================================
# 10. TAHMİN VE SONUÇ
# ============================================================

st.subheader("5️⃣ Ertesi Gün Riskini Hesapla")

predict_button = st.button(
    "🔥 Yangın Risk Skorunu Hesapla",
    type="primary",
    use_container_width=True,
)

if predict_button:
    api_payload = {
        "location": f"{district}, {province}, {country}",
        "risk_date": forecast_date.isoformat(),
        "threshold_policy": selected_policy,
        "inputs": {
            "x": float(longitude),
            "y": float(latitude),
            "t2m": float(t2m),
            "d2m": float(d2m),
            "rh_percent": int(rh_percent),
            "smi_percent": int(smi_percent),
            "tp_mm": float(tp_mm),
            "wind_speed": float(wind_speed),
            "ndvi": float(ndvi),
            "lai": float(lai),
            "lst_day": float(lst_day),
            "lst_night": float(lst_night),
            "lc_forest_percent": int(forest_percent),
            "lc_shrubland_percent": int(shrub_percent),
            "dem": float(dem),
            "slope": float(slope),
            "roads_distance": float(roads_distance),
            "population": float(population),
            "lc_agriculture_percent": int(agriculture_percent),
            "lc_grassland_percent": int(grassland_percent),
            "lc_settlement_percent": int(settlement_percent),
            "ssrd": float(ssrd),
            "sp": float(sp),
        },
    }

    try:
        api_response = requests.post(
            f"{API_URL}/predict",
            json=api_payload,
            timeout=40,
        )

        try:
            api_result = api_response.json()
        except ValueError as error:
            raise RuntimeError(
                "API geçerli bir JSON cevabı döndürmedi."
            ) from error

        if api_response.status_code != 200:
            raise RuntimeError(
                api_result.get("message", "API tahmin üretemedi.")
            )

    except requests.RequestException as exc:
        st.error(
            "Flask API'ye ulaşılamadı. Ayrı terminalde "
            "`python api_app.py` komutunun çalıştığını kontrol edin."
        )
        st.exception(exc)
        st.stop()
    except RuntimeError as exc:
        st.error(f"API tahmini sırasında hata oluştu: {exc}")
        st.stop()

    risk_score = float(api_result["risk_score"])
    raw_score = risk_score / 100.0
    alert = bool(api_result["alarm"])
    selected_threshold = float(api_result["threshold"])
    api_model_name = api_result.get(
        "model_name",
        bundle.get("model_name", "Model"),
    )

    if risk_score < 30:
        risk_text = "Düşük"
        risk_color = "#2A9D8F"
        risk_message = "Koşullar şu anda görece düşük risk gösteriyor."
    elif risk_score < 55:
        risk_text = "Orta"
        risk_color = "#E9C46A"
        risk_message = "Koşullar izlenmeli; çevresel değerlerdeki değişimler takip edilmeli."
    elif risk_score < 75:
        risk_text = "Yüksek"
        risk_color = "#F4A261"
        risk_message = "Yangın açısından dikkat gerektiren koşullar oluşmuş olabilir."
    else:
        risk_text = "Çok Yüksek"
        risk_color = "#E63946"
        risk_message = "Risk skoru yüksek. Erken uyarı ve saha takibi önceliklendirilmeli."

    r1, r2, r3 = st.columns(3)
    r1.metric("🔥 Yangın Risk Skoru", f"{risk_score:.1f} / 100")
    r2.metric("📌 Risk Düzeyi", risk_text)
    r3.metric("🎯 Karar Eşiği", f"{selected_threshold:.2f}")

    st.markdown(
        f"""
        <div style="
            border-left: 8px solid {risk_color};
            padding: 14px 18px;
            border-radius: 8px;
            background-color: rgba(120,120,120,0.08);
            margin-top: 8px;
            margin-bottom: 12px;">
            <b>{risk_message}</b><br>
            Operasyonel karar: <b>{'UYARI ÜRET' if alert else 'UYARI ÜRETME / İZLE'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(round(risk_score)))

    # Basit açıklama tablosu
    st.markdown("#### 🔎 Hangi koşullar dikkat çekiyor?")

    explanation_rows = []

    def add_comparison(label, key, current, high_is_risky=True):
        median = get_default(key)
        if current > median:
            direction = "Eğitim medyanının üzerinde"
            risk_direction = high_is_risky
        elif current < median:
            direction = "Eğitim medyanının altında"
            risk_direction = not high_is_risky
        else:
            direction = "Eğitim medyanına yakın"
            risk_direction = False

        explanation_rows.append(
            {
                "Gösterge": label,
                "Girilen": round(current, 3),
                "Tipik Eğitim Değeri": round(median, 3),
                "Yorum": direction,
                "Risk Açısından": "Dikkat" if risk_direction else "Normal / tek başına belirleyici değil",
            }
        )

    add_comparison("Hava sıcaklığı", "t2m", t2m, True)
    add_comparison("Bağıl nem", "rh", rh, False)
    add_comparison("Yağış", "tp", tp, False)
    add_comparison("Toprak nemi", "smi", smi, False)
    add_comparison("Gündüz yüzey sıcaklığı", "lst_day", lst_day, True)
    add_comparison("Orman alanı", "lc_forest", lc_forest, True)

    st.dataframe(
        pd.DataFrame(explanation_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Bu tablo SHAP analizi değildir. Girilen değerleri eğitim medyanlarıyla karşılaştıran "
        "sade bir kullanıcı açıklamasıdır. Nihai risk skoru tüm model feature'larının birlikte değerlendirilmesiyle oluşur."
    )

    st.markdown("#### ✅ Girdi kontrolü")
    if out_of_range:
        st.warning(
            "Bazı girdiler eğitim verisinin tipik %1–%99 aralığının dışında. "
            "Bu durumda skor daha dikkatli yorumlanmalıdır."
        )
        st.dataframe(pd.DataFrame(out_of_range), use_container_width=True, hide_index=True)
    else:
        st.success("Girilen temel değerler eğitim verisinin tipik aralıkları içinde.")

    with st.expander("🧪 Teknik ayrıntılar", expanded=False):
        st.write(f"**Tahmin kaynağı:** Flask API ({API_URL})")
        st.write(f"**Model:** {api_model_name}")
        st.write(f"**Feature sayısı:** {len(feature_columns)}")
        st.write(f"**Ham model skoru:** {raw_score:.6f}")
        st.write(f"**Seçilen eşik:** {selected_threshold:.2f}")
        st.write(
            "Kullanıcının girmediği 3/7/30/59 günlük geçmiş özetleri "
            "model paketindeki eğitim medyanlarıyla tamamlanır."
        )


# ============================================================
# 11. UYGULAMA NOTU
# ============================================================

st.divider()
st.caption(
    "📌 Portfolyo sürümü: Bu ekranda temel değerler elle girilir. "
    "Gerçek günlük kullanımda meteoroloji ve uydu servislerinden 59 günlük geçmiş otomatik alınarak "
    "tüm feature'lar canlı biçimde üretilebilir."
)
