"""
AKDENİZ YANGIN ERKEN UYARI DESTEK SİSTEMİ
Başlangıç düzeyi / MIUUL tarzı sıralı sürüm

Bu dosya yukarıdan aşağıya doğru çalıştırılır.
Her # %% başlığı Spyder ve Jupyter'da ayrı bir hücre gibi kullanılabilir.

Tahmin mantığı:
    time_idx = 0..58  -> modelin bildiği 59 günlük geçmiş
    time_idx = 59     -> tahmin edilen hedef gün

Önemli:
    burned_area_has, burned_areas ve ignition_points modele verilmez.
    Validation 2021'de model/eşik seçilir.
    Test 2022 yalnızca son değerlendirme için kullanılır.
"""


# %% =========================================================
# 1. KÜTÜPHANELER
# ============================================================

from pathlib import Path
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# %% =========================================================
# 2. AYARLAR
# ============================================================

RANDOM_STATE = 17
N_JOBS = -1

# İlk denemede True yapabilirsiniz. Ağaç sayıları azalır ve daha hızlı çalışır.
FAST_MODE = "--fast" in sys.argv

# Grafiklerin yalnızca kaydedilmesini isterseniz çalıştırırken --no-show yazın.
SHOW_PLOTS = "--no-show" not in sys.argv

# Tam çalışmada 2017-2021 walk-forward analizi yapılır.
RUN_WALK_FORWARD = "--skip-walk-forward" not in sys.argv

# Bir yangını kaçırmak, yanlış alarmdan 5 kat maliyetli kabul edilmiştir.
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
MINIMUM_RECALL = 0.85

# Python dosyası ve CSV'ler aynı klasördeyse bu kısmı değiştirmeyin.
if "__file__" in globals():
    DATA_PATH = Path(__file__).resolve().parent
else:
    DATA_PATH = Path.cwd()

OUTPUT_PATH = DATA_PATH / "yangin_model_ciktilari_basit"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

NO_FIRE_COLOR = "#2A9D8F"
FIRE_COLOR = "#D62828"
BLUE_COLOR = "#457B9D"
ORANGE_COLOR = "#E76F51"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda value: f"{value:.3f}")
sns.set_theme(style="whitegrid")


# %% =========================================================
# 3. DEĞİŞKEN GRUPLARI
# ============================================================

STATIC_FEATURES = [
    "aspect", "slope", "dem", "curvature", "roads_distance", "x", "y",
    "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement",
    "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies",
    "lc_wetland", "population",
]

DYNAMIC_FEATURES = [
    "t2m", "d2m", "rh", "wind_speed", "ssrd", "sp", "tp",
    "lst_day", "lst_night", "smi", "lai", "ndvi",
]

LEAKAGE_COLUMNS = [
    "burned_area_has", "burned_areas", "ignition_points",
]

TURKISH_NAMES = {
    "x": "Boylam",
    "y": "Enlem",
    "t2m": "Hava sıcaklığı",
    "d2m": "Çiğ noktası sıcaklığı",
    "rh": "Bağıl nem",
    "wind_speed": "Rüzgâr hızı",
    "ssrd": "Güneş radyasyonu",
    "sp": "Yüzey basıncı",
    "tp": "Yağış",
    "lst_day": "Gündüz yüzey sıcaklığı",
    "lst_night": "Gece yüzey sıcaklığı",
    "smi": "Toprak nemi",
    "lai": "Yaprak alanı",
    "ndvi": "Bitki canlılığı",
    "dem": "Yükseklik",
    "slope": "Eğim",
    "aspect": "Yamaç yönü",
    "curvature": "Arazi eğriliği",
    "roads_distance": "Yola uzaklık",
    "population": "Nüfus yoğunluğu",
    "lc_agriculture": "Tarım alanı oranı",
    "lc_forest": "Orman alanı oranı",
    "lc_grassland": "Otlak oranı",
    "lc_settlement": "Yerleşim alanı oranı",
    "lc_shrubland": "Çalılık oranı",
    "lc_sparse_vegetation": "Seyrek bitki örtüsü oranı",
    "lc_water_bodies": "Su alanı oranı",
    "lc_wetland": "Sulak alan oranı",
    "temp_dewpoint_difference": "Sıcaklık ve çiğ noktası farkı",
    "vpd": "Havanın kurutucu gücü",
    "dryness_index": "Sıcaklık ve düşük nem kuruluk göstergesi",
    "wind_dryness_interaction": "Rüzgâr ve düşük nem birleşik etkisi",
    "hot_dry_windy_index": "Sıcak-kuru-rüzgârlı hava göstergesi",
    "heat_soil_dryness": "Sıcaklık ve toprak kuruluğu etkisi",
    "atmosphere_soil_dryness": "Hava ve toprak kuruluğu etkisi",
    "radiation_dryness": "Güneş radyasyonu ve kuruluk etkisi",
    "temperature_short_long_change": "Kısa ve uzun dönem sıcaklık değişimi",
    "humidity_short_long_change": "Kısa ve uzun dönem nem değişimi",
    "soil_moisture_short_long_change": "Kısa ve uzun dönem toprak nemi değişimi",
    "elevation_temperature_interaction": "Yükseklik ve sıcaklık etkisi",
    "forest_hot_dry_windy": "Ormanda sıcak-kuru-rüzgârlı hava etkisi",
    "shrubland_dryness": "Çalılık ve toprak kuruluğu etkisi",
    "log_population": "Dengelenmiş nüfus yoğunluğu",
    "human_pressure": "İnsan baskısı göstergesi",
}


# %% =========================================================
# 4. CSV DOSYALARINI BULMA VE OKUMA
# ============================================================

positive_candidates = list(DATA_PATH.glob("positives*.csv"))
negative_candidates = list(DATA_PATH.glob("negatives*.csv"))

if not positive_candidates:
    raise FileNotFoundError("positives.csv bulunamadı.")
if not negative_candidates:
    raise FileNotFoundError("negatives.csv bulunamadı.")

# Aynı dosyanın birden fazla kopyası varsa en büyük olan tam veri kabul edilir.
POSITIVE_FILE = max(positive_candidates, key=lambda path: path.stat().st_size)
NEGATIVE_FILE = max(negative_candidates, key=lambda path: path.stat().st_size)

print("Positive dosyası:", POSITIVE_FILE.name)
print("Negative dosyası:", NEGATIVE_FILE.name)

positive = pd.read_csv(POSITIVE_FILE, low_memory=False)
negative = pd.read_csv(NEGATIVE_FILE, low_memory=False)

positive["fire"] = 1
negative["fire"] = 0
positive["sample_uid"] = "P_" + positive["sample"].astype(str)
negative["sample_uid"] = "N_" + negative["sample"].astype(str)

positive["time"] = pd.to_datetime(positive["time"], errors="coerce")
negative["time"] = pd.to_datetime(negative["time"], errors="coerce")

print("Positive veri boyutu:", positive.shape)
print("Negative veri boyutu:", negative.shape)


# %% =========================================================
# 5. HAM VERİ KALİTE KONTROLLERİ
# ============================================================

raw_check = pd.concat([negative, positive], ignore_index=True)
raw_check = raw_check.sort_values(["sample_uid", "time_idx"]).reset_index(drop=True)

sample_summary = raw_check.groupby("sample_uid").agg(
    satir_sayisi=("time_idx", "size"),
    benzersiz_gun=("time_idx", "nunique"),
    ilk_indeks=("time_idx", "min"),
    son_indeks=("time_idx", "max"),
    x_sayisi=("x", "nunique"),
    y_sayisi=("y", "nunique"),
)

bad_samples = sample_summary[
    (sample_summary["satir_sayisi"] != 60)
    | (sample_summary["benzersiz_gun"] != 60)
    | (sample_summary["ilk_indeks"] != 0)
    | (sample_summary["son_indeks"] != 59)
    | (sample_summary["x_sayisi"] != 1)
    | (sample_summary["y_sayisi"] != 1)
]

date_gaps = raw_check.groupby("sample_uid")["time"].diff().dt.days
bad_date_gap_count = int(date_gaps.dropna().ne(1).sum())

print("\nToplam sample:", len(sample_summary))
print("Hatalı sample sayısı:", len(bad_samples))
print("Bir günlük olmayan tarih geçişi:", bad_date_gap_count)

if len(bad_samples) > 0 or bad_date_gap_count > 0:
    raise ValueError("Sample yapısı veya tarih sırası hatalı.")

target_rows = raw_check[raw_check["time_idx"] == 59].copy()
target_rows["area_based_label"] = (
    target_rows["burned_area_has"] >= 30
).astype(int)

label_control = pd.crosstab(
    target_rows["fire"],
    target_rows["area_based_label"],
    rownames=["Dosya sınıfı"],
    colnames=["burned_area_has >= 30"],
)
label_mismatch = int(
    (target_rows["fire"] != target_rows["area_based_label"]).sum()
)

print("\nDosya etiketi ile hedef kontrolü:")
print(label_control)
print("Hedefi uyuşmayan sample sayısı:", label_mismatch)

if label_mismatch > 0:
    raise ValueError("Dosya sınıfı ile hedef etiketi uyuşmuyor.")

sample_time_duplicates = int(
    raw_check.duplicated(["sample_uid", "time_idx"], keep=False).sum()
)
target_conflicts = int(
    target_rows.groupby(["x", "y", "time"])["fire"].nunique().gt(1).sum()
)

print("Sample-time tekrar sayısı:", sample_time_duplicates)
print("Hedef konum-tarih çelişkisi:", target_conflicts)

if sample_time_duplicates > 0 or target_conflicts > 0:
    raise ValueError("Tekrar veya hedef çelişkisi bulundu.")

del raw_check


# %% =========================================================
# 6. 59 GÜNLÜK GEÇMİŞTEN SAMPLE TABLOSU OLUŞTURMA
# ============================================================

# Bu tek yardımcı fonksiyon aynı işlemi positive ve negative dosyalarına
# iki kez yazmamak için kullanılır.
def build_sample_table(dataframe, label, prefix):
    dataframe = dataframe.sort_values(["sample", "time_idx"]).copy()

    # Model yalnızca 0-58 arasını bilir.
    history = dataframe[dataframe["time_idx"] <= 58].copy()
    last_day = history[history["time_idx"] == 58].set_index("sample")

    # time_idx=59 yalnızca hedef tarihi ve etiket kontrolü içindir.
    target_day = dataframe[dataframe["time_idx"] == 59].set_index("sample")

    sample_table = last_day[
        STATIC_FEATURES + ["wind_direction"] + DYNAMIC_FEATURES
    ].copy()

    sample_table["last_observation_date"] = last_day["time"]
    sample_table["forecast_date"] = target_day["time"]
    sample_table["fire"] = label
    sample_table["sample_uid"] = prefix + "_" + sample_table.index.astype(str)

    # Hedef günü son bilinen günden tam bir gün sonra olmalıdır.
    day_difference = (
        sample_table["forecast_date"]
        - sample_table["last_observation_date"]
    ).dt.days
    if not day_difference.eq(1).all():
        raise ValueError("Son gözlem ile hedef tarih arasında bir gün yok.")

    mean_columns = DYNAMIC_FEATURES
    max_columns = [
        "t2m", "d2m", "wind_speed", "ssrd", "lst_day", "lst_night"
    ]
    min_columns = ["rh", "smi", "ndvi", "lai"]

    for day_count in [3, 7, 30, 59]:
        first_index = 59 - day_count
        window = history[
            history["time_idx"].between(first_index, 58)
        ]
        grouped = window.groupby("sample")

        sample_table = sample_table.join(
            grouped[mean_columns].mean().add_suffix(f"_{day_count}day_mean")
        )
        sample_table = sample_table.join(
            grouped[max_columns].max().add_suffix(f"_{day_count}day_max")
        )
        sample_table = sample_table.join(
            grouped[min_columns].min().add_suffix(f"_{day_count}day_min")
        )
        sample_table = sample_table.join(
            grouped["tp"].sum().rename(f"tp_{day_count}day_sum")
        )

        dry_days = (
            window.assign(no_rain=window["tp"].fillna(0).lt(0.0001).astype(int))
            .groupby("sample")["no_rain"]
            .sum()
            .rename(f"dry_days_{day_count}day")
        )
        sample_table = sample_table.join(dry_days)

    # Yakın tarihlere daha yüksek ağırlık verilen 59 günlük ortalamalar.
    weights = history["time_idx"].to_numpy(dtype=float) + 1
    sample_groups = history["sample"].to_numpy()

    for column in DYNAMIC_FEATURES:
        values = history[column].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        numerator = pd.Series(
            np.where(valid, values, 0) * weights
        ).groupby(sample_groups).sum()
        denominator = pd.Series(
            np.where(valid, weights, 0)
        ).groupby(sample_groups).sum()
        sample_table[f"{column}_weighted_59day"] = (
            numerator / denominator.replace(0, np.nan)
        )

    return sample_table.reset_index()


negative_sample = build_sample_table(negative, label=0, prefix="N")
positive_sample = build_sample_table(positive, label=1, prefix="P")

sample_frame = pd.concat(
    [negative_sample, positive_sample],
    ignore_index=True,
)

temperature_prefixes = ("t2m", "d2m", "lst_day", "lst_night")
temperature_columns = [
    column
    for column in sample_frame.columns
    if column.startswith(temperature_prefixes)
]

for column in temperature_columns:
    if sample_frame[column].max(skipna=True) > 100:
        sample_frame[column] = sample_frame[column] - 273.15

sample_frame["forecast_year"] = sample_frame["forecast_date"].dt.year
sample_frame["forecast_month"] = sample_frame["forecast_date"].dt.month

print("\nSample seviyesi veri boyutu:", sample_frame.shape)
print("Hedef dağılımı:")
print(sample_frame["fire"].value_counts().sort_index())
print(sample_frame["fire"].value_counts(normalize=True).mul(100).round(2))


# %% =========================================================
# 7. BASİT EDA
# ============================================================

# Feature kararlarını verirken validation ve test yıllarına bakmıyoruz.
train_eda = sample_frame[
    sample_frame["forecast_year"].between(2006, 2020)
].copy()

class_counts = sample_frame["fire"].value_counts().reindex([0, 1])

plt.figure(figsize=(8, 5))
sns.barplot(
    x=["Düşük Tehlike", "Yüksek Tehlike"],
    y=class_counts.values,
    hue=["Düşük Tehlike", "Yüksek Tehlike"],
    palette={"Düşük Tehlike": NO_FIRE_COLOR, "Yüksek Tehlike": FIRE_COLOR},
    legend=False,
)
plt.title("Sample Sınıf Dağılımı")
plt.ylabel("Sample Sayısı")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "01_hedef_dagilimi.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()

numeric_eda_columns = [
    column
    for column in sample_frame.select_dtypes(include="number").columns
    if column not in {"sample", "fire", "forecast_year", "forecast_month"}
]

missing_table = pd.DataFrame(
    {
        "Eksik Sayısı": sample_frame[numeric_eda_columns].isna().sum(),
        "Eksik Oranı (%)": (
            sample_frame[numeric_eda_columns].isna().mean() * 100
        ),
    }
)
missing_table = missing_table[
    missing_table["Eksik Sayısı"] > 0
].sort_values("Eksik Oranı (%)", ascending=False)

print("\nEn yüksek eksik değer oranları:")
print(missing_table.head(20).round(2))
missing_table.to_csv(
    OUTPUT_PATH / "eda_eksik_degerler.csv",
    encoding="utf-8-sig",
)

eda_columns = [
    "t2m", "d2m", "rh", "tp", "wind_speed", "ssrd", "smi",
    "ndvi", "lai", "lst_day", "lst_night", "dem", "slope",
    "roads_distance", "population", "lc_forest",
]
eda_columns = [column for column in eda_columns if column in train_eda]

class_means = train_eda.groupby("fire")[eda_columns].mean().T
class_means.columns = ["Düşük Tehlike", "Yüksek Tehlike"]
class_means["Fark"] = (
    class_means["Yüksek Tehlike"] - class_means["Düşük Tehlike"]
)

print("\nSınıflara göre temel değişken ortalamaları:")
print(class_means.round(4))
class_means.to_csv(
    OUTPUT_PATH / "eda_sinif_ortalamalari.csv",
    encoding="utf-8-sig",
)

correlation_data = train_eda[["fire"] + eda_columns].copy()
correlation_data = correlation_data.rename(
    columns={column: TURKISH_NAMES.get(column, column) for column in eda_columns}
)

plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    linewidths=0.4,
)
plt.title("Temel Değişkenler Korelasyon Matrisi - Train 2006-2020")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "02_korelasyon.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()


# %% =========================================================
# 8. FEATURE ENGINEERING
# Teknik olmayan açıklama: geçmiş koşullardan yeni risk göstergeleri üretme
# ============================================================

model_frame = sample_frame.copy()

# Mevsimi 1-12 gibi düz bir sayı yerine döngüsel olarak anlatıyoruz.
model_frame["forecast_day_of_year"] = model_frame["forecast_date"].dt.dayofyear
model_frame["month_sin"] = np.sin(
    2 * np.pi * model_frame["forecast_month"] / 12
)
model_frame["month_cos"] = np.cos(
    2 * np.pi * model_frame["forecast_month"] / 12
)
model_frame["day_of_year_sin"] = np.sin(
    2 * np.pi * model_frame["forecast_day_of_year"] / 365.25
)
model_frame["day_of_year_cos"] = np.cos(
    2 * np.pi * model_frame["forecast_day_of_year"] / 365.25
)
model_frame["is_fire_season"] = (
    model_frame["forecast_month"].between(5, 10).astype(int)
)

# Sıcaklık, nem, rüzgâr ve kuruluk göstergeleri.
model_frame["temp_dewpoint_difference"] = (
    model_frame["t2m"] - model_frame["d2m"]
)
model_frame["saturation_vapor_pressure"] = 0.6108 * np.exp(
    (17.27 * model_frame["t2m"])
    / (model_frame["t2m"] + 237.3)
)
model_frame["vpd"] = model_frame["saturation_vapor_pressure"] * (
    1 - model_frame["rh"].clip(0, 1)
)
model_frame["dryness_index"] = model_frame["t2m"].clip(lower=0) * (
    1 - model_frame["rh"].clip(0, 1)
)
model_frame["wind_dryness_interaction"] = (
    model_frame["wind_speed"].clip(lower=0)
    * (1 - model_frame["rh"].clip(0, 1))
)
model_frame["hot_dry_windy_index"] = (
    model_frame["t2m"].clip(lower=0)
    * (1 - model_frame["rh"].clip(0, 1))
    * np.log1p(model_frame["wind_speed"].clip(lower=0))
)
model_frame["heat_soil_dryness"] = (
    model_frame["t2m"].clip(lower=0)
    * (1 - model_frame["smi"].clip(0, 1))
)
model_frame["atmosphere_soil_dryness"] = (
    (1 - model_frame["rh"].clip(0, 1))
    * (1 - model_frame["smi"].clip(0, 1))
)
model_frame["radiation_dryness"] = (
    np.log1p(model_frame["ssrd"].clip(lower=0))
    * (1 - model_frame["rh"].clip(0, 1))
)

# Kısa dönem ile uzun dönem arasındaki değişimler.
model_frame["temperature_short_long_change"] = (
    model_frame["t2m_3day_mean"] - model_frame["t2m_30day_mean"]
)
model_frame["humidity_short_long_change"] = (
    model_frame["rh_3day_mean"] - model_frame["rh_30day_mean"]
)
model_frame["soil_moisture_short_long_change"] = (
    model_frame["smi_3day_mean"] - model_frame["smi_30day_mean"]
)

# Kara yüzeyi sıcaklığı farkları.
model_frame["lst_day_night_difference"] = (
    model_frame["lst_day"] - model_frame["lst_night"]
)
model_frame["lst_air_temperature_difference"] = (
    model_frame["lst_day"] - model_frame["t2m"]
)
model_frame["lst_night_air_difference"] = (
    model_frame["lst_night"] - model_frame["t2m"]
)
model_frame["lst_mean"] = (
    model_frame["lst_day"] + model_frame["lst_night"]
) / 2

# Rüzgâr ve yamaç yönlerini sinüs-kosinüs biçimine dönüştürme.
model_frame["wind_direction_sin"] = np.sin(
    np.deg2rad(model_frame["wind_direction"])
)
model_frame["wind_direction_cos"] = np.cos(
    np.deg2rad(model_frame["wind_direction"])
)
model_frame["aspect_sin"] = np.sin(np.deg2rad(model_frame["aspect"]))
model_frame["aspect_cos"] = np.cos(np.deg2rad(model_frame["aspect"]))
model_frame["south_facing_index"] = np.cos(
    np.deg2rad(model_frame["aspect"] - 180)
)
model_frame["slope_south_exposure"] = (
    model_frame["slope"] * model_frame["south_facing_index"]
)
model_frame["elevation_temperature_interaction"] = (
    model_frame["dem"] * model_frame["t2m"]
)

# Rüzgârın sekiz yöndeki etkisi.
wind_directions = {
    "north": 0,
    "northeast": 45,
    "east": 90,
    "southeast": 135,
    "south": 180,
    "southwest": 225,
    "west": 270,
    "northwest": 315,
}

wind_feature_dict = {}
for direction_name, direction_degree in wind_directions.items():
    angle_difference = np.deg2rad(
        model_frame["wind_direction"] - direction_degree
    )
    wind_feature_dict[f"wind_{direction_name}_component"] = (
        model_frame["wind_speed"] * np.cos(angle_difference)
    ).clip(lower=0)

model_frame = pd.concat(
    [model_frame, pd.DataFrame(wind_feature_dict, index=model_frame.index)],
    axis=1,
)

# Bitki örtüsü ve yanıcı madde göstergeleri.
land_cover_columns = [
    "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement",
    "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies", "lc_wetland",
]

model_frame["vegetation_fuel_fraction"] = (
    model_frame["lc_forest"]
    + model_frame["lc_grassland"]
    + model_frame["lc_shrubland"]
    + model_frame["lc_sparse_vegetation"]
).clip(0, 1)

land_cover = model_frame[land_cover_columns].clip(lower=0)
land_cover_sum = land_cover.sum(axis=1).replace(0, np.nan)
land_cover_probability = land_cover.div(land_cover_sum, axis=0)
land_cover_entropy = -(
    land_cover_probability
    * np.log(land_cover_probability.replace(0, np.nan))
).sum(axis=1)
model_frame["land_cover_diversity"] = (
    land_cover_entropy / np.log(len(land_cover_columns))
).clip(0, 1)

soil_dryness = 1 - model_frame["smi"].clip(0, 1)
model_frame["forest_dryness"] = model_frame["lc_forest"] * soil_dryness
model_frame["shrubland_dryness"] = model_frame["lc_shrubland"] * soil_dryness
model_frame["grassland_dryness"] = model_frame["lc_grassland"] * soil_dryness
model_frame["agriculture_dryness"] = (
    model_frame["lc_agriculture"] * soil_dryness
)
model_frame["dry_fuel_proxy"] = (
    model_frame["vegetation_fuel_fraction"] * soil_dryness
)
model_frame["live_fuel_moisture_proxy"] = (
    model_frame["ndvi"].clip(lower=0)
    * model_frame["smi"].clip(0, 1)
)
model_frame["vegetation_vpd_stress"] = (
    model_frame["vegetation_fuel_fraction"] * model_frame["vpd"]
)
model_frame["forest_hot_dry_windy"] = (
    model_frame["lc_forest"] * model_frame["hot_dry_windy_index"]
)

# İnsan etkisini temsil eden göstergeler.
model_frame["log_population"] = np.log1p(
    model_frame["population"].clip(lower=0)
)
model_frame["road_accessibility"] = 1 / (
    model_frame["roads_distance"].clip(lower=0) + 1
)
model_frame["human_pressure"] = (
    model_frame["log_population"] * model_frame["road_accessibility"]
)
model_frame["forest_settlement_interaction"] = (
    model_frame["lc_forest"] * model_frame["lc_settlement"]
)
model_frame["forest_human_pressure"] = (
    model_frame["lc_forest"] * model_frame["human_pressure"]
)

model_frame = model_frame.replace([np.inf, -np.inf], np.nan).copy()


# %% =========================================================
# 9. DATA LEAKAGE KONTROLÜ VE FEATURE LİSTESİ
# ============================================================

found_leakage = sorted(set(LEAKAGE_COLUMNS).intersection(model_frame.columns))
target_day_columns = [
    column for column in model_frame.columns if column.endswith("_d59")
]

print("\nBulunan leakage kolonları:", found_leakage)
print("Bulunan hedef günü (_d59) kolonları:", target_day_columns)

if found_leakage or target_day_columns:
    raise ValueError("Model tablosunda hedef gününe ait bilgi bulundu.")

metadata_columns = {
    "sample", "sample_uid", "last_observation_date", "forecast_date",
    "forecast_year", "forecast_month", "forecast_day_of_year", "fire",
    "wind_direction", "aspect",
}

full_feature_columns = [
    column
    for column in model_frame.select_dtypes(include="number").columns
    if column not in metadata_columns
]

# Baseline yalnızca geçmişten gelen ham/özet feature'ları kullanır.
base_feature_columns = [
    column
    for column in sample_frame.select_dtypes(include="number").columns
    if column not in {
        "sample", "fire", "forecast_year", "forecast_month",
        "wind_direction", "aspect",
    }
]

print("Baseline feature sayısı:", len(base_feature_columns))
print("Tüm feature sayısı:", len(full_feature_columns))
print("Data leakage kontrolü: TEMİZ")

# Özet feature adlarını da anlaşılır Türkçeye çeviriyoruz.
suffix_names = {
    "_3day_mean": " - son 3 gün ortalaması",
    "_7day_mean": " - son 7 gün ortalaması",
    "_30day_mean": " - son 30 gün ortalaması",
    "_59day_mean": " - son 59 gün ortalaması",
    "_3day_max": " - son 3 gün en yüksek",
    "_7day_max": " - son 7 gün en yüksek",
    "_30day_max": " - son 30 gün en yüksek",
    "_59day_max": " - son 59 gün en yüksek",
    "_3day_min": " - son 3 gün en düşük",
    "_7day_min": " - son 7 gün en düşük",
    "_30day_min": " - son 30 gün en düşük",
    "_59day_min": " - son 59 gün en düşük",
    "_3day_sum": " - son 3 gün toplamı",
    "_7day_sum": " - son 7 gün toplamı",
    "_30day_sum": " - son 30 gün toplamı",
    "_59day_sum": " - son 59 gün toplamı",
    "_weighted_59day": " - yakın günlere ağırlıklı 59 gün ortalaması",
}

FEATURE_TURKISH_NAMES = {}

for column in full_feature_columns:
    turkish_name = TURKISH_NAMES.get(column, column)

    for suffix, suffix_name in suffix_names.items():
        if column.endswith(suffix):
            base_name = column[:-len(suffix)]
            turkish_name = TURKISH_NAMES.get(base_name, base_name) + suffix_name
            break

    if column.startswith("dry_days_"):
        day_count = column.replace("dry_days_", "").replace("day", "")
        turkish_name = f"Son {day_count} gündeki yağışsız gün sayısı"

    if column.startswith("wind_") and column.endswith("_component"):
        turkish_name = "Belirli yöndeki rüzgâr gücü"

    FEATURE_TURKISH_NAMES[column] = turkish_name

feature_dictionary = pd.DataFrame(
    {
        "Teknik Ad": full_feature_columns,
        "Türkçe Ad": [
            FEATURE_TURKISH_NAMES[column]
            for column in full_feature_columns
        ],
    }
)
feature_dictionary.to_csv(
    OUTPUT_PATH / "feature_sozlugu.csv",
    index=False,
    encoding="utf-8-sig",
)


# %% =========================================================
# 10. ZAMAN BAZLI TRAIN - VALIDATION - TEST AYRIMI
# ============================================================

train = model_frame[
    model_frame["forecast_year"].between(2006, 2020)
].copy()
validation = model_frame[model_frame["forecast_year"] == 2021].copy()
test = model_frame[model_frame["forecast_year"] == 2022].copy()

for split_name, split_data in [
    ("Train 2006-2020", train),
    ("Validation 2021", validation),
    ("Test 2022", test),
]:
    print(f"\n{split_name}: {split_data.shape}")
    print(split_data["fire"].value_counts().sort_index())
    print(
        split_data["fire"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )
    if split_data.empty or split_data["fire"].nunique() < 2:
        raise ValueError(f"{split_name} içinde iki sınıf bulunmuyor.")

X_train = train[full_feature_columns].copy()
y_train = train["fire"].copy()
X_validation = validation[full_feature_columns].copy()
y_validation = validation["fire"].copy()
X_test = test[full_feature_columns].copy()
y_test = test["fire"].copy()


# %% =========================================================
# 11. DAĞILIM KAYMASI KONTROLÜ
# ============================================================

drift_columns = [
    "t2m", "rh", "tp", "wind_speed", "ssrd", "smi", "ndvi",
    "lai", "population", "roads_distance", "lc_forest", "x", "y",
]
drift_rows = []

for period_name, comparison_data in [
    ("Validation 2021", validation),
    ("Test 2022", test),
]:
    for column in drift_columns:
        train_values = train[column].dropna()
        comparison_values = comparison_data[column].dropna()
        ks_statistic, p_value = ks_2samp(train_values, comparison_values)
        drift_rows.append(
            {
                "Dönem": period_name,
                "Değişken": column,
                "Türkçe Ad": FEATURE_TURKISH_NAMES[column],
                "KS İstatistiği": ks_statistic,
                "p-değeri": p_value,
            }
        )

drift_table = pd.DataFrame(drift_rows).sort_values(
    ["Dönem", "KS İstatistiği"],
    ascending=[True, False],
)
print("\nEn yüksek dönemsel dağılım değişimleri:")
print(drift_table.groupby("Dönem").head(10).round(4).to_string(index=False))
drift_table.to_csv(
    OUTPUT_PATH / "dagilim_kaymasi.csv",
    index=False,
    encoding="utf-8-sig",
)


# %% =========================================================
# 12. MODEL ÖN İŞLEME ADIMLARI
# ============================================================

# Logistic Regression için eksik değerleri doldurup ölçekleme yapıyoruz.
linear_preprocessor_base = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", add_indicator=True),
                    ),
                    ("scaler", RobustScaler()),
                ]
            ),
            base_feature_columns,
        )
    ],
    remainder="drop",
)

linear_preprocessor_full = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", add_indicator=True),
                    ),
                    ("scaler", RobustScaler()),
                ]
            ),
            full_feature_columns,
        )
    ],
    remainder="drop",
)

# Ağaç modellerinde ölçekleme gerekli değildir; yalnızca eksikler doldurulur.
tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            full_feature_columns,
        )
    ],
    remainder="drop",
)


# %% =========================================================
# 13. KARŞILAŞTIRILACAK MODELLER
# ============================================================

tree_count = 120 if FAST_MODE else 500
xgb_count = 160 if FAST_MODE else 600

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

models = {
    "Dummy Baseline": Pipeline(
        steps=[
            ("preprocessor", clone(linear_preprocessor_base)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    ),
    "Logistic Baseline - Ham Feature": Pipeline(
        steps=[
            ("preprocessor", clone(linear_preprocessor_base)),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Class Weight Logistic - Tüm Feature": Pipeline(
        steps=[
            ("preprocessor", clone(linear_preprocessor_full)),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Random Forest - Tüm Feature": Pipeline(
        steps=[
            ("preprocessor", clone(tree_preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=tree_count,
                    max_depth=14,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    ),
}

# XGBoost kuruluysa otomatik eklenir.
xgboost_class = None
try:
    from xgboost import XGBClassifier

    xgboost_class = XGBClassifier
    models["XGBoost - Tüm Feature"] = Pipeline(
        steps=[
            ("preprocessor", clone(tree_preprocessor)),
            (
                "model",
                XGBClassifier(
                    n_estimators=xgb_count,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )
except ImportError:
    print("\nXGBoost kurulu değil; XGBoost modeli atlandı.")

# LightGBM kuruluysa otomatik eklenir.
try:
    from lightgbm import LGBMClassifier

    models["LightGBM - Tüm Feature"] = Pipeline(
        steps=[
            ("preprocessor", clone(tree_preprocessor)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=tree_count,
                    learning_rate=0.03,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    verbosity=-1,
                ),
            ),
        ]
    )
except ImportError:
    print("\nLightGBM kurulu değil; LightGBM modeli atlandı.")

# SMOTE yalnızca Pipeline içindeki fit aşamasında X_train'e uygulanır.
# X_validation ve X_test kesinlikle çoğaltılmaz/değiştirilmez.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline

    models["SMOTE Logistic - Deney"] = ImbalancedPipeline(
        steps=[
            ("preprocessor", clone(linear_preprocessor_full)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # SMOTE sınıfları dengelediği için burada scale_pos_weight kullanılmaz.
    if xgboost_class is not None:
        models["SMOTE XGBoost - Deney"] = ImbalancedPipeline(
            steps=[
                ("preprocessor", clone(tree_preprocessor)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    xgboost_class(
                        n_estimators=xgb_count,
                        max_depth=5,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                    ),
                ),
            ]
        )
except ImportError:
    print("\nimbalanced-learn kurulu değil; SMOTE modelleri atlandı.")

print("\nKarşılaştırılacak modeller:")
for model_name in models:
    print("-", model_name)


# %% =========================================================
# 14. METRİK HESAPLAMAK İÇİN İKİ KÜÇÜK YARDIMCI FONKSİYON
# ============================================================

def calculate_metrics(model_name, y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "F2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
    }


def find_best_f2_threshold(y_true, probability):
    rows = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        prediction = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true, prediction, zero_division=0
                ),
                "recall": recall_score(
                    y_true, prediction, zero_division=0
                ),
                "f1": f1_score(y_true, prediction, zero_division=0),
                "f2": fbeta_score(
                    y_true, prediction, beta=2, zero_division=0
                ),
            }
        )

    threshold_table = pd.DataFrame(rows)
    best_row = threshold_table.sort_values(
        ["f2", "precision"],
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), threshold_table


# %% =========================================================
# 15. MODELLERİ EĞİTME VE VALIDATION ÜZERİNDE KARŞILAŞTIRMA
# ============================================================

trained_models = {}
validation_probabilities = {}
threshold_tables = {}
validation_results = []

for model_name, model in models.items():
    print(f"\n{model_name} eğitiliyor...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    validation_probability = model.predict_proba(X_validation)[:, 1]

    trained_models[model_name] = model
    validation_probabilities[model_name] = validation_probability

    validation_results.append(
        calculate_metrics(
            f"{model_name} - Eşik 0.50",
            y_validation,
            validation_probability,
            threshold=0.50,
        )
    )

    if "Dummy" not in model_name:
        best_threshold, threshold_table = find_best_f2_threshold(
            y_validation,
            validation_probability,
        )
        threshold_tables[model_name] = threshold_table

        validation_results.append(
            calculate_metrics(
                f"{model_name} - Optimize",
                y_validation,
                validation_probability,
                threshold=best_threshold,
            )
        )

validation_results_df = pd.DataFrame(validation_results).sort_values(
    ["F2", "PR-AUC"],
    ascending=False,
)

print("\nTÜM VALIDATION SONUÇLARI")
print(validation_results_df.round(4).to_string(index=False))

validation_results_df.to_csv(
    OUTPUT_PATH / "validation_model_karsilastirmasi.csv",
    index=False,
    encoding="utf-8-sig",
)

# Yalnızca optimize edilmiş satırlar arasından en yüksek F2 seçilir.
optimized_results = validation_results_df[
    validation_results_df["Model"].str.endswith(" - Optimize")
].copy()

best_validation_row = optimized_results.sort_values(
    ["F2", "PR-AUC"],
    ascending=False,
).iloc[0]

best_model_name = best_validation_row["Model"].replace(" - Optimize", "")
best_threshold = float(best_validation_row["Threshold"])
best_model = trained_models[best_model_name]
best_validation_probability = validation_probabilities[best_model_name]

print("\nSEÇİLEN MODEL:", best_model_name)
print("F2 için validation eşiği:", round(best_threshold, 4))

threshold_tables[best_model_name].to_csv(
    OUTPUT_PATH / "validation_f2_esik_tablosu.csv",
    index=False,
    encoding="utf-8-sig",
)


# %% =========================================================
# 16. MALİYET BAZLI EŞİK ANALİZİ - SADECE VALIDATION 2021
# ============================================================

cost_rows = []

for threshold in np.arange(0.01, 1.00, 0.01):
    prediction = (best_validation_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        prediction,
        labels=[0, 1],
    ).ravel()

    total_cost = (
        fn * FALSE_NEGATIVE_COST
        + fp * FALSE_POSITIVE_COST
    )

    cost_rows.append(
        {
            "Eşik": threshold,
            "Doğru Negatif": tn,
            "Yanlış Alarm": fp,
            "Kaçırılan Yangın": fn,
            "Yakalanan Yangın": tp,
            "Toplam Maliyet": total_cost,
            "Maliyet / Gözlem": total_cost / len(y_validation),
            "Precision": precision_score(
                y_validation, prediction, zero_division=0
            ),
            "Recall": recall_score(
                y_validation, prediction, zero_division=0
            ),
            "F2": fbeta_score(
                y_validation, prediction, beta=2, zero_division=0
            ),
        }
    )

cost_table = pd.DataFrame(cost_rows)

# Önce en az %85 recall koşulunu sağlayan eşikler alınır.
eligible_cost_rows = cost_table[
    cost_table["Recall"] >= MINIMUM_RECALL
].copy()

if eligible_cost_rows.empty:
    eligible_cost_rows = cost_table[
        cost_table["Recall"] == cost_table["Recall"].max()
    ].copy()

# Bu eşikler içinde toplam maliyeti en düşük olan seçilir.
best_cost_row = eligible_cost_rows.sort_values(
    ["Toplam Maliyet", "Recall", "Precision"],
    ascending=[True, False, False],
).iloc[0]

cost_threshold = float(best_cost_row["Eşik"])

print("\nVALIDATION MALİYET BAZLI EŞİK SONUCU")
print(best_cost_row.round(4).to_string())

cost_table.to_csv(
    OUTPUT_PATH / "validation_maliyet_esik_analizi.csv",
    index=False,
    encoding="utf-8-sig",
)

plt.figure(figsize=(10, 5))
plt.plot(
    cost_table["Eşik"],
    cost_table["Toplam Maliyet"],
    color=FIRE_COLOR,
    linewidth=2,
)
plt.axvline(
    cost_threshold,
    color=BLUE_COLOR,
    linestyle="--",
    label=f"Seçilen eşik: {cost_threshold:.2f}",
)
plt.title("Validation 2021 - Maliyet Bazlı Eşik Analizi")
plt.xlabel("Karar Eşiği")
plt.ylabel("Göreceli Toplam Maliyet")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "03_validation_maliyet_esigi.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()


# %% =========================================================
# 17. WALK-FORWARD EŞİK KARARLILIĞI
# ============================================================

walk_forward_threshold = None
walk_forward_summary = pd.DataFrame()

if RUN_WALK_FORWARD:
    validation_years = [2020, 2021] if FAST_MODE else [
        2017, 2018, 2019, 2020, 2021
    ]
    walk_forward_rows = []

    for validation_year in validation_years:
        train_fold = model_frame[
            model_frame["forecast_year"] < validation_year
        ].copy()
        validation_fold = model_frame[
            model_frame["forecast_year"] == validation_year
        ].copy()

        if (
            train_fold.empty
            or validation_fold.empty
            or train_fold["fire"].nunique() < 2
            or validation_fold["fire"].nunique() < 2
        ):
            print(f"Walk-forward {validation_year}: yetersiz veri, atlandı.")
            continue

        fold_model = clone(best_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(
                train_fold[full_feature_columns],
                train_fold["fire"],
            )

        fold_probability = fold_model.predict_proba(
            validation_fold[full_feature_columns]
        )[:, 1]

        for threshold in np.arange(0.01, 1.00, 0.01):
            fold_prediction = (fold_probability >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                validation_fold["fire"],
                fold_prediction,
                labels=[0, 1],
            ).ravel()
            fold_cost = (
                fn * FALSE_NEGATIVE_COST
                + fp * FALSE_POSITIVE_COST
            )

            walk_forward_rows.append(
                {
                    "Validation Yılı": validation_year,
                    "Eşik": threshold,
                    "Maliyet / Gözlem": fold_cost / len(validation_fold),
                    "Precision": precision_score(
                        validation_fold["fire"],
                        fold_prediction,
                        zero_division=0,
                    ),
                    "Recall": recall_score(
                        validation_fold["fire"],
                        fold_prediction,
                        zero_division=0,
                    ),
                    "F2": fbeta_score(
                        validation_fold["fire"],
                        fold_prediction,
                        beta=2,
                        zero_division=0,
                    ),
                }
            )

        print(
            f"Walk-forward {validation_year}: "
            f"{len(validation_fold)} gözlem tamamlandı."
        )

    walk_forward_detail = pd.DataFrame(walk_forward_rows)
    walk_forward_detail.to_csv(
        OUTPUT_PATH / "walk_forward_esik_yil_detayi.csv",
        index=False,
        encoding="utf-8-sig",
    )

    walk_forward_summary = (
        walk_forward_detail.groupby("Eşik", as_index=False)
        .agg(
            Ortalama_Maliyet=("Maliyet / Gözlem", "mean"),
            Ortalama_Precision=("Precision", "mean"),
            Ortalama_Recall=("Recall", "mean"),
            Ortalama_F2=("F2", "mean"),
            Yil_Sayisi=("Validation Yılı", "nunique"),
        )
    )

    eligible_walk_rows = walk_forward_summary[
        walk_forward_summary["Ortalama_Recall"] >= MINIMUM_RECALL
    ].copy()

    if eligible_walk_rows.empty:
        eligible_walk_rows = walk_forward_summary[
            walk_forward_summary["Ortalama_Recall"]
            == walk_forward_summary["Ortalama_Recall"].max()
        ].copy()

    best_walk_row = eligible_walk_rows.sort_values(
        ["Ortalama_Maliyet", "Ortalama_Recall"],
        ascending=[True, False],
    ).iloc[0]

    walk_forward_threshold = float(best_walk_row["Eşik"])

    print("\nWALK-FORWARD EŞİK SONUCU")
    print(best_walk_row.round(4).to_string())

    walk_forward_summary.to_csv(
        OUTPUT_PATH / "walk_forward_esik_ozeti.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, cost_axis = plt.subplots(figsize=(10, 5))
    cost_axis.plot(
        walk_forward_summary["Eşik"],
        walk_forward_summary["Ortalama_Maliyet"],
        color=FIRE_COLOR,
        label="Ortalama maliyet",
    )
    cost_axis.set_xlabel("Karar Eşiği")
    cost_axis.set_ylabel("Ortalama Maliyet", color=FIRE_COLOR)
    cost_axis.axvline(
        walk_forward_threshold,
        color=BLUE_COLOR,
        linestyle="--",
        label=f"Seçilen: {walk_forward_threshold:.2f}",
    )

    recall_axis = cost_axis.twinx()
    recall_axis.plot(
        walk_forward_summary["Eşik"],
        walk_forward_summary["Ortalama_Recall"],
        color=NO_FIRE_COLOR,
        label="Ortalama recall",
    )
    recall_axis.axhline(
        MINIMUM_RECALL,
        color=NO_FIRE_COLOR,
        linestyle=":",
    )
    recall_axis.set_ylabel("Ortalama Recall", color=NO_FIRE_COLOR)
    plt.title("Walk-Forward Eşik Kararlılığı")
    fig.tight_layout()
    plt.savefig(OUTPUT_PATH / "04_walk_forward_esigi.png", dpi=170)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# %% =========================================================
# 18. KOORDİNAT BAĞIMLILIĞI KONTROLÜ
# ============================================================

full_validation_pr_auc = average_precision_score(
    y_validation,
    best_validation_probability,
)

no_location_features = [
    column
    for column in full_feature_columns
    if column not in {"x", "y"}
]

if "Logistic" in best_model_name:
    no_location_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                no_location_features,
            )
        ],
        remainder="drop",
    )
else:
    no_location_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                no_location_features,
            )
        ],
        remainder="drop",
    )

no_location_model = clone(best_model)
no_location_model.set_params(preprocessor=no_location_preprocessor)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    no_location_model.fit(X_train, y_train)

no_location_probability = no_location_model.predict_proba(X_validation)[:, 1]
no_location_pr_auc = average_precision_score(
    y_validation,
    no_location_probability,
)

coordinate_comparison = pd.DataFrame(
    {
        "Model": ["x-y dahil", "x-y hariç"],
        "Validation PR-AUC": [
            full_validation_pr_auc,
            no_location_pr_auc,
        ],
    }
)
coordinate_comparison["Tam Modele Göre Kayıp"] = (
    full_validation_pr_auc
    - coordinate_comparison["Validation PR-AUC"]
)

print("\nKOORDİNAT BAĞIMLILIĞI KONTROLÜ")
print(coordinate_comparison.round(4).to_string(index=False))
coordinate_comparison.to_csv(
    OUTPUT_PATH / "koordinat_karsilastirmasi.csv",
    index=False,
    encoding="utf-8-sig",
)


# %% =========================================================
# 19. 2022 TEST SONUCU
# Test burada yalnızca raporlama için kullanılır.
# ============================================================

test_probability = best_model.predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= best_threshold).astype(int)

test_result = calculate_metrics(
    f"{best_model_name} - 2022 Test",
    y_test,
    test_probability,
    best_threshold,
)

print("\n2022 TEST SONUCU")
print(pd.DataFrame([test_result]).round(4).to_string(index=False))
print("\n2022 CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        test_prediction,
        target_names=["Düşük Tehlike", "Yüksek Tehlike"],
        zero_division=0,
    )
)

pd.DataFrame([test_result]).to_csv(
    OUTPUT_PATH / "2022_test_sonucu.csv",
    index=False,
    encoding="utf-8-sig",
)

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, test_prediction),
    display_labels=["Düşük Tehlike", "Yüksek Tehlike"],
).plot(cmap="Reds", values_format="d")
plt.title(f"{best_model_name} - 2022 Test\nEşik: {best_threshold:.2f}")
plt.grid(False)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "05_test_confusion_matrix.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
RocCurveDisplay.from_predictions(
    y_test,
    test_probability,
    name=best_model_name,
    ax=axes[0],
)
axes[0].set_title("2022 Test ROC Eğrisi")
PrecisionRecallDisplay.from_predictions(
    y_test,
    test_probability,
    name=best_model_name,
    ax=axes[1],
)
axes[1].set_title("2022 Test Precision-Recall Eğrisi")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "06_test_roc_pr.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()


# %% =========================================================
# 20. ÖNCEDEN SEÇİLEN EŞİKLERİ TESTTE RAPORLAMA
# ============================================================

threshold_options = {
    "F2 - Validation 2021": best_threshold,
    "Maliyet - Validation 2021": cost_threshold,
    "Walk-forward 2017-2021": walk_forward_threshold,
    "Varsayılan 0.50": 0.50,
}

threshold_comparison_rows = []

for threshold_name, threshold in threshold_options.items():
    if threshold is None:
        continue

    row = calculate_metrics(
        threshold_name,
        y_test,
        test_probability,
        float(threshold),
    )
    threshold_prediction = (
        test_probability >= float(threshold)
    ).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        threshold_prediction,
        labels=[0, 1],
    ).ravel()
    row["Yanlış Alarm"] = int(fp)
    row["Kaçırılan Yangın"] = int(fn)
    row["Göreceli Maliyet"] = (
        fn * FALSE_NEGATIVE_COST
        + fp * FALSE_POSITIVE_COST
    )
    threshold_comparison_rows.append(row)

threshold_comparison = pd.DataFrame(threshold_comparison_rows)

print("\n2022 TEST - ÖNCEDEN SEÇİLMİŞ EŞİKLER")
print("Test sonuçları eşik seçmek için kullanılmamıştır.")
print(threshold_comparison.round(4).to_string(index=False))

threshold_comparison.to_csv(
    OUTPUT_PATH / "2022_esik_karsilastirmasi.csv",
    index=False,
    encoding="utf-8-sig",
)


# %% =========================================================
# 21. FEATURE IMPORTANCE
# ============================================================

importance_repeats = 2 if FAST_MODE else 5

importance_result = permutation_importance(
    estimator=best_model,
    X=X_validation,
    y=y_validation,
    scoring="average_precision",
    n_repeats=importance_repeats,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)

feature_importance_df = pd.DataFrame(
    {
        "Değişken": full_feature_columns,
        "Türkçe Ad": [
            FEATURE_TURKISH_NAMES[column]
            for column in full_feature_columns
        ],
        "Ortalama Önem": importance_result.importances_mean,
        "Önem Standart Sapması": importance_result.importances_std,
    }
).sort_values("Ortalama Önem", ascending=False)

print("\nEN ÖNEMLİ 20 FEATURE")
print(feature_importance_df.head(20).round(4).to_string(index=False))

feature_importance_df.to_csv(
    OUTPUT_PATH / "feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

top_features = feature_importance_df.head(20).sort_values("Ortalama Önem")

plt.figure(figsize=(12, 8))
plt.barh(
    top_features["Türkçe Ad"],
    top_features["Ortalama Önem"],
    xerr=top_features["Önem Standart Sapması"],
    color=ORANGE_COLOR,
    alpha=0.9,
)
plt.title("En Önemli 20 Değişken")
plt.xlabel("Değişken karıştırıldığında PR-AUC azalması")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "07_feature_importance.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()


# %% =========================================================
# 22. FEATURE GRUPLARININ ORTAK ÖNEMİ
# ============================================================

feature_groups = {
    "Konum": [],
    "İnsan Etkisi": [],
    "Topoğrafya": [],
    "Bitki ve Toprak": [],
    "Meteoroloji": [],
    "Zaman": [],
    "Diğer": [],
}

for column in full_feature_columns:
    if column in {"x", "y"}:
        group_name = "Konum"
    elif any(key in column for key in [
        "population", "road", "human", "settlement"
    ]):
        group_name = "İnsan Etkisi"
    elif any(key in column for key in [
        "dem", "slope", "aspect", "curvature", "elevation"
    ]):
        group_name = "Topoğrafya"
    elif any(key in column for key in [
        "ndvi", "lai", "smi", "forest", "grass", "shrub",
        "vegetation", "fuel", "agriculture", "wetland", "water_bodies"
    ]):
        group_name = "Bitki ve Toprak"
    elif any(key in column for key in [
        "t2m", "d2m", "rh", "wind", "ssrd", "sp", "tp",
        "vpd", "dryness", "temperature", "humidity", "lst"
    ]):
        group_name = "Meteoroloji"
    elif any(key in column for key in [
        "month", "day_of_year", "fire_season"
    ]):
        group_name = "Zaman"
    else:
        group_name = "Diğer"

    feature_groups[group_name].append(column)

baseline_pr_auc = average_precision_score(
    y_validation,
    best_validation_probability,
)
rng = np.random.default_rng(RANDOM_STATE)
group_importance_rows = []

for group_name, group_columns in feature_groups.items():
    if not group_columns:
        continue

    decreases = []
    group_repeats = 2 if FAST_MODE else 5

    for repeat in range(group_repeats):
        shuffled_validation = X_validation.copy()
        shuffled_order = rng.permutation(len(shuffled_validation))
        shuffled_validation[group_columns] = (
            shuffled_validation[group_columns]
            .iloc[shuffled_order]
            .to_numpy()
        )
        shuffled_probability = best_model.predict_proba(
            shuffled_validation
        )[:, 1]
        shuffled_pr_auc = average_precision_score(
            y_validation,
            shuffled_probability,
        )
        decreases.append(baseline_pr_auc - shuffled_pr_auc)

    group_importance_rows.append(
        {
            "Feature Grubu": group_name,
            "PR-AUC Azalması": np.mean(decreases),
            "Standart Sapma": np.std(decreases),
            "Feature Sayısı": len(group_columns),
        }
    )

group_importance_df = pd.DataFrame(group_importance_rows).sort_values(
    "PR-AUC Azalması",
    ascending=False,
)

print("\nFEATURE GRUPLARININ ORTAK ÖNEMİ")
print(group_importance_df.round(4).to_string(index=False))

group_importance_df.to_csv(
    OUTPUT_PATH / "feature_grup_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

group_plot = group_importance_df.sort_values("PR-AUC Azalması")
plt.figure(figsize=(10, 6))
plt.barh(
    group_plot["Feature Grubu"],
    group_plot["PR-AUC Azalması"],
    xerr=group_plot["Standart Sapma"],
    color="#4BAA9F",
)
plt.title("Feature Gruplarının Model Üzerindeki Ortak Etkisi")
plt.xlabel("Grup karıştırıldığında PR-AUC azalması")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "08_feature_grup_importance.png", dpi=170)
if SHOW_PLOTS:
    plt.show()
plt.close()


# %% =========================================================
# 23. STREAMLIT İÇİN MODELİ VE GİRİŞ REHBERİNİ KAYDETME
# ============================================================

input_summary = pd.DataFrame(
    {
        "Değişken": full_feature_columns,
        "Türkçe Ad": [
            FEATURE_TURKISH_NAMES[column]
            for column in full_feature_columns
        ],
        "Varsayılan (Train Medyan)": (
            X_train.median().reindex(full_feature_columns).to_numpy()
        ),
        "Alt Sınır (Train %1)": (
            X_train.quantile(0.01).reindex(full_feature_columns).to_numpy()
        ),
        "Üst Sınır (Train %99)": (
            X_train.quantile(0.99).reindex(full_feature_columns).to_numpy()
        ),
    }
)

input_summary.to_csv(
    OUTPUT_PATH / "streamlit_feature_giris_rehberi.csv",
    index=False,
    encoding="utf-8-sig",
)

streamlit_threshold_options = {
    threshold_name: float(threshold)
    for threshold_name, threshold in threshold_options.items()
    if threshold is not None
}

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "threshold": best_threshold,
    "threshold_purpose": "Validation 2021 üzerinde en yüksek F2",
    "streamlit_threshold_options": streamlit_threshold_options,
    "feature_columns": full_feature_columns,
    "base_feature_columns": base_feature_columns,
    "feature_names_turkish": {
        column: FEATURE_TURKISH_NAMES[column]
        for column in full_feature_columns
    },
    "streamlit_input_defaults": dict(
        zip(
            input_summary["Değişken"],
            input_summary["Varsayılan (Train Medyan)"],
        )
    ),
    "streamlit_input_lower_bounds": dict(
        zip(
            input_summary["Değişken"],
            input_summary["Alt Sınır (Train %1)"],
        )
    ),
    "streamlit_input_upper_bounds": dict(
        zip(
            input_summary["Değişken"],
            input_summary["Üst Sınır (Train %99)"],
        )
    ),
    "train_years": "2006-2020",
    "validation_year": 2021,
    "test_year": 2022,
    "last_known_time_indices": "0-58",
    "target_time_index": 59,
    "history_days": 59,
    "target_definition": "time_idx=59 gününde en az 30 hektarlık yangın örneği",
    "risk_output_note": (
        "Kalibrasyon yapılana kadar çıktı olasılık değil, risk skorudur."
    ),
}

MODEL_FILE = OUTPUT_PATH / "mesogeos_erken_uyari_modeli.joblib"
joblib.dump(model_bundle, MODEL_FILE)

test_predictions = test[
    ["sample_uid", "x", "y", "forecast_date", "fire"]
].copy()
test_predictions["risk_score"] = test_probability
test_predictions["prediction_f2_threshold"] = test_prediction
test_predictions["prediction_cost_threshold"] = (
    test_probability >= cost_threshold
).astype(int)

if walk_forward_threshold is not None:
    test_predictions["prediction_walk_forward_threshold"] = (
        test_probability >= walk_forward_threshold
    ).astype(int)

test_predictions.to_csv(
    OUTPUT_PATH / "2022_test_tahminleri.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n============================================================")
print("ÇALIŞMA TAMAMLANDI")
print("Seçilen model:", best_model_name)
print("F2 eşiği:", round(best_threshold, 4))
print("Maliyet eşiği:", round(cost_threshold, 4))
print("Walk-forward eşiği:", walk_forward_threshold)
print("Model dosyası:", MODEL_FILE)
print("Tüm sonuçlar:", OUTPUT_PATH)
print("Streamlit'te çıktı adı 'risk skoru' olarak kullanılmalıdır.")
print("============================================================")
