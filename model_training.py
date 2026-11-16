
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. DOSYA YOLLARI
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs_feature_engineering"
OUTPUT_DIR = BASE_DIR / "outputs_modeling"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "session_start_count",
    "started_session_energy_kwh"
]

print("=" * 65)
print("MODEL EĞİTİMİ VE KARŞILAŞTIRMASI")
print("=" * 65)

# --------------------------------------------------
# 2. VERİLERİ OKU
# --------------------------------------------------

train = pd.read_csv(
    INPUT_DIR / "train_verisi.csv",
    parse_dates=["origin_datetime", "target_datetime"]
)

validation = pd.read_csv(
    INPUT_DIR / "validation_verisi.csv",
    parse_dates=["origin_datetime", "target_datetime"]
)

test = pd.read_csv(
    INPUT_DIR / "test_verisi.csv",
    parse_dates=["origin_datetime", "target_datetime"]
)

print(f"Eğitim satırı: {len(train)}")
print(f"Doğrulama satırı: {len(validation)}")
print(f"Test satırı: {len(test)}")

# Model girdisi olmayan sütunlar
excluded_columns = [
    "origin_datetime",
    "target_datetime",
    "session_start_count",
    "started_session_energy_kwh"
]

feature_columns = [
    col for col in train.columns
    if col not in excluded_columns
]

# Özelliklerin tamamının sayısal olduğunu kontrol et
for col in feature_columns:
    train[col] = pd.to_numeric(train[col], errors="coerce")
    validation[col] = pd.to_numeric(validation[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

if (
    train[feature_columns].isna().any().any()
    or validation[feature_columns].isna().any().any()
    or test[feature_columns].isna().any().any()
):
    raise ValueError("Model özelliklerinde eksik veya sayısal olmayan değer var.")

X_train = train[feature_columns]
X_val = validation[feature_columns]
X_test = test[feature_columns]

# --------------------------------------------------
# 3. DEĞERLENDİRME METRİKLERİ
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Sıfır değerlerde MAPE sorununu önlemek için WAPE
    total_actual = np.sum(np.abs(actual))

    if total_actual > 0:
        wape = (
            np.sum(np.abs(actual - predicted))
            / total_actual
        ) * 100
    else:
        wape = np.nan

    # Simetrik ortalama mutlak yüzde hata
    denominator = np.abs(actual) + np.abs(predicted)

    valid = denominator != 0

    if valid.any():
        smape = np.mean(
            2 * np.abs(actual[valid] - predicted[valid])
            / denominator[valid]
        ) * 100
    else:
        smape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "sMAPE_percent": smape
    }


# --------------------------------------------------
# 4. BASELINE MODELLER
# --------------------------------------------------

baseline_predictions = {}

for target in TARGETS:

    # Son bilinen değer
    baseline_predictions[
        (target, "Baseline_Lag0")
    ] = validation[f"{target}_lag_0"].to_numpy()

    # Geçen haftanın aynı saati
    baseline_predictions[
        (target, "Baseline_Lag168")
    ] = validation[f"{target}_lag_168"].to_numpy()


# --------------------------------------------------
# 5. MAKİNE ÖĞRENMESİ MODELLERİ
# --------------------------------------------------

def create_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        )
    }

results = []
best_models = {}

for target in TARGETS:

    print("\n" + "=" * 65)
    print(f"HEDEF: {target}")
    print("=" * 65)

    y_train = train[target]
    y_val = validation[target]

    # Baseline sonuçları
    for model_name in [
        "Baseline_Lag0",
        "Baseline_Lag168"
    ]:

        predictions = baseline_predictions[
            (target, model_name)
        ]

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        results.append({
            "target": target,
            "model": model_name,
            "dataset": "validation",
            **metrics
        })

        print(
            f"{model_name}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"WAPE={metrics['WAPE_percent']:.2f}%"
        )

    # Makine öğrenmesi modelleri
    model_scores = []

    for model_name, model in create_models().items():

        print(f"\nEğitiliyor: {model_name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        predictions = np.maximum(predictions, 0)

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        results.append({
            "target": target,
            "model": model_name,
            "dataset": "validation",
            **metrics
        })

        model_scores.append(
            (metrics["MAE"], model_name, model)
        )

        print(
            f"{model_name}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"WAPE={metrics['WAPE_percent']:.2f}%"
        )

    # En düşük doğrulama MAE değerine sahip modeli seç
    best_score, best_name, best_model = min(
        model_scores,
        key=lambda item: item[0]
    )

    best_models[target] = (best_name, best_model)

    print(
        f"\nSeçilen model: {best_name} "
        f"(Doğrulama MAE: {best_score:.4f})"
    )

    # Modeli kaydet
    model_path = MODEL_DIR / f"{target}_{best_name}.joblib"

    joblib.dump(best_model, model_path)


# --------------------------------------------------
# 6. TEST VERİSİNDE SON DEĞERLENDİRME
# --------------------------------------------------

test_predictions = []

for target in TARGETS:

    best_name, best_model = best_models[target]

    predictions = best_model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    metrics = calculate_metrics(
        test[target],
        predictions
    )

    results.append({
        "target": target,
        "model": best_name,
        "dataset": "test",
        **metrics
    })

    print("\n" + "=" * 65)
    print(f"TEST SONUCU: {target}")
    print(f"Model: {best_name}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"WAPE: {metrics['WAPE_percent']:.2f}%")
    print(f"sMAPE: {metrics['sMAPE_percent']:.2f}%")

    prediction_df = pd.DataFrame({
        "origin_datetime": test["origin_datetime"],
        "target_datetime": test["target_datetime"],
        "forecast_horizon": test["forecast_horizon"],
        "actual": test[target].to_numpy(),
        "predicted": predictions,
        "target": target,
        "model": best_name
    })

    test_predictions.append(prediction_df)


# --------------------------------------------------
# 7. SONUÇLARI KAYDET
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "model_karsilastirma_sonuclari.csv",
    index=False,
    encoding="utf-8-sig"
)

all_predictions = pd.concat(
    test_predictions,
    ignore_index=True
)

all_predictions.to_csv(
    OUTPUT_DIR / "test_tahminleri.csv",
    index=False,
    encoding="utf-8-sig"
)

# Tahmin ufkuna göre test hataları
horizon_results = []

for target in TARGETS:

    subset = all_predictions[
        all_predictions["target"] == target
    ]

    for horizon, group in subset.groupby(
        "forecast_horizon"
    ):

        metrics = calculate_metrics(
            group["actual"],
            group["predicted"]
        )

        horizon_results.append({
            "target": target,
            "forecast_horizon": horizon,
            **metrics
        })

horizon_df = pd.DataFrame(horizon_results)

horizon_df.to_csv(
    OUTPUT_DIR / "ufka_gore_test_hatalari.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 8. SON ÖZET
# --------------------------------------------------

print("\n" + "=" * 65)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 65)

print(
    results_df.round(4).to_string(index=False)
)

print("\nDosyalar kaydedildi:")
print(OUTPUT_DIR / "model_karsilastirma_sonuclari.csv")
print(OUTPUT_DIR / "test_tahminleri.csv")
print(OUTPUT_DIR / "ufka_gore_test_hatalari.csv")
print(MODEL_DIR)

print("\nModel eğitimi tamamlandı.")
