from datetime import datetime

import polars as pl
import numpy as np
import geohash2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

train_test_df = pl.read_csv("data/train-test.csv", try_parse_dates=True)
# ========= Getting training data information first, to get acquainted to the data.
validation_df = pl.read_csv("data/validation.csv", try_parse_dates=True)

df_grouped_by_seasonality = train_test_df.with_columns([
    pl.col("date").dt.month().alias("month"),])
validation_df = validation_df.with_columns([
    pl.col("date").dt.month().alias("month"),])
df_grouped_by_month_and_area = df_grouped_by_seasonality.with_columns([
    pl.col("pickup_lat").round(1),
    pl.col("pickup_lon").round(1),
    pl.col("delivery_lat").round(1),
    pl.col("delivery_lon").round(1),]
)
print(df_grouped_by_month_and_area.head())

unique_equipment_types = train_test_df["equipment"].unique().to_list()
print(unique_equipment_types)
# ['Dry Van', 'Flatbed', 'Reefer']

# ======== Grouping the data according to seasonality and spatial area
df_full = df_grouped_by_month_and_area.with_columns([
    pl.concat_str([
            pl.col("month"),
            pl.col("pickup_lat"),
            pl.col("pickup_lon"),
            pl.col("delivery_lat"),
            pl.col("delivery_lon"),
            pl.col("equipment"),],
        separator="_",
    ).alias("group_key")
])
# Clean nans
df_full = df_full.drop_nulls()

group_shuffle_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
groups = df_full["group_key"].to_numpy()
train_idx, test_idx = next(group_shuffle_split.split(df_full, groups=groups))
print(train_idx)

# Split data
train_df = df_full[train_idx]
test_df = df_full[test_idx]

train_df = train_df.with_columns(
    pl.col("pickup").cast(pl.Categorical).to_physical().alias("pickup_encoded"),
    pl.col("delivery").cast(pl.Categorical).to_physical().alias("delivery_encoded"),
    pl.col("equipment").cast(pl.Categorical).to_physical().alias("equipment_encoded"),

)
print(train_df.head())
test_df = test_df.with_columns(
    pl.col("pickup").cast(pl.Categorical).to_physical().alias("pickup_encoded"),
    pl.col("delivery").cast(pl.Categorical).to_physical().alias("delivery_encoded"),
    pl.col("equipment").cast(pl.Categorical).to_physical().alias("equipment_encoded"),

)
validation_df = validation_df.with_columns(
    pl.col("pickup").cast(pl.Categorical).to_physical().alias("pickup_encoded"),
    pl.col("delivery").cast(pl.Categorical).to_physical().alias("delivery_encoded"),
    pl.col("equipment").cast(pl.Categorical).to_physical().alias("equipment_encoded"),

)
# ======== Data validation
# 1. are all ids unique?
if len(train_df) == train_df["load_id"].unique().shape[0]:
    print("All ids are unique.")

# 2. checking datetime format
def check_datetime(datetime_str: str):
    datetime_str = bool(datetime.strptime(datetime_str, "%Y-%m-%d"))
    if datetime_str:
        return True
    else:
        return False

validated_df = train_df.with_columns(
    pl.col("date")
    .cast(pl.String)
    .str.to_date("%Y-%m-%d", strict=False)
    .is_not_null()
    .alias("is_valid_date")
)

if len(validated_df["is_valid_date"]) == train_df["load_id"].unique().shape[0]:
    print("All dates are valid")

# ======= Feature Selection
print(train_df.columns)

selected_features = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance', 'weight', 'market_index', 'quote_signal', 'month', 'pickup_encoded', 'delivery_encoded', 'equipment_encoded']
categorical_features = ['pickup_encoded', 'delivery_encoded', 'equipment_encoded']
target_col = "posted_rate"
X_train = train_df[selected_features]
y_train = train_df[target_col].to_numpy()
X_test = test_df[selected_features]
y_test = test_df[target_col].to_numpy()
# Models
hist_gradient_boost_model_5 = HistGradientBoostingRegressor(categorical_features=categorical_features, max_iter=5, learning_rate=0.01, random_state=42)
hist_gradient_boost_model_100 = HistGradientBoostingRegressor(categorical_features=categorical_features, max_iter=100, learning_rate=0.01, random_state=42)
hist_gradient_boost_model_300 = HistGradientBoostingRegressor(categorical_features=categorical_features, max_iter=300, learning_rate=0.01, random_state=42)

random_forest_model = RandomForestRegressor(n_estimators=100, n_jobs=2, random_state=42)
gradient_boost_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=42)

hist_gradient_boost_model_5.fit(X_train, y_train)
hist_gradient_boost_model_100.fit(X_train, y_train)
hist_gradient_boost_model_300.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)
gradient_boost_model.fit(X_train, y_train)

# --- 1. Predictions ---
hist_gradient_boost_preds = hist_gradient_boost_model_5.predict(X_test)
hist_gradient_boost_preds_100 = hist_gradient_boost_model_100.predict(X_test)
hist_gradient_boost_preds_300 = hist_gradient_boost_model_300.predict(X_test)
random_forest_preds = random_forest_model.predict(X_test)
gradient_boost_preds = gradient_boost_model.predict(X_test)

# --- 2. RMSE ---
rmse_hist_gradient_boost = root_mean_squared_error(y_test, hist_gradient_boost_preds)
rmse_hist_gradient_boost_100 = root_mean_squared_error(y_test, hist_gradient_boost_preds_100)
rmse_hist_gradient_boost_300 = root_mean_squared_error(y_test, hist_gradient_boost_preds_300)
rmse_random_forest = root_mean_squared_error(y_test, random_forest_preds)
rmse_gradient_boost = root_mean_squared_error(y_test, gradient_boost_preds)

# --- 3. MAE ---
mae_hist_gradient_boost = mean_absolute_error(y_test, hist_gradient_boost_preds)
mae_hist_gradient_boost_100 = mean_absolute_error(y_test, hist_gradient_boost_preds_100)
mae_hist_gradient_boost_300 = mean_absolute_error(y_test, hist_gradient_boost_preds_300)
mae_random_forest = mean_absolute_error(y_test, random_forest_preds)
mae_gradient_boost = mean_absolute_error(y_test, gradient_boost_preds)

# --- 4. R2 SCORE ---
r2_score_hist_gradient_boost = r2_score(y_test, hist_gradient_boost_preds)
r2_score_hist_gradient_boost_100 = r2_score(y_test, hist_gradient_boost_preds_100)
r2_score_hist_gradient_boost_300 = r2_score(y_test, hist_gradient_boost_preds_300)
r2_score_random_forest = r2_score(y_test, random_forest_preds)
r2_score_gradient_boost = r2_score(y_test, gradient_boost_preds)

print("=== HistGradientBoosting with 5 iterations ===")
print("RMSE:", rmse_hist_gradient_boost)
print("MAE :", mae_hist_gradient_boost)
print("R2  :", r2_score_hist_gradient_boost)
print()

print("=== HistGradientBoosting with 100 iterations ===")
print("RMSE:", rmse_hist_gradient_boost_100)
print("MAE :", mae_hist_gradient_boost_100)
print("R2  :", r2_score_hist_gradient_boost_100)
print()

print("=== HistGradientBoosting with 300 iterations ===")
print("RMSE:", rmse_hist_gradient_boost_300)
print("MAE :", mae_hist_gradient_boost_300)
print("R2  :", r2_score_hist_gradient_boost_300)
print()

print("=== RandomForest ===")
print("RMSE:", rmse_random_forest)
print("MAE :", mae_random_forest)
print("R2  :", r2_score_random_forest)
print()

print("=== GradientBoosting ===")
print("RMSE:", rmse_gradient_boost)
print("MAE :", mae_gradient_boost)
print("R2  :", r2_score_gradient_boost)

X_val = validation_df.select(selected_features).to_pandas()
hist_preds_300 = hist_gradient_boost_model_300.predict(X_val)
validation_with_preds = validation_df.with_columns(
    pl.Series(name="predicted_rate", values=hist_preds_300)
)
val_template = pl.read_csv("data/validation-predictions-template.csv")
val_predictions_df = val_template.select("load_id").join(
    validation_with_preds.select(["load_id", "predicted_rate"]),
    on="load_id",
    how="left"
)

val_predictions_df.write_csv("data/validation-predictions-template.csv")

dec_df = pl.read_csv("data/december-chart-inputs.csv")
dec_df = dec_df.with_columns(
    pl.lit(12).alias("month"), # December month=12
    pl.col("pickup").cast(pl.Categorical).to_physical().alias("pickup_encoded"),
    pl.col("delivery").cast(pl.Categorical).to_physical().alias("delivery_encoded"),
    pl.col("equipment").cast(pl.Categorical).to_physical().alias("equipment_encoded"),
)

selected_features = [
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'distance', 'weight', 'market_index', 'quote_signal',
    'month', 'pickup_encoded', 'delivery_encoded', 'equipment_encoded'
]

for col in selected_features:
    if col not in dec_df.columns:

        dec_df = dec_df.with_columns(pl.lit(0.0).alias(col))


X_dec = dec_df.select(selected_features)
dec_df = dec_df.with_columns(
    pl.Series(name="predicted_rate", values=hist_gradient_boost_model_300.predict(X_dec))
)


final_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
dec_df.select(final_cols).write_csv("data/december_chart_inputs.csv")
