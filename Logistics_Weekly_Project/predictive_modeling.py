import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("simulated_logistics_shipments_clean.csv")
features = ["distance_km", "weight_kg", "volume_m3", "scheduled_days",
            "region", "transport_mode", "traffic_level", "weather"]
X, y = df[features], df["actual_delivery_days"]
numeric = ["distance_km", "weight_kg", "volume_m3", "scheduled_days"]
categorical = ["region", "transport_mode", "traffic_level", "weather"]
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
models = {
    "Linear Regression": Pipeline([("prep", prep), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("prep", prep), ("model", RandomForestRegressor(
        n_estimators=160, min_samples_leaf=3, random_state=42))])
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MAE:", round(mean_absolute_error(y_test, pred), 3),
          "RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3),
          "R2:", round(r2_score(y_test, pred), 3))
