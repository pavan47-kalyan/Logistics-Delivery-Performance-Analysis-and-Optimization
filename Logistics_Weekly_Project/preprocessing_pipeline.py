import pandas as pd
df = pd.read_csv("simulated_logistics_shipments_raw.csv")
print("Shape before cleaning:", df.shape)
print("Missing values:\n", df.isna().sum())
df = df.drop_duplicates(subset=["shipment_id"]).copy()
numeric_cols = ["weight_kg", "shipping_cost", "distance_km", "volume_m3",
                "actual_delivery_days", "scheduled_days"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
df["shipping_cost"] = df["shipping_cost"].fillna(df["shipping_cost"].median())
df["traffic_level"] = df["traffic_level"].fillna(df["traffic_level"].mode()[0])
upper = df["shipping_cost"].quantile(0.99)
df["shipping_cost"] = df["shipping_cost"].clip(upper=upper)
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")
df["delay_days"] = (df["actual_delivery_days"] - df["scheduled_days"]).clip(lower=0)
df["on_time"] = (df["actual_delivery_days"] <= df["scheduled_days"]).astype(int)
df.to_csv("simulated_logistics_shipments_clean.csv", index=False)
