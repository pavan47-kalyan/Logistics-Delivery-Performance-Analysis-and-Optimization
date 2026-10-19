import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("simulated_logistics_shipments_clean.csv")
df["ship_date"] = pd.to_datetime(df["ship_date"])
print(df[["actual_delivery_days", "shipping_cost", "distance_km"]].describe())
print("On-time rate (%):", round(df["on_time"].mean() * 100, 2))
sns.histplot(data=df, x="actual_delivery_days", bins=24, kde=True)
plt.title("Distribution of Actual Delivery Time"); plt.tight_layout(); plt.show()
mode_rate = df.groupby("transport_mode")["on_time"].mean().mul(100)
mode_rate.plot(kind="bar", title="On-Time Delivery Rate by Transport Mode")
plt.ylabel("On-time shipments (%)"); plt.tight_layout(); plt.show()
sns.heatmap(df[["distance_km", "weight_kg", "scheduled_days",
                "actual_delivery_days", "shipping_cost", "delay_days"]].corr(),
            annot=True, cmap="coolwarm", center=0)
plt.tight_layout(); plt.show()
