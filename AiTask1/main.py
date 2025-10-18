import pandas as pd

df = pd.read_csv("Preview__houses_day1__first_20_rows_.csv")

# 1. Sürətli baxış
print(df.head(5))
print(df.tail(5))
print(df.sample(3, random_state=42))

# 2. Struktur yoxlaması
print(df.isnull().sum())
print(df.dtypes)

# 3. Statistik icmal
print(df[["Area_m2", "Price_AZN"]].describe())

# 4. Tip düzəlişi
df["Price_AZN"] = pd.to_numeric(df["Price_AZN"], errors="coerce")
print(df.dtypes)

# 5. Qiymət outlier-ləri (təxmini)
df_sorted = df.sort_values("Price_AZN", ascending=False)
print(df_sorted[["Price_AZN", "Area_m2", "Rooms", "District"]].head(10))

# 6. Kateqorik balans
print(df["District"].value_counts(dropna=False))

# 7. Rooms distribusiyası
df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
print(df["Rooms"].value_counts(dropna=False).sort_index())

# 8. Mean vs Median (Price)
print(df["Price_AZN"].mean(), df["Price_AZN"].median())

# 9. Mode və yayılma ölçüləri
print(df["Rooms"].mode())
print(df["Price_AZN"].var(), df["Price_AZN"].std())

# 10. Filter + seçim
filtered = df[(df["Rooms"] >= 3) & (df["Area_m2"] >= 100)]
print(filtered["Price_AZN"].mean())

# 11. District üzrə mərkəz ölçüləri
print(df.groupby("District")["Price_AZN"].agg(["mean", "median", "count"]))

# 12. Outlier aşkarlanması (IQR)
Q1 = df["Price_AZN"].quantile(0.25)
Q3 = df["Price_AZN"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["IsOutlier_IQR"] = (df["Price_AZN"] < lower) | (df["Price_AZN"] > upper)
print(df[df["IsOutlier_IQR"]][["Price_AZN", "Area_m2", "Rooms", "District"]])

# 13. Outlier aşkarlanması (Z-score yalnız pandas ilə)
price_mean = df["Price_AZN"].mean()
price_std = df["Price_AZN"].std()
df["zscore_Price"] = (df["Price_AZN"] - price_mean) / price_std
df["IsOutlier_zscore"] = df["zscore_Price"].abs() > 3
print(df[df["IsOutlier_zscore"]][["Price_AZN", "Area_m2", "Rooms", "District"]])

# 14. Top 10 ən bahalı və ən ucuz evlər
df["IsOutlier"] = df["IsOutlier_IQR"] | df["IsOutlier_zscore"]
print(df.sort_values("Price_AZN", ascending=False).head(10)[["Price_AZN", "Area_m2", "Rooms", "District", "IsOutlier"]])
print(df.sort_values("Price_AZN", ascending=True).head(10)[["Price_AZN", "Area_m2", "Rooms", "District", "IsOutlier"]])

# 15. Room-Effect ideyası
print(df.groupby("Rooms")["Price_AZN"].median().sort_index())

# 16. Price per m² (ppm)
df["ppm"] = df["Price_AZN"] / df["Area_m2"]
print(df.sort_values("ppm", ascending=False).head(10))

# 17. Kateqorik təmizləmə (map)
region_map = {
    "Sabayil": "Prime",
    "Sabail": "Prime",
    "Yasamal": "Central",
    "Nizami": "Central",
    "Nasimi": "Central",
    "Nerimanov": "Central",
    "Khatai": "Outer",
    "Khatai Rayon": "Outer",
    "Binagadi": "Outer"
}
df["region"] = df["District"].map(region_map).fillna("Other")
print(df.groupby("region")["Price_AZN"].median())

# 18. Tip problemləri və boşluqların təsiri
print(df[df["Price_AZN"].isna()][["District", "Rooms", "Area_m2"]])

# 19. Simulyasiya “təmiz” qiymət medianı
clean_df = df[~df["IsOutlier_IQR"]]
print(clean_df["Price_AZN"].median(), df["Price_AZN"].median())

# 20. Mini-profil hesabatı
profile = {
    "shape": df.shape,
    "nulls_per_column": df.isnull().sum().to_dict(),
    "numeric_describe": df.describe().to_dict(),
    "district_counts": df["District"].value_counts().to_dict(),
    "mean_price": df["Price_AZN"].mean(),
    "median_price": df["Price_AZN"].median(),
    "top_ppm_5": df.sort_values("ppm", ascending=False).head(5)[["ppm", "Price_AZN", "Area_m2", "District", "Rooms"]].to_dict(orient="records")
}
print(profile)
