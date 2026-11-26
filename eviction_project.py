# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "geopandas",
#   "statsmodels",
# ]
# ///
# MAGIC %md
# MAGIC # Chicago evictions Model
# MAGIC **Part 1: Loading and Cleaning the Data.**
# MAGIC
# MAGIC | Dataset | Where it is | What we keep |
# MAGIC |---|---|---|
# MAGIC | Eviction Lab (2000–2018) + Chicago Health Atlas | `health_atlas` volume | eviction filings, filing rate, neighbourhood indicators |
# MAGIC | LCBH | `lcbh_raw` volume | eviction filings and rental units (second source, used as a check) |
# MAGIC | Cook County foreclosures 2012 | `lcbh_raw` volume | number of foreclosure filings per tract |
# MAGIC
# MAGIC **Part 2: Models.**
# MAGIC 1. Linear regression
# MAGIC 2. Model comparison: linear regression vs ridge vs lasso vs decision tree vs random forest
# MAGIC 3. Foreclosures: did 2012 foreclosures predict later evictions?
# MAGIC 4. k-means: group tracts into neighbourhood types
# MAGIC
# MAGIC Each model is followed by checks, collected in the checks summary at the end.

# COMMAND ----------

# MAGIC %pip install geopandas statsmodels

# COMMAND ----------

raw = "/Volumes/cs483eviction/raw"

combined_csv = raw + "/health_atlas/cook_county_evictions_combined_2000_2018.csv"
lcbh_dir = raw + "/lcbh_raw"
foreclosure_file = raw + "/lcbh_raw/cook_county_foreclosures_2012.csv"
tract_map_file = raw + "/lcbh_raw/tl_2010_17031_tract10.zip"

# Eviction Lab columns
el_filings_col = "filings"
el_rate_col = "filing_rate"

# Health Atlas columns not used as features
exclude_features = ["cha_evc_2018", "cha_evf_2018"]

panel_table = "cs483eviction.default.chicago_tract_year_panel"

# COMMAND ----------

import re
from pyspark.sql import functions as F


def tract_id(col_name):
    """Census tract ID as an 11-character string, e.g. '17031010100'."""
    return F.lpad(F.col(col_name).cast("double").cast("long").cast("string"), 11, "0")


def to_number(col_name):
    """Convert a column to a number; anything that isn't a number becomes null."""
    return F.expr("try_cast(`" + col_name + "` AS DOUBLE)")


def safe_name(name):
    """Column names with only lowercase letters, numbers and underscores."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def clean_column_names(spark_df):
    for col_name in spark_df.columns:
        spark_df = spark_df.withColumnRenamed(col_name, safe_name(col_name))
    return spark_df


checks = []


def add_check(name, result, note):
    checks.append({"check": name, "result": str(result), "how to read it": note})


# COMMAND ----------

# MAGIC %md ## 1. Import the Eviction Lab and Chicago Health Atlas data

# COMMAND ----------

combined = spark.read.csv(combined_csv, header=True, inferSchema=True)
combined = clean_column_names(combined)
print("Columns:", combined.columns)

cha_cols = []
for col_name in combined.columns:
    if col_name.startswith("cha_") and col_name != "cha_tract_name":
        cha_cols.append(col_name)

columns_to_keep = [
    tract_id("geoid").alias("tract_id"),
    F.col("year").cast("int").alias("year"),
    to_number(el_filings_col).alias("el_filings"),
    to_number(el_rate_col).alias("el_rate"),
    F.col("cha_tract_name").isNotNull().cast("int").alias("is_chicago"),
]
for col_name in cha_cols:
    columns_to_keep.append(to_number(col_name).alias(col_name))
el_cha = combined.select(columns_to_keep)

el_cha = el_cha.withColumn("el_renters",
                           F.when(F.col("el_rate") > 0, F.col("el_filings") / F.col("el_rate") * 100))

print("Eviction Lab + Health Atlas rows:", el_cha.count(), "| Health Atlas indicators:", len(cha_cols))

# COMMAND ----------

# MAGIC %md ## 2. Import the LCBH evictions and census data

# COMMAND ----------

lcbh_ev = spark.read.csv(lcbh_dir + "/eviction_data_tract.csv", header=True, inferSchema=True)
lcbh_cen = spark.read.csv(lcbh_dir + "/census_data_tract.csv", header=True, inferSchema=True)

lcbh_ev = (lcbh_ev
           .withColumn("tract_id", tract_id("tract"))
           .withColumnRenamed("filing_year", "year")
           .dropDuplicates(["tract_id", "year"])
           .withColumn("acs_key", F.when(F.col("year") <= 2012, "2006-2010").otherwise("2014-2018")))

lcbh_cen = (lcbh_cen
            .withColumn("tract_id", tract_id("tract"))
            .withColumn("acs_key", F.regexp_extract(F.col("census_year").cast("string"), r"^(\d{4}-\d{4})", 1))
            .dropDuplicates(["tract_id", "acs_key"]))

lcbh = (lcbh_ev.join(lcbh_cen.select("tract_id", "acs_key", "housing_units_rental"),
                     ["tract_id", "acs_key"], "left")
        .select("tract_id", F.col("year").cast("int").alias("year"),
                F.col("eviction_filings_total").alias("lcbh_filings"),
                F.col("housing_units_rental").alias("lcbh_renters")))

print("LCBH rows:", lcbh.count())
display(lcbh.limit(5))

# COMMAND ----------

# MAGIC %md ## 3. Foreclosures 2012 Data

# COMMAND ----------

import geopandas as gpd

tracts = gpd.read_file(tract_map_file).to_crs("EPSG:4326")
tracts["tract_id"] = tracts["GEOID10"].astype(str)
tracts = tracts[["tract_id", "geometry"]]

fc = spark.read.csv(foreclosure_file, header=True, inferSchema=True, multiLine=True)
fc = clean_column_names(fc)
fc = fc.dropDuplicates(["doc_number"])
fc = fc.withColumn("year", F.lit(2012))

# Coordinates from location_1, e.g. "(41.88, -87.63)"
fc = (fc
      .withColumn("lat_text", F.regexp_extract(F.col("location_1").cast("string"), r"\(([^,]+),", 1))
      .withColumn("lon_text", F.regexp_extract(F.col("location_1").cast("string"), r",\s*(-?[0-9.]+)\)", 1))
      .withColumn("latitude", to_number("lat_text"))
      .withColumn("longitude", to_number("lon_text")))

fc_pd = fc.filter(F.col("latitude").isNotNull()).select("year", "latitude", "longitude").toPandas()
points = gpd.GeoDataFrame(fc_pd, geometry=gpd.points_from_xy(fc_pd["longitude"], fc_pd["latitude"]),
                          crs="EPSG:4326")
fc_in_tracts = gpd.sjoin(points, tracts, how="inner", predicate="within")

total_foreclosures = fc.count()
matched = len(fc_in_tracts)
print(matched, "of", total_foreclosures, "foreclosures matched to a tract")
add_check("Foreclosures matched to a tract", str(round(matched / total_foreclosures * 100, 1)) + "%",
          "Most should match. A low share means the coordinates weren't read correctly.")

counts_pd = fc_in_tracts.groupby(["tract_id", "year"]).size().reset_index(name="foreclosures")
foreclosure_counts = spark.createDataFrame(counts_pd)

# COMMAND ----------

# MAGIC %md ## 4. Combine everything into one table

# COMMAND ----------

panel = el_cha.join(lcbh, ["tract_id", "year"], "full")
panel = panel.join(foreclosure_counts, ["tract_id", "year"], "left")
panel = panel.withColumn("foreclosures",
                         F.when(F.col("year") == 2012, F.coalesce(F.col("foreclosures"), F.lit(0))))
panel = panel.fillna(0, subset=["is_chicago"])

panel.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(panel_table)
panel = spark.table(panel_table)
print("panel rows:", panel.count(), "| columns:", len(panel.columns))

display(panel.groupBy("year")
             .agg(F.count("*").alias("tracts"),
                  F.sum("is_chicago").alias("chicago_tracts"),
                  F.sum("el_filings").alias("eviction_lab_filings"),
                  F.sum("lcbh_filings").alias("lcbh_filings"),
                  F.sum("foreclosures").alias("foreclosures"))
             .orderBy("year"))

# COMMAND ----------

# MAGIC %md ## 5. Data checks

# COMMAND ----------

duplicates = panel.groupBy("tract_id", "year").count().filter("count > 1").count()
add_check("Duplicate tract-years", duplicates, "Should be 0: each tract appears once per year.")

bad_ids = panel.filter(F.length("tract_id") != 11).count()
add_check("Tract IDs not 11 characters", bad_ids, "Should be 0: every Census tract ID has 11 digits.")

bad_rates = panel.filter((F.col("el_rate") < 0) | (F.col("el_rate") > 100)).count()
add_check("Eviction Lab rates below 0 or above 100", bad_rates,
          "Should be 0 or close to it: a rate is filings per 100 renter households.")

chicago_tracts = panel.filter("is_chicago = 1").select("tract_id").distinct().count()
add_check("Chicago tracts with Health Atlas data", chicago_tracts,
          "Chicago has about 800 census tracts (2010 boundaries).")

missing_rows = []
for col_name in ["el_filings", "el_rate", "el_renters", "lcbh_filings", "lcbh_renters", "foreclosures"]:
    n_missing = panel.filter(F.col(col_name).isNull()).count()
    missing_rows.append((col_name, n_missing))
display(spark.createDataFrame(missing_rows, "column_name string, rows_missing long"))

for row in checks:
    print(row["check"], "->", row["result"])

# COMMAND ----------

# MAGIC %md # Part 2: Models

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_features = 8
features = []
n_test_years = 3
k_clusters = 4
blue = "#257de0"
grey = "#9c9c9c"

df = spark.table(panel_table).toPandas()

for col_name in df.columns:
    if col_name.startswith("cha_"):
        tract_values = df.groupby("tract_id")[col_name].max()
        df[col_name] = df[col_name].fillna(df["tract_id"].map(tract_values))
df["is_chicago"] = df["tract_id"].map(df.groupby("tract_id")["is_chicago"].max())

tract_median_renters = df.groupby("tract_id")["el_renters"].median()
df["el_renters"] = df["el_renters"].fillna(df["tract_id"].map(tract_median_renters))
df["lcbh_rate"] = df["lcbh_filings"] / df["lcbh_renters"] * 100

d = df[(df["is_chicago"] == 1) & (df["el_renters"] >= 50) & df["el_rate"].notna()].copy()
d["log_rate"] = np.log1p(d["el_rate"])
print("Model data:", len(d), "tract-years,", d["tract_id"].nunique(), "tracts,",
      d["year"].min(), "to", d["year"].max())

# COMMAND ----------

# MAGIC %md ## Explore the outcome

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(d["el_rate"], bins=50, color=blue)
axes[0].set_title("Filing rate (per 100 renter households)", loc="left")
axes[1].hist(d["log_rate"], bins=50, color=blue)
axes[1].set_title("log(1 + rate): closer to symmetric", loc="left")
yearly = d.groupby("year")[["el_filings", "el_renters"]].sum()
axes[2].plot(yearly.index, yearly["el_filings"] / yearly["el_renters"] * 100, color=blue, linewidth=2, marker="o")
axes[2].set_title("Chicago filing rate by year", loc="left")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## Choose features

# COMMAND ----------

cha_cols = []
for col_name in d.columns:
    if col_name.startswith("cha_") and col_name not in exclude_features:
        cha_cols.append(col_name)

tract_table = d.groupby("tract_id")[cha_cols + ["log_rate"]].mean()

usable = []
for col_name in cha_cols:
    if tract_table[col_name].isna().mean() < 0.2:
        usable.append(col_name)

corr = tract_table[usable].corrwith(tract_table["log_rate"]).dropna()
corr = corr[corr.abs() < 0.9]
corr = corr.loc[corr.abs().sort_values(ascending=False).index]
display(corr.round(3).rename("corr_with_log_rate").reset_index())

if len(features) == 0:
    for col_name in corr.index:
        too_similar = False
        for chosen in features:
            if abs(tract_table[col_name].corr(tract_table[chosen])) > 0.8:
                too_similar = True
        if not too_similar:
            features.append(col_name)
        if len(features) == n_features:
            break
print("Features:", features)

feature_stats = {}
for col_name in features:
    median = d[col_name].median()
    filled = d[col_name].fillna(median)
    feature_stats[col_name] = [median, filled.mean(), filled.std()]


def prepare_features(frame):
    for col_name in features:
        median, mean, sd = feature_stats[col_name]
        frame[col_name] = (frame[col_name].fillna(median) - mean) / sd
    return frame


d = prepare_features(d)

# COMMAND ----------

# MAGIC %md ## Model 1: Linear regression (OLS)

# COMMAND ----------

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

formula = "log_rate ~ " + " + ".join(features) + " + C(year)"
tract_groups = pd.factorize(d["tract_id"])[0]
ols = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": tract_groups})

coef = pd.DataFrame({"coef": ols.params[features], "p_value": ols.pvalues[features]})
coef["pct_change_per_SD"] = (np.exp(coef["coef"]) - 1) * 100
print("R-squared:", round(ols.rsquared, 3), "| observations:", int(ols.nobs))
display(coef.round(3).reset_index().rename(columns={"index": "feature"}))

# COMMAND ----------

# MAGIC %md ### Model 1 checks

# COMMAND ----------

# Residual plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(ols.fittedvalues, ols.resid, s=4, alpha=0.3, color=blue)
axes[0].axhline(0, color=grey, linewidth=1)
axes[0].set(xlabel="Fitted value", ylabel="Residual")
axes[0].set_title("Residuals vs fitted", loc="left")
axes[1].hist(ols.resid, bins=50, color=blue)
axes[1].set_title("Residual distribution", loc="left")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
add_check("Model 1: mean residual (size)", round(abs(ols.resid.mean()), 4), "Should be about 0.")

# VIF
x_vif = d[features].to_numpy()
vif_rows = []
for i in range(len(features)):
    vif_rows.append({"feature": features[i], "VIF": variance_inflation_factor(x_vif, i)})
vif = pd.DataFrame(vif_rows)
display(vif.round(2))
add_check("Model 1: largest VIF", round(vif["VIF"].max(), 2), "Below 5 means multicollinearity isn't a big problem.")

# Year dummies only vs full model
ols_years_only = smf.ols("log_rate ~ C(year)", data=d).fit()
print("R-squared, year dummies only:", round(ols_years_only.rsquared, 3))
print("R-squared, with neighbourhood features:", round(ols.rsquared, 3))
add_check("Model 1: R-squared added by neighbourhood features",
          round(ols.rsquared - ols_years_only.rsquared, 3),
          "How much of the variation the neighbourhood features explain beyond year-to-year changes.")

# Clustered vs regular standard errors
ols_plain = smf.ols(formula, data=d).fit()
se_table = pd.DataFrame({"se_regular": ols_plain.bse[features], "se_clustered": ols.bse[features]})
se_table["ratio"] = se_table["se_clustered"] / se_table["se_regular"]
display(se_table.round(3).reset_index().rename(columns={"index": "feature"}))
add_check("Model 1: clustered / regular standard error (average)", round(se_table["ratio"].mean(), 2),
          "Above 1 means regular standard errors would have made results look more certain than they are.")

# COMMAND ----------

# MAGIC %md ### Model 1 robustness

# COMMAND ----------

robust = pd.DataFrame({"main": ols.params[features]})

# Each feature alone
alone = []
for col_name in features:
    single = smf.ols("log_rate ~ " + col_name + " + C(year)", data=d).fit()
    alone.append(single.params[col_name])
robust["alone"] = alone

# LCBH rate as the outcome
lcbh_d = df[(df["is_chicago"] == 1) & (df["lcbh_renters"] >= 50) & df["lcbh_rate"].notna()].copy()
lcbh_d = prepare_features(lcbh_d)
lcbh_d["log_lcbh_rate"] = np.log1p(lcbh_d["lcbh_rate"])
print("LCBH outcome:", len(lcbh_d), "tract-years,", lcbh_d["year"].min(), "to", lcbh_d["year"].max())
ols_lcbh = smf.ols("log_lcbh_rate ~ " + " + ".join(features) + " + C(year)", data=lcbh_d).fit()
robust["lcbh_outcome"] = ols_lcbh.params[features]

# Tracts with 200+ renter households
big = d[d["el_renters"] >= 200]
robust["renters_200_plus"] = smf.ols(formula, data=big).fit().params[features]

# Without the top 1% of rates
cutoff = d["el_rate"].quantile(0.99)
trimmed = d[d["el_rate"] <= cutoff]
robust["no_outliers"] = smf.ols(formula, data=trimmed).fit().params[features]

# Sign agreement with the main model
same_sign_counts = []
for col_name in features:
    count = 0
    for version in ["alone", "lcbh_outcome", "renters_200_plus", "no_outliers"]:
        if np.sign(robust.loc[col_name, version]) == np.sign(robust.loc[col_name, "main"]):
            count = count + 1
    same_sign_counts.append(str(count) + " of 4")
robust["same_sign_as_main"] = same_sign_counts
display(robust.round(3).reset_index().rename(columns={"index": "feature"}))

agree = 0
for col_name in features:
    if np.sign(robust.loc[col_name, "lcbh_outcome"]) == np.sign(robust.loc[col_name, "main"]):
        agree = agree + 1
add_check("Model 1: features with the same sign using LCBH data", str(agree) + " of " + str(len(features)),
          "Close to all means the findings don't depend on which eviction dataset is used.")

# COMMAND ----------

# MAGIC %md ## Model 2: Compare prediction models

# COMMAND ----------

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

x_cols = features + ["year"]
tuned_models = ["Ridge", "Lasso", "Decision tree"]


def make_models():
    folds = GroupKFold(n_splits=5)
    models = {}
    models["Baseline (predict the average)"] = DummyRegressor()
    models["Linear regression"] = make_pipeline(StandardScaler(), LinearRegression())
    models["Ridge"] = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                                   {"ridge__alpha": [0.1, 1, 10, 100]}, cv=folds)
    models["Lasso"] = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
                                   {"lasso__alpha": [0.001, 0.01, 0.1]}, cv=folds)
    models["Decision tree"] = GridSearchCV(DecisionTreeRegressor(random_state=0),
                                           {"max_depth": [3, 5, 8, 12]}, cv=folds)
    models["Random forest"] = RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                                                    random_state=0, n_jobs=-1)
    return models


def fit_model(name, model, x, y, groups):
    if name in tuned_models:
        model.fit(x, y, groups=groups)
    else:
        model.fit(x, y)


years = sorted(d["year"].unique())
print("Tract-years with data, by year:", d.groupby("year").size().to_dict())
test_years = []
for y in years[-n_test_years:]:
    test_years.append(int(y))
train = d[d["year"] < min(test_years)]
test = d[d["year"].isin(test_years)]
print("Train:", int(years[0]), "to", min(test_years) - 1, "(", len(train), "rows ) | Test:", test_years,
      "(", len(test), "rows )")

x_train, y_train, train_groups = train[x_cols], train["log_rate"], train["tract_id"]
x_test, y_test = test[x_cols], test["log_rate"]

models = make_models()
results = []
for name in models:
    model = models[name]
    fit_model(name, model, x_train, y_train, train_groups)
    train_pred = model.predict(x_train)
    pred = model.predict(x_test)
    tuned_setting = ""
    if name in tuned_models:
        tuned_setting = str(model.best_params_)
    results.append({
        "model": name,
        "train R2": r2_score(y_train, train_pred),
        "test R2": r2_score(y_test, pred),
        "RMSE (log scale)": mean_squared_error(y_test, pred) ** 0.5,
        "MAE (filings per 100 renters)": mean_absolute_error(np.expm1(y_test), np.expm1(pred)),
        "tuned setting": tuned_setting,
    })

# Baseline: each tract's own past average
tract_past_average = train.groupby("tract_id")["log_rate"].mean()
past_pred = test["tract_id"].map(tract_past_average).fillna(y_train.mean())
results.append({
    "model": "Baseline (tract's own past average)",
    "train R2": np.nan,
    "test R2": r2_score(y_test, past_pred),
    "RMSE (log scale)": mean_squared_error(y_test, past_pred) ** 0.5,
    "MAE (filings per 100 renters)": mean_absolute_error(np.expm1(y_test), np.expm1(past_pred)),
    "tuned setting": "",
})

results = pd.DataFrame(results).sort_values("test R2", ascending=False)
display(results.round(3))
best_name = results.iloc[0]["model"]
print("Best on the test years:", best_name)

# COMMAND ----------

# MAGIC %md ### What drives the predictions?

# COMMAND ----------

rf = models["Random forest"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Predicted vs actual
axes[0].scatter(np.expm1(y_test), np.expm1(rf.predict(x_test)), s=6, alpha=0.4, color=blue)
lim = np.expm1(y_test).quantile(0.99)
axes[0].plot([0, lim], [0, lim], color=grey, linewidth=1)
axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual rate", ylabel="Predicted rate")
axes[0].set_title("Random forest: predicted vs actual (test years)", loc="left")

# Random forest feature importance
importance = pd.Series(rf.feature_importances_, index=x_cols).sort_values()
axes[1].barh(importance.index, importance.values, color=blue)
axes[1].set_title("Random forest: feature importance", loc="left")

# Lasso coefficients
lasso_coef = pd.Series(models["Lasso"].best_estimator_[-1].coef_, index=x_cols).sort_values()
axes[2].barh(lasso_coef.index, lasso_coef.values, color=blue)
axes[2].axvline(0, color=grey, linewidth=1)
axes[2].set_title("Lasso coefficients (standardized)", loc="left")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### Model 2 checks

# COMMAND ----------

# Overfitting gap
rf_row = results[results["model"] == "Random forest"].iloc[0]
add_check("Model 2: random forest train R2 minus test R2", round(rf_row["train R2"] - rf_row["test R2"], 3),
          "A large gap (over ~0.15) suggests overfitting.")

# Test on tracts the models have never seen
x_all, y_all, groups_all = d[x_cols], d["log_rate"], d["tract_id"]
new_tract_scores = {}
for name in make_models():
    new_tract_scores[name] = []

for train_idx, test_idx in GroupKFold(n_splits=5).split(x_all, y_all, groups_all):
    fold_models = make_models()
    for name in fold_models:
        model = fold_models[name]
        fit_model(name, model, x_all.iloc[train_idx], y_all.iloc[train_idx], groups_all.iloc[train_idx])
        pred = model.predict(x_all.iloc[test_idx])
        new_tract_scores[name].append(r2_score(y_all.iloc[test_idx], pred))

compare_rows = []
for name in new_tract_scores:
    year_r2 = results[results["model"] == name]["test R2"].iloc[0]
    compare_rows.append({"model": name,
                         "R2, later years (same tracts)": year_r2,
                         "R2, new tracts": np.mean(new_tract_scores[name])})
compare = pd.DataFrame(compare_rows)
compare["drop"] = compare["R2, later years (same tracts)"] - compare["R2, new tracts"]
display(compare.round(3))

rf_compare = compare[compare["model"] == "Random forest"].iloc[0]
lin_compare = compare[compare["model"] == "Linear regression"].iloc[0]
add_check("Model 2: random forest R2 on new tracts", round(rf_compare["R2, new tracts"], 3),
          "Compare with its R2 on later years. A big drop means it was mostly remembering tracts.")
add_check("Model 2: linear regression R2 on new tracts", round(lin_compare["R2, new tracts"], 3),
          "If close to the random forest here, the extra accuracy above came from remembering tracts.")

# Past-average baseline
past_r2 = results[results["model"] == "Baseline (tract's own past average)"]["test R2"].iloc[0]
add_check("Model 2: R2 of 'tract's own past average' baseline", round(past_r2, 3),
          "If this matches the best model, eviction rates mostly stay the same within a tract.")

# COMMAND ----------

# MAGIC %md ## Model 3: Did 2012 foreclosures predict later evictions?

# COMMAND ----------

fcm = d[d["year"] == 2012][["tract_id", "foreclosures", "el_renters", "el_rate"] + features]
fcm = fcm.rename(columns={"el_rate": "rate_2012"})
for y in [2010, 2011, 2013, 2014]:
    year_rates = d[d["year"] == y][["tract_id", "el_rate"]]
    year_rates = year_rates.rename(columns={"el_rate": "rate_" + str(y)})
    fcm = fcm.merge(year_rates, on="tract_id", how="left")
fcm["foreclosure_rate_2012"] = fcm["foreclosures"] / fcm["el_renters"] * 100

controls = " + ".join(features)
versions = [
    ["basic (2013 outcome)", "np.log1p(rate_2013) ~ np.log1p(foreclosure_rate_2012) + np.log1p(rate_2012)"],
    ["with neighbourhood controls", "np.log1p(rate_2013) ~ np.log1p(foreclosure_rate_2012) + np.log1p(rate_2012) + " + controls],
    ["2014 outcome, with controls", "np.log1p(rate_2014) ~ np.log1p(foreclosure_rate_2012) + np.log1p(rate_2012) + " + controls],
    ["placebo: 2011 outcome, with controls", "np.log1p(rate_2011) ~ np.log1p(foreclosure_rate_2012) + np.log1p(rate_2010) + " + controls],
]

fc_rows = []
for version in versions:
    name = version[0]
    fc_formula = version[1]
    fit = smf.ols(fc_formula, data=fcm).fit()
    term = "np.log1p(foreclosure_rate_2012)"
    fc_rows.append({"version": name, "foreclosure coef": fit.params[term],
                    "p_value": fit.pvalues[term], "tracts": int(fit.nobs)})
fc_results = pd.DataFrame(fc_rows)
display(fc_results.round(4))

with_controls = fc_results[fc_results["version"] == "with neighbourhood controls"].iloc[0]
placebo = fc_results[fc_results["version"] == "placebo: 2011 outcome, with controls"].iloc[0]
add_check("Model 3: foreclosure coef with neighbourhood controls (p-value)",
          str(round(with_controls["foreclosure coef"], 3)) + " (p=" + str(round(with_controls["p_value"], 3)) + ")",
          "Still positive with p < 0.05 means foreclosures add something beyond neighbourhood conditions.")
add_check("Model 3: placebo coef, 2011 outcome (p-value)",
          str(round(placebo["foreclosure coef"], 3)) + " (p=" + str(round(placebo["p_value"], 3)) + ")",
          "Should be near 0 / not significant. If it's as big as the main result, be cautious.")

plt.figure(figsize=(6, 4))
plt.scatter(np.log1p(fcm["foreclosure_rate_2012"]), np.log1p(fcm["rate_2013"]), s=8, alpha=0.5, color=blue)
plt.xlabel("log(1 + foreclosure rate, 2012)")
plt.ylabel("log(1 + eviction rate, 2013)")
plt.title("Foreclosures 2012 vs evictions 2013", loc="left")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.show()

# COMMAND ----------

# MAGIC %md ### Check: do the two eviction sources agree?

# COMMAND ----------

both = d[d["lcbh_rate"].notna() & (d["lcbh_renters"] > 0)]
overall_corr = both["el_rate"].corr(both["lcbh_rate"])
print(len(both), "tract-years in both sources | correlation of rates:", round(overall_corr, 3))
add_check("Eviction Lab vs LCBH: correlation of rates", round(overall_corr, 3),
          "Above ~0.7 means the two sources broadly agree.")

year_rows = []
for y in sorted(both["year"].unique()):
    one_year = both[both["year"] == y]
    year_rows.append({"year": int(y),
                      "correlation": one_year["el_rate"].corr(one_year["lcbh_rate"]),
                      "eviction_lab_filings": one_year["el_filings"].sum(),
                      "lcbh_filings": one_year["lcbh_filings"].sum()})
by_year = pd.DataFrame(year_rows)
by_year["ratio"] = by_year["eviction_lab_filings"] / by_year["lcbh_filings"]
display(by_year.round(3))

plt.figure(figsize=(5, 5))
plt.scatter(np.log1p(both["lcbh_rate"]), np.log1p(both["el_rate"]), s=5, alpha=0.3, color=blue)
plt.xlabel("log(1 + LCBH rate)")
plt.ylabel("log(1 + Eviction Lab rate)")
plt.title("Eviction Lab vs LCBH, same tract and year", loc="left")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 4: k-means, grouping tracts into neighbourhood types
# MAGIC One row per tract: the standardized features plus the tract's average eviction rate.

# COMMAND ----------

from sklearn.cluster import KMeans
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_score, adjusted_rand_score

tract_features = d.groupby("tract_id")[features + ["log_rate"]].mean()
x_k = StandardScaler().fit_transform(tract_features)

k_values = [2, 3, 4, 5, 6, 7, 8]
inertia = []
silhouette = []
for k in k_values:
    km_try = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x_k)
    inertia.append(km_try.inertia_)
    silhouette.append(silhouette_score(x_k, km_try.labels_))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(k_values, inertia, color=blue, marker="o", linewidth=2)
axes[0].set_title("Elbow plot (inertia)", loc="left")
axes[1].plot(k_values, silhouette, color=blue, marker="o", linewidth=2)
axes[1].set_title("Silhouette score (higher = better separated)", loc="left")
for ax in axes:
    ax.set_xlabel("k")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# COMMAND ----------

km = KMeans(n_clusters=k_clusters, n_init=10, random_state=0).fit(x_k)
tract_features["cluster"] = km.labels_

# Renumber clusters from lowest to highest eviction rate
order = tract_features.groupby("cluster")["log_rate"].mean().sort_values().index
new_label = {}
for position in range(len(order)):
    new_label[order[position]] = position + 1
tract_features["cluster"] = tract_features["cluster"].map(new_label)

profile = tract_features.groupby("cluster").mean()
profile["avg_filing_rate"] = np.expm1(profile["log_rate"])
profile = profile.drop(columns="log_rate")
profile.insert(0, "tracts", tract_features["cluster"].value_counts().sort_index())
display(profile.round(2).reset_index())

shades = ["#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0f4a96", "#0e3d7e", "#0a2f63", "#06224a"]
cluster_map = tracts.merge(tract_features["cluster"].reset_index(), on="tract_id")
fig, ax = plt.subplots(figsize=(7, 9))
legend_items = []
for c in sorted(cluster_map["cluster"].unique()):
    cluster_map[cluster_map["cluster"] == c].plot(ax=ax, color=shades[c - 1], edgecolor="white", linewidth=0.3)
    legend_items.append(Patch(color=shades[c - 1], label="Cluster " + str(c)))
ax.legend(handles=legend_items, title="1 = lowest eviction rate", loc="lower left", frameon=False)
ax.set_title("Chicago tract types (k-means)", loc="left")
ax.set_axis_off()
plt.show()

# COMMAND ----------

# MAGIC %md ### Model 4 checks

# COMMAND ----------

# Stability across random seeds
rand_scores = []
for seed in [1, 2, 3, 4, 5]:
    km_seed = KMeans(n_clusters=k_clusters, n_init=10, random_state=seed).fit(x_k)
    rand_scores.append(adjusted_rand_score(km.labels_, km_seed.labels_))
print("Adjusted Rand index vs the main run:", np.round(rand_scores, 3))
add_check("Model 4: lowest adjusted Rand index across seeds", round(min(rand_scores), 3),
          "Close to 1 means the clusters don't depend on the random start.")

# Clusters from neighbourhood features only
x_features_only = StandardScaler().fit_transform(tract_features[features])
km_features = KMeans(n_clusters=k_clusters, n_init=10, random_state=0).fit(x_features_only)
tract_features["cluster_features_only"] = km_features.labels_

rate_by_group = np.expm1(tract_features.groupby("cluster_features_only")["log_rate"].mean()).sort_values()
group_rows = []
for group in rate_by_group.index:
    n_tracts = (tract_features["cluster_features_only"] == group).sum()
    group_rows.append({"group": int(group), "tracts": int(n_tracts),
                       "avg_filing_rate": rate_by_group[group]})
display(pd.DataFrame(group_rows).round(2))

ratio = rate_by_group.iloc[-1] / rate_by_group.iloc[0]
add_check("Model 4: highest / lowest eviction rate, clusters built without eviction data", round(ratio, 1),
          "Well above 1 means neighbourhood type alone separates high- and low-eviction areas.")

# How the two clusterings line up
crosstab = pd.crosstab(tract_features["cluster"], tract_features["cluster_features_only"])
new_names = {}
for col_name in crosstab.columns:
    new_names[col_name] = "features_only_group_" + str(col_name)
crosstab = crosstab.rename(columns=new_names)
display(crosstab.reset_index())

# COMMAND ----------

# MAGIC %md ## Checks summary

# COMMAND ----------

checks_table = pd.DataFrame(checks)
display(checks_table)

# COMMAND ----------

# MAGIC %md ## Save Results

# COMMAND ----------

def save(pdf, name):
    new_names = {}
    for col_name in pdf.columns:
        new_names[col_name] = safe_name(str(col_name))
    pdf = pdf.rename(columns=new_names)
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable("cs483eviction.default." + name))
    print("saved", name)


save(results, "model_comparison")
save(compare, "model_new_tract_check")
save(coef.reset_index().rename(columns={"index": "feature"}), "ols_coefficients")
save(robust.reset_index().rename(columns={"index": "feature"}), "ols_robustness")
save(tract_features.reset_index(), "tract_clusters")
save(checks_table, "checks_summary")

# COMMAND ----------

# MAGIC %md ## AI Usage Statement
# MAGIC We used Claude (Opus 5.5, Anthropic) to help write the PySpark and Python code that combines the Eviction Lab, Chicago Health Atlas, LCBH and Cook County foreclosure data into one tract-by-year table, as permitted by CS483. Claude also wrote the code for our models (linear regression, ridge, lasso, decision tree, random forest and k-means) and the checks that test them. All AI-written code was run on our data, checked against the actual outputs, and edited by our team, and we wrote the final conclusions ourselves. 