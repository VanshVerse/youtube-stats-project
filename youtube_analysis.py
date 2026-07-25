"""
Analysis and Prediction of YouTube Video Popularity
Using Probability and Statistical Techniques

Probability and Statistics Course Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

# Change filename here if using a different region's file
DATA_FILE = "INvideos.csv"
CATEGORY_FILE = "IN_category_id.json"

df = pd.read_csv(DATA_FILE, encoding="latin1")

# Safety check: if the first row(s) are entirely NaN, the parser likely
# misread the delimiter/encoding. Show the raw file lines to help debug,
# then try a couple of common fallbacks automatically.
if df.head(5).isnull().all(axis=1).all():
    print("\n[WARNING] Data loaded as all-NaN — showing raw file lines for debugging:")
    with open(DATA_FILE, "r", encoding="latin1", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  Line {i}: {line[:200]}")

    print("\nRetrying with encoding='utf-8-sig' and engine='python'...")
    try:
        df2 = pd.read_csv(DATA_FILE, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        if not df2.head(5).isnull().all(axis=1).all():
            df = df2
            print("-> Fixed using utf-8-sig encoding.")
        else:
            print("-> Still NaN. Check the raw lines printed above — the file may be")
            print("   using a different delimiter (e.g. semicolon or tab), or may be corrupted.")
    except Exception as e:
        print(f"-> Retry failed: {e}")

print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nColumns:", list(df.columns))

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop duplicates
before = df.shape[0]
df = df.drop_duplicates(subset="video_id", keep="last")
print(f"\nRemoved {before - df.shape[0]} duplicate rows")

# Fill missing descriptions/tags with empty string (not needed for stats but avoids errors)
df["description"] = df.get("description", pd.Series()).fillna("")

# Map category_id -> category name using the JSON file
try:
    with open(CATEGORY_FILE, "r") as f:
        cat_data = json.load(f)
    cat_map = {int(item["id"]): item["snippet"]["title"] for item in cat_data["items"]}
    df["category_name"] = df["category_id"].map(cat_map)
except FileNotFoundError:
    print("Category JSON file not found — skipping category name mapping.")
    df["category_name"] = df["category_id"].astype(str)

# Convert publish_time to datetime and extract useful features
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_dayofweek"] = df["publish_time"].dt.day_name()
df["is_weekend"] = df["publish_time"].dt.dayofweek >= 5

# Remove rows with impossible values (0 views is invalid for a trending video)
df = df[df["views"] > 0]

print(f"\nFinal cleaned shape: {df.shape}")

# ============================================================
# STEP 3: DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Descriptive Statistics")
print("=" * 60)

num_cols = [c for c in ["views", "likes", "dislikes", "comment_count"] if c in df.columns]
if "comment_count" not in df.columns:
    print("\nNote: 'comment_count' column not found in this dataset — continuing without it.")
print(df[num_cols].describe())

print("\nMean, Median, Std Dev, Variance for key columns:")
for col in num_cols:
    print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
          f"std={df[col].std():.2f}, var={df[col].var():.2f}")

# Correlation matrix
corr = df[num_cols].corr()
print("\nCorrelation matrix:\n", corr)

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Views, Likes, Dislikes, Comments")
plt.tight_layout()
plt.savefig("output_correlation_heatmap.png")
plt.close()

# Views distribution (raw is highly skewed, so also plot log-transformed)
plt.figure()
sns.histplot(df["views"], bins=50, kde=True)
plt.title("Distribution of Views (Raw)")
plt.savefig("output_views_histogram.png")
plt.close()

df["log_views"] = np.log1p(df["views"])
plt.figure()
sns.histplot(df["log_views"], bins=50, kde=True)
plt.title("Distribution of Views (Log-transformed)")
plt.savefig("output_log_views_histogram.png")
plt.close()

# Boxplot: views by category (top 10 categories by count)
top_categories = df["category_name"].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df["category_name"].isin(top_categories)],
            x="category_name", y="log_views")
plt.xticks(rotation=45, ha="right")
plt.title("Log(Views) by Category")
plt.tight_layout()
plt.savefig("output_views_by_category_boxplot.png")
plt.close()

# Scatter: likes vs views
plt.figure()
sns.scatterplot(data=df, x="likes", y="views", alpha=0.3)
plt.title("Likes vs Views")
plt.savefig("output_likes_vs_views_scatter.png")
plt.close()

print("\nSaved plots: correlation heatmap, views histograms, boxplot, scatter plot")

# ============================================================
# STEP 4: PROBABILITY & DISTRIBUTION CHECKS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Probability & Distribution Analysis")
print("=" * 60)

# Shapiro-Wilk test works best on samples; use a random sample if data is large
sample = df["log_views"].sample(min(5000, len(df)), random_state=42)
stat, p_value = stats.shapiro(sample)
print(f"Shapiro-Wilk test on log(views): stat={stat:.4f}, p-value={p_value:.4g}")
if p_value < 0.05:
    print("-> Log(views) does NOT perfectly follow a Normal distribution (p < 0.05),")
    print("   but is much closer to Normal than raw views (common with real-world engagement data).")
else:
    print("-> Log(views) is consistent with a Normal distribution (p >= 0.05).")

# Probability a random trending video belongs to each category
cat_prob = df["category_name"].value_counts(normalize=True).head(10)
print("\nProbability of a trending video belonging to top categories:")
print(cat_prob)

# ============================================================
# STEP 5: HYPOTHESIS TESTING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Hypothesis Testing")
print("=" * 60)

# --- ANOVA: Do average views differ significantly across categories? ---
groups = [g["log_views"].values for _, g in df[df["category_name"].isin(top_categories)].groupby("category_name")]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA (log-views across top categories): F={f_stat:.4f}, p-value={p_anova:.4g}")
if p_anova < 0.05:
    print("-> Reject H0: average views differ significantly across categories.")
else:
    print("-> Fail to reject H0: no significant difference across categories.")

# --- t-test: Weekday vs Weekend views ---
weekday_views = df[df["is_weekend"] == False]["log_views"]
weekend_views = df[df["is_weekend"] == True]["log_views"]
t_stat, p_ttest = stats.ttest_ind(weekday_views, weekend_views, equal_var=False)
print(f"\nt-test (weekday vs weekend log-views): t={t_stat:.4f}, p-value={p_ttest:.4g}")
if p_ttest < 0.05:
    print("-> Reject H0: views significantly differ between weekday and weekend uploads.")
else:
    print("-> Fail to reject H0: no significant difference between weekday and weekend uploads.")

# ============================================================
# STEP 6: PREDICTION MODEL (Multiple Linear Regression)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Prediction Model — Multiple Linear Regression")
print("=" * 60)

features = [c for c in ["likes", "dislikes", "comment_count"] if c in df.columns]
X = df[features]
y = df["views"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel coefficients: {dict(zip(features, model.coef_))}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"\nR^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Actual vs predicted plot
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Actual vs Predicted Views")
plt.tight_layout()
plt.savefig("output_actual_vs_predicted.png")
plt.close()

print("\nSaved plot: output_actual_vs_predicted.png")

# ============================================================
# STEP 7: SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Summary of Findings")
print("=" * 60)
print(f"""
- Strongest correlation with views: check correlation matrix above
  (typically likes and comment_count correlate most strongly with views)
- ANOVA p-value = {p_anova:.4g} -> category {'does' if p_anova < 0.05 else 'does not'} significantly affect views
- Weekday/Weekend t-test p-value = {p_ttest:.4g}
- Regression model R^2 = {r2:.4f} (explains {r2*100:.1f}% of variance in views)

All output plots (.png) have been saved in the project folder — use these
directly in your presentation and report.
""")
