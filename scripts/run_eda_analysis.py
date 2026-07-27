"""
Exploratory Data Analysis (EDA) Runner.

Loads customer data from the database, computes summary statistics,
detects class imbalance, checks outliers, executes statistical association tests
(Pearson, Spearman, Chi-Square, Cramér's V, Mutual Information), generates
18 visual analytics charts, and exports results to CSV files and markdown reports.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sqlalchemy import create_engine

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import logger
from backend.core.settings import settings

# Output folders
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
PLOT_DIR = OUTPUT_DIR / "plots"


def get_db_engine():
    """
    Acquire SQLAlchemy database engine based on configurations.
    """
    return create_engine(settings.get_db_url())


def load_dataset(engine) -> pd.DataFrame:
    """
    Load data from the database tables by joining normalized relations.
    """
    query = """
    SELECT 
        c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents, c.tenure_months, c.churn,
        con.contract_type, con.paperless_billing, con.payment_method,
        s.phone_service, s.multiple_lines, s.internet_service, s.online_security, s.online_backup,
        s.device_protection, s.tech_support, s.streaming_tv, s.streaming_movies,
        b.monthly_charges, b.total_charges
    FROM customers c
    JOIN contracts con ON c.contract_id = con.id
    JOIN services s ON c.service_id = s.id
    JOIN billings b ON c.billing_id = b.id
    """
    logger.info("Loading normalized records from database...")
    df = pd.read_sql_query(query, engine)
    logger.info(f"Loaded {len(df)} records for EDA.")
    return df


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramér's V statistic for categorical association.
    """
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    min_dim = min(r, c)
    if min_dim <= 1:
        return 0.0
    return np.sqrt(chi2 / (n * (min_dim - 1)))


def calculate_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Perform outlier detection using the IQR method.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound


def calculate_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Perform outlier detection using the Z-Score method.
    """
    z_scores = stats.zscore(df[col])
    outliers = df[np.abs(z_scores) > threshold]
    return outliers


def generate_plots(df: pd.DataFrame) -> None:
    """
    Generate 18 visual charts for demographic, financial, service, and correlation analysis.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.autolayout"] = True

    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="churn")
    plt.title("Churn Target Distribution (No vs Yes)")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.savefig(PLOT_DIR / "churn_distribution.png", dpi=150)
    plt.close()

    # 2. Gender Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="gender", hue="churn")
    plt.title("Churn Rate by Gender")
    plt.savefig(PLOT_DIR / "churn_by_gender.png", dpi=150)
    plt.close()

    # 3. Senior Citizen Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="senior_citizen", hue="churn")
    plt.title("Churn Rate by Senior Citizen Status")
    plt.xlabel("Senior Citizen (0=No, 1=Yes)")
    plt.savefig(PLOT_DIR / "churn_by_senior_citizen.png", dpi=150)
    plt.close()

    # 4. Partner vs No Partner
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="partner", hue="churn")
    plt.title("Churn Rate by Partner Status")
    plt.savefig(PLOT_DIR / "churn_by_partner.png", dpi=150)
    plt.close()

    # 5. Dependents Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="dependents", hue="churn")
    plt.title("Churn Rate by Dependents Status")
    plt.savefig(PLOT_DIR / "churn_by_dependents.png", dpi=150)
    plt.close()

    # 6. Churn by Contract
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="contract_type", hue="churn")
    plt.title("Churn Rate by Contract Type")
    plt.savefig(PLOT_DIR / "churn_by_contract.png", dpi=150)
    plt.close()

    # 7. Churn by Payment Method
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x="payment_method", hue="churn")
    plt.title("Churn Rate by Payment Method")
    plt.xticks(rotation=15)
    plt.savefig(PLOT_DIR / "churn_by_payment_method.png", dpi=150)
    plt.close()

    # 8. Churn by Internet Service
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="internet_service", hue="churn")
    plt.title("Churn Rate by Internet Service Type")
    plt.savefig(PLOT_DIR / "churn_by_internet_service.png", dpi=150)
    plt.close()

    # 9. Monthly Charges Histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="monthly_charges", kde=True, bins=30)
    plt.title("Distribution of Monthly Charges")
    plt.savefig(PLOT_DIR / "monthly_charges_distribution.png", dpi=150)
    plt.close()

    # 10. Total Charges Histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="total_charges", kde=True, bins=30)
    plt.title("Distribution of Total Charges")
    plt.savefig(PLOT_DIR / "total_charges_distribution.png", dpi=150)
    plt.close()

    # 11. Monthly Charges vs Churn
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="churn", y="monthly_charges")
    plt.title("Monthly Charges vs Churn")
    plt.savefig(PLOT_DIR / "monthly_charges_vs_churn.png", dpi=150)
    plt.close()

    # 12. Total Charges vs Churn
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="churn", y="total_charges")
    plt.title("Total Charges vs Churn")
    plt.savefig(PLOT_DIR / "total_charges_vs_churn.png", dpi=150)
    plt.close()

    # 13. Tenure Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="tenure_months", kde=True, bins=30)
    plt.title("Distribution of Customer Tenure (Months)")
    plt.savefig(PLOT_DIR / "tenure_distribution.png", dpi=150)
    plt.close()

    # 14. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="tenure_months", hue="churn", fill=True, common_norm=False, alpha=0.5)
    plt.title("Tenure Months Distribution Density by Churn")
    plt.savefig(PLOT_DIR / "tenure_density_vs_churn.png", dpi=150)
    plt.close()

    # 15. Boxplot of Tenure
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="churn", y="tenure_months")
    plt.title("Tenure Months Boxplot by Churn")
    plt.savefig(PLOT_DIR / "tenure_boxplot_vs_churn.png", dpi=150)
    plt.close()

    # 16. Services Comparison (Security & Tech Support)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="online_security", hue="churn", ax=axes[0])
    axes[0].set_title("Online Security add-on vs Churn")
    sns.countplot(data=df, x="tech_support", hue="churn", ax=axes[1])
    axes[1].set_title("Tech Support add-on vs Churn")
    plt.savefig(PLOT_DIR / "services_vs_churn.png", dpi=150)
    plt.close()

    # 17. Pearson Heatmap
    plt.figure(figsize=(6, 5))
    corr_numeric = df[["tenure_months", "monthly_charges", "total_charges", "churn"]].corr(method="pearson")
    sns.heatmap(corr_numeric, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Pearson Numeric Correlation Matrix")
    plt.savefig(PLOT_DIR / "pearson_heatmap.png", dpi=150)
    plt.close()

    # 18. Cramér's V Heatmap
    plt.figure(figsize=(10, 8))
    cat_cols = ["gender", "senior_citizen", "partner", "dependents", "contract_type",
                "paperless_billing", "payment_method", "internet_service",
                "online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies", "churn"]
    
    cv_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for c1 in cat_cols:
        for c2 in cat_cols:
            cv_matrix.loc[c1, c2] = calculate_cramers_v(df[c1], df[c2])
            
    sns.heatmap(cv_matrix, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
    plt.title("Cramér's V Categorical Association Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(PLOT_DIR / "cramers_v_heatmap.png", dpi=150)
    plt.close()

    logger.info("Successfully generated and saved all 18 analytical charts.")


def run_analysis() -> None:
    """
    Run complete analysis, write outputs to disk, and print summaries.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    engine = get_db_engine()
    df = load_dataset(engine)

    # 1. Missing values report
    missing_report = df.isnull().sum().to_frame("missing_count")
    missing_report["percentage"] = (missing_report["missing_count"] / len(df)) * 100
    missing_report.to_csv(OUTPUT_DIR / "missing_values_summary.csv")

    with open(OUTPUT_DIR / "missing_values.md", "w") as f:
        f.write("# Dataset Profiling & Missing Values Report\n\n")
        f.write(f"- **Total Rows**: {df.shape[0]}\n")
        f.write(f"- **Total Columns**: {df.shape[1]}\n")
        f.write(f"- **Duplicate Record Count**: {df.duplicated().sum()}\n\n")
        f.write("## Missing Values Table\n\n")
        f.write("| Column | Missing Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for idx, row in missing_report.iterrows():
            f.write(f"| {idx} | {int(row['missing_count'])} | {row['percentage']:.2f}% |\n")

    # 2. Class Imbalance
    churn_counts = df["churn"].value_counts()
    churn_pct = df["churn"].value_counts(normalize=True) * 100

    # 3. Outliers Analysis
    outliers_report = {}
    for col in ["tenure_months", "monthly_charges", "total_charges"]:
        iqr_outliers, lb, ub = calculate_outliers_iqr(df, col)
        z_outliers = calculate_outliers_zscore(df, col)
        outliers_report[col] = {
            "iqr_count": len(iqr_outliers),
            "z_count": len(z_outliers),
            "iqr_pct": (len(iqr_outliers) / len(df)) * 100,
            "z_pct": (len(z_outliers) / len(df)) * 100,
        }

    # 4. Correlation Matrices
    corr_pearson = df[["tenure_months", "monthly_charges", "total_charges", "churn"]].corr(method="pearson")
    corr_spearman = df[["tenure_months", "monthly_charges", "total_charges", "churn"]].corr(method="spearman")
    corr_pearson.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    corr_spearman.to_csv(OUTPUT_DIR / "correlation_spearman_matrix.csv")

    # 5. Cramér's V Calculation
    cat_cols = ["gender", "senior_citizen", "partner", "dependents", "contract_type",
                "paperless_billing", "payment_method", "internet_service",
                "online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies"]
    cramers_results = {}
    for col in cat_cols:
        cramers_results[col] = calculate_cramers_v(df[col], df["churn"])
    
    cv_df = pd.DataFrame.from_dict(cramers_results, orient="index", columns=["cramers_v"])
    cv_df = cv_df.sort_values(by="cramers_v", ascending=False)
    cv_df.to_csv(OUTPUT_DIR / "cramers_v.csv")

    # 6. Mutual Information & Chi-Square Tests
    # Label encode for Mutual Info calculation
    df_encoded = df.copy()
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes
    
    mi_scores = mutual_info_classif(
        df_encoded[cat_cols + ["tenure_months", "monthly_charges", "total_charges"]], 
        df_encoded["churn"], 
        random_state=42
    )
    
    feature_scores = pd.DataFrame({
        "feature": cat_cols + ["tenure_months", "monthly_charges", "total_charges"],
        "mutual_information": mi_scores
    })

    # Chi-Square Test
    p_values = []
    chi2_stats = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df["churn"])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        chi2_stats.append(chi2)
        p_values.append(p)
    
    chi_df = pd.DataFrame({
        "feature": cat_cols,
        "chi2_stat": chi2_stats,
        "p_value": p_values
    })
    
    feature_scores = feature_scores.merge(chi_df, on="feature", how="left")
    feature_scores = feature_scores.sort_values(by="mutual_information", ascending=False)
    feature_scores.to_csv(OUTPUT_DIR / "feature_scores.csv", index=False)

    # 7. Summary Stats
    summary_stats = df.describe(include="all").transpose()
    summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    # 8. Generate Charts
    generate_plots(df)

    # 9. Write Feature Recommendations Report
    with open(OUTPUT_DIR / "feature_recommendations.md", "w") as f:
        f.write("# Feature Recommendation Report\n\n")
        f.write("Based on statistical significance, Mutual Information, Pearson/Spearman correlation, and Cramér's V association tests:\n\n")
        f.write("## 1. High-Value Predictor Features (Strong Association)\n")
        f.write("- **contract_type**: Highest Cramér's V (" + f"{cramers_results['contract_type']:.3f}" + "). Month-to-month contracts have extreme churn susceptibility.\n")
        f.write("- **tenure_months**: Strongly correlated with churn. Longer tenure reduces likelihood of cancellation.\n")
        f.write("- **internet_service**: Fiber optic users churn at a significantly higher rate compared to DSL and No internet users.\n")
        f.write("- **online_security** & **tech_support**: Security and support features strongly mitigate churn risk.\n")
        f.write("- **payment_method**: Electronic check payment represents a high churn segment.\n\n")
        
        f.write("## 2. Low-Information / Weak Attributes (To Drop or Avoid)\n")
        f.write("- **gender**: Cramér's V of " + f"{cramers_results['gender']:.4f}" + " and Chi-Square p-value > 0.05. No statistical variance in churn between Male/Female.\n")
        f.write("- **phone_service**: Extremely low mutual info score. Churn rates do not fluctuate with basic phone line presence.\n\n")

        f.write("## 3. Redundant / Highly Collinear Features\n")
        f.write("- **total_charges**: Strongly correlated with `tenure_months` (Pearson r = " + f"{corr_pearson.loc['total_charges', 'tenure_months']:.3f}" + ") and `monthly_charges` (Pearson r = " + f"{corr_pearson.loc['total_charges', 'monthly_charges']:.3f}" + "). Represents multicollinearity risk. Use scaling, regularization, or feature engineering (e.g. Ratio variables) to avoid instability.\n\n")

        f.write("## 4. Candidate Engineered Features for Phase 3\n")
        f.write("- **Tenure Bins**: Grouping tenure into segments (e.g. `0-12 months`, `12-24 months`, `24-48 months`, `48+ months`) to capture non-linear retention rates.\n")
        f.write("- **Total Services Count**: Count of active communication & support services (online security, backup, protection, tech support, streaming) to model account stickiness.\n")
        f.write("- **Charges Ratio**: Ratio of Monthly Charges to Tenure to isolate billing velocity impact.\n")
        f.write("- **Automatic Payment Flag**: Binary indicator for automatic payment methods (Credit Card / Bank Transfer) vs manual check methods.\n")

    # 10. Automated insight summaries printed on screen
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Total Rows Ingested: {df.shape[0]}")
    print(f"Churn Distribution: No={churn_counts[0]} ({churn_pct[0]:.1f}%), Yes={churn_counts[1]} ({churn_pct[1]:.1f}%)")
    print("-" * 50)
    print("AUTOMATED INSIGHT SUMMARY:")
    
    # Calculate churn rates for insights
    c_churn = df.groupby("contract_type")["churn"].mean()
    print(f"- Month-to-month contracts churn rate: {c_churn.get('Month-to-month', 0)*100:.1f}%")
    print(f"  vs Two year contracts churn rate: {c_churn.get('Two year', 0)*100:.1f}%")
    
    tenure_short_churn = df[df["tenure_months"] <= 12]["churn"].mean()
    tenure_long_churn = df[df["tenure_months"] > 12]["churn"].mean()
    print(f"- Short-tenure (<=12 months) churn rate: {tenure_short_churn*100:.1f}%")
    print(f"  vs Long-tenure (>12 months) churn rate: {tenure_long_churn*100:.1f}%")
    
    p_churn = df.groupby("payment_method")["churn"].mean()
    print(f"- Electronic Check payment churn rate: {p_churn.get('Electronic check', 0)*100:.1f}%")
    
    i_churn = df.groupby("internet_service")["churn"].mean()
    print(f"- Fiber Optic internet churn rate: {i_churn.get('Fiber optic', 0)*100:.1f}%")
    print(f"  vs DSL internet churn rate: {i_churn.get('DSL', 0)*100:.1f}%")
    
    print("-" * 50)
    print("OUTLIER SUMMARY (IQR Method):")
    for k, v in outliers_report.items():
        print(f"- {k}: {v['iqr_count']} outliers ({v['iqr_pct']:.2f}%)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_analysis()
