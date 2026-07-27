"""
Generate EDA Jupyter Notebook.

Programmatically creates a fully annotated notebooks/EDA.ipynb file containing
markdown explanations and code blocks matching our EDA pipeline.
"""

import json
import os
from pathlib import Path


def generate_notebook() -> None:
    """
    Construct notebooks/EDA.ipynb JSON format.
    """
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Customer Churn Prediction - Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "This notebook runs the complete statistical and visual analysis on the Telco Churn dataset to isolate primary drivers of customer churn and build key recommendations for feature engineering.\n",
                    "\n",
                    "## 📋 Objectives:\n",
                    "1. Profile dataset shapes, column types, and missing values.\n",
                    "2. Analyze class imbalance and target properties.\n",
                    "3. Perform numeric outlier detection (IQR and Z-score).\n",
                    "4. Run statistical association tests (Pearson, Spearman, Chi-Square, Cramér's V, and Mutual Information).\n",
                    "5. Generate visual distribution charts."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from scipy import stats\n",
                    "from sklearn.feature_selection import mutual_info_classif\n",
                    "from sqlalchemy import create_engine\n",
                    "\n",
                    "# Setup environment variables to use testing database configuration\n",
                    "os.environ['ENV'] = 'testing'\n",
                    "os.environ['USE_SQLITE_TEST'] = 'true'\n",
                    "\n",
                    "# Get DB connection engine\n",
                    "engine = create_engine('sqlite:///../test.db')\n",
                    "print(\"Connected to database successfully.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 💾 1. Load Normalized Relations\n",
                    "We join customer records with contracts, billing details, and communication subscription services."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "query = \"\"\"\n",
                    "SELECT \n",
                    "    c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents, c.tenure_months, c.churn,\n",
                    "    con.contract_type, con.paperless_billing, con.payment_method,\n",
                    "    s.phone_service, s.multiple_lines, s.internet_service, s.online_security, s.online_backup,\n",
                    "    s.device_protection, s.tech_support, s.streaming_tv, s.streaming_movies,\n",
                    "    b.monthly_charges, b.total_charges\n",
                    "FROM customers c\n",
                    "JOIN contracts con ON c.contract_id = con.id\n",
                    "JOIN services s ON c.service_id = s.id\n",
                    "JOIN billings b ON c.billing_id = b.id\n",
                    "\"\"\"\n",
                    "df = pd.read_sql_query(query, engine)\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 2. Profiling & Class Imbalance\n",
                    "Let's look at the dataset shape, data types, and check the target class imbalance distribution (`churn`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Dataset Shape: {df.shape}\")\n",
                    "print(f\"Duplicate Count: {df.duplicated().sum()}\")\n",
                    "print(f\"Missing Values Summary:\\n{df.isnull().sum()}\")\n",
                    "\n",
                    "churn_counts = df['churn'].value_counts()\n",
                    "churn_pct = df['churn'].value_counts(normalize=True) * 100\n",
                    "print(f\"\\nChurn Distribution:\\nNo: {churn_counts[0]} ({churn_pct[0]:.2f}%)\\nYes: {churn_counts[1]} ({churn_pct[1]:.2f}%)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 3. Outlier Analysis (IQR & Z-Score)\n",
                    "We inspect the numeric variables (`tenure_months`, `monthly_charges`, and `total_charges`) for statistical outliers."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for col in ['tenure_months', 'monthly_charges', 'total_charges']:\n",
                    "    # IQR Method\n",
                    "    q1 = df[col].quantile(0.25)\n",
                    "    q3 = df[col].quantile(0.75)\n",
                    "    iqr = q3 - q1\n",
                    "    lb = q1 - 1.5 * iqr\n",
                    "    ub = q3 + 1.5 * iqr\n",
                    "    iqr_outliers = df[(df[col] < lb) | (df[col] > ub)]\n",
                    "    \n",
                    "    # Z-Score Method\n",
                    "    z_scores = np.abs(stats.zscore(df[col]))\n",
                    "    z_outliers = df[z_scores > 3.0]\n",
                    "    \n",
                    "    print(f\"{col.upper()} outlier count: IQR={len(iqr_outliers)} ({len(iqr_outliers)/len(df)*100:.2f}%), Z-Score={len(z_outliers)} ({len(z_outliers)/len(df)*100:.2f}%)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🧪 4. Statistical Association Testing\n",
                    "We run Pearson/Spearman numeric correlations, Chi-Square contingency testing, Cramér's V categorical association, and Mutual Information feature scoring."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"--- 1. Pearson Numeric Correlations ---\")\n",
                    "print(df[['tenure_months', 'monthly_charges', 'total_charges', 'churn']].corr(method='pearson'))\n",
                    "\n",
                    "print(\"\\n--- 2. Cramér's V Categorical Associations ---\")\n",
                    "def cramers_v(x, y):\n",
                    "    contingency = pd.crosstab(x, y)\n",
                    "    chi2 = stats.chi2_contingency(contingency)[0]\n",
                    "    n = contingency.sum().sum()\n",
                    "    r, c = contingency.shape\n",
                    "    min_dim = min(r, c)\n",
                    "    return np.sqrt(chi2 / (n * (min_dim - 1))) if min_dim > 1 else 0.0\n",
                    "\n",
                    "cat_cols = ['gender', 'senior_citizen', 'partner', 'dependents', 'contract_type',\n",
                    "            'paperless_billing', 'payment_method', 'internet_service',\n",
                    "            'online_security', 'online_backup', 'device_protection', 'tech_support',\n",
                    "            'streaming_tv', 'streaming_movies']\n",
                    "\n",
                    "for col in cat_cols:\n",
                    "    contingency = pd.crosstab(df[col], df['churn'])\n",
                    "    chi2, p, _, _ = stats.chi2_contingency(contingency)\n",
                    "    cv = cramers_v(df[col], df['churn'])\n",
                    "    print(f\"{col:25} -> Cramér's V: {cv:.3f}, Chi2 p-val: {p:.4g}\")\n",
                    "\n",
                    "print(\"\\n--- 3. Mutual Information Scores ---\")\n",
                    "df_encoded = df.copy()\n",
                    "for col in cat_cols:\n",
                    "    df_encoded[col] = df_encoded[col].astype('category').cat.codes\n",
                    "\n",
                    "features = cat_cols + ['tenure_months', 'monthly_charges', 'total_charges']\n",
                    "mi_scores = mutual_info_classif(df_encoded[features], df_encoded['churn'], random_state=42)\n",
                    "mi_df = pd.DataFrame({'feature': features, 'mutual_info': mi_scores}).sort_values(by='mutual_info', ascending=False)\n",
                    "print(mi_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📈 5. Key Exploratory Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
                    "\n",
                    "# Churn vs Contract Type\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.countplot(data=df, x='contract_type', hue='churn')\n",
                    "plt.title(\"Churn Rate by Contract Type\")\n",
                    "plt.show()\n",
                    "\n",
                    "# Tenure Boxplot\n",
                    "plt.figure(figsize=(6, 4))\n",
                    "sns.boxplot(data=df, x='churn', y='tenure_months')\n",
                    "plt.title(\"Tenure Months Distribution by Churn\")\n",
                    "plt.show()\n",
                    "\n",
                    "# Monthly Charges vs Churn\n",
                    "plt.figure(figsize=(6, 4))\n",
                    "sns.boxplot(data=df, x='churn', y='monthly_charges')\n",
                    "plt.title(\"Monthly Charges vs Churn\")\n",
                    "plt.show()\n",
                    "\n",
                    "# Internet Service vs Churn\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.countplot(data=df, x='internet_service', hue='churn')\n",
                    "plt.title(\"Churn by Internet Service Type\")\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    os.makedirs(notebook_dir, exist_ok=True)
    notebook_file = notebook_dir / "EDA.ipynb"

    with open(notebook_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[INFO] Jupyter Notebook generated successfully at: {notebook_file}")


if __name__ == "__main__":
    generate_notebook()
