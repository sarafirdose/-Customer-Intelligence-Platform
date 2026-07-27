# Visual Analytics Catalog

This catalog presents all 18 visual charts generated during the Exploratory Data Analysis (EDA) stage. It groups the visualizations by feature category (Demographics, Churn Rates, Financials, Services, and Correlations) and details their structural findings.

---

## 👥 1. Demographics Analysis

### Gender Distribution
- **Image**: ![Gender Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_gender.png)
- **Description**: Churn is evenly split between genders: Female churn is **26.9%** and Male is **26.1%**. Gender holds low predictor information.

### Senior Citizen Distribution
- **Image**: ![Senior Citizen Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_senior_citizen.png)
- **Description**: Senior Citizens have a much higher churn rate (**41.7%**) compared to non-seniors (**23.6%**).

### Partner vs No Partner
- **Image**: ![Partner Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_partner.png)
- **Description**: Customers without a partner churn at a higher rate (**32.9%**) than those with a partner (**19.6%**).

### Dependents Distribution
- **Image**: ![Dependents Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_dependents.png)
- **Description**: Customers without dependents show high churn (**31.2%**) compared to those with dependents (**15.4%**).

---

## 📊 2. Churn Drivers & Distribution

### Churn Balance
- **Image**: ![Churn Distribution Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_distribution.png)
- **Description**: The target class `Churn` contains **26.5% Yes** and **73.5% No**. This represents a moderate class imbalance.

### Churn by Contract Type
- **Image**: ![Contract Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_contract.png)
- **Description**: Month-to-month contracts represent the highest risk factor (**42.7%** churn) compared to Two-year contracts (**2.8%**).

### Churn by Payment Method
- **Image**: ![Payment Method Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_payment_method.png)
- **Description**: Electronic Check is the highest churn payment method (**45.3%**). Automatic payment options mitigate churn down to **~15%**.

### Churn by Internet Service
- **Image**: ![Internet Service Churn Chart](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/churn_by_internet_service.png)
- **Description**: Fiber optic users churn at **41.9%**, significantly higher than DSL (**19.0%**) or No Internet (**7.4%**).

---

## 💵 3. Financial Analysis

### Monthly Charges Histogram
- **Image**: ![Monthly Charges Distribution](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/monthly_charges_distribution.png)
- **Description**: Bimodal distribution with peaks near \$20 (basic telephone) and \$80-\$100 (high-speed fiber with add-ons).

### Total Charges Histogram
- **Image**: ![Total Charges Distribution](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/total_charges_distribution.png)
- **Description**: Strongly right-skewed distribution, reflecting high concentrations of short-tenure customers with lower accumulated spend.

### Monthly Charges vs Churn Boxplot
- **Image**: ![Monthly Charges vs Churn Boxplot](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/monthly_charges_vs_churn.png)
- **Description**: Churned customers have a higher median monthly charge (\$79.65) compared to retained customers (\$64.43). Price sensitivity is key.

### Total Charges vs Churn Boxplot
- **Image**: ![Total Charges vs Churn Boxplot](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/total_charges_vs_churn.png)
- **Description**: Churned customers show lower median total charges due to their much shorter tenure.

---

## ⏳ 4. Tenure Analysis

### Tenure Distribution
- **Image**: ![Tenure Distribution](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/tenure_distribution.png)
- **Description**: High customer concentrations at month 1 (new signups) and month 72 (long-term loyal cohort).

### Tenure Density by Churn
- **Image**: ![Tenure Density vs Churn](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/tenure_density_vs_churn.png)
- **Description**: Churn risk is heavily concentrated in the first 12 months. Once customers survive past 24 months, churn density drops significantly.

### Tenure Boxplot by Churn
- **Image**: ![Tenure Boxplot](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/tenure_boxplot_vs_churn.png)
- **Description**: Churned customers have a median tenure of only 10 months, whereas retained customers have a median tenure of 38 months.

---

## 🛠️ 5. Services & Correlations

### Services Comparison (Security & Tech Support)
- **Image**: ![Services vs Churn](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/services_vs_churn.png)
- **Description**: Subscribing to Online Security and Tech Support add-ons reduces churn rate to under **15%**. Lacking these services pushes churn rate above **40%**.

### Pearson Heatmap
- **Image**: ![Pearson Heatmap](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/pearson_heatmap.png)
- **Description**: Highlights strong linear relationships. `TotalCharges` is highly collinear with `tenure_months` (r = **0.83**) and `monthly_charges` (r = **0.65**).

### Cramér's V Heatmap
- **Image**: ![Cramér's V Heatmap](file:///c:/Users/saraf/Downloads/intership%20p-1/Customer-Intelligence-Platform/reports/plots/cramers_v_heatmap.png)
- **Description**: Visualizes strength of associations for categorical columns. `contract_type` (0.41), `online_security` (0.35), `tech_support` (0.34), and `internet_service` (0.32) are the strongest categorical churn indicators.
