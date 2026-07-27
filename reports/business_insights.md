# Business Insights Report: Telco Customer Churn & LTV Analysis

This report provides a business-oriented translation of the Exploratory Data Analysis (EDA) performed on the 7,043 customer accounts in our database. It answers key strategic questions, outlines customer segments, and provides recommendations for reducing churn and increasing Lifetime Value (LTV).

---

## 📊 Key Executive Statistics
- **Total Customer Database**: 7,043 accounts
- **Overall Churn Rate**: **26.5%** (1,869 churned, 5,174 retained)
- **Average Monthly Charges**: \$64.76
- **Average Customer Tenure**: 32.37 months
- **Average Lifetime Value (approx.)**: \$2,280 in accumulated revenue per customer

---

## 🔍 Critical Business Questions Answered

### 1. Why do customers churn?
Churn is driven by a combination of friction points:
- **Contract Type**: Flexible month-to-month contracts have a very high churn rate (**42.7%**), whereas locked-in contracts are highly stable. Month-to-month contracts account for **88.5%** of all churned customers.
- **Internet Infrastructure**: Customers subscribed to **Fiber Optic** internet churn at **41.9%**, which is more than double the rate of **DSL** customers (**19.0%**). This suggests potential service quality issues, high price points, or onboarding friction.
- **Billing Methods**: Customers paying via **Electronic Check** churn at **45.3%**, compared to only ~15% for automatic billing (Credit Card / Bank Transfer).
- **Service Support Add-ons**: Customers who do **not** have **Online Security** or **Tech Support** churn at rates of **41.8%** and **41.6%** respectively, showing that account complexity and "stickiness" protect against churn.

### 2. Which contract type churns the most?
- **Month-to-month**: **42.7% Churn Rate** (3,875 total accounts, 1,655 churned).
- **One year**: **11.3% Churn Rate** (1,473 total accounts, 166 churned).
- **Two year**: **2.8% Churn Rate** (1,695 total accounts, 48 churned).

> [!WARNING]
> Month-to-month customers represent **88.5% of all churn cases**. Converting these customers to at least a 1-year contract reduces churn probability by **73.5%**.

### 3. Which payment method has the highest churn?
- **Electronic check**: **45.3% Churn Rate** (2,365 accounts, 1,071 churned).
- **Mailed check**: **19.1% Churn Rate** (1,612 accounts, 308 churned).
- **Bank transfer (automatic)**: **16.7% Churn Rate** (1,544 accounts, 258 churned).
- **Credit card (automatic)**: **15.2% Churn Rate** (1,522 accounts, 232 churned).

Automatic payment methods show a **66% reduction in churn** compared to manual electronic checks.

### 4. Do higher monthly charges correlate with churn?
Yes, higher monthly charges are statistically associated with higher churn:
- **Retained Customer Median Monthly Charges**: **\$64.43**
- **Churned Customer Median Monthly Charges**: **\$79.65**
- **Insight**: Customers with monthly bills exceeding **\$70.00** show a sharp increase in churn density. Price sensitivity is a major churn factor, especially for Fiber Optic accounts where the baseline fee is higher.

### 5. Does tenure reduce churn?
Yes, tenure is one of the strongest mitigators of churn:
- **Short-tenure (<= 12 months) Churn Rate**: **47.4% Churn Rate**
- **Long-tenure (> 12 months) Churn Rate**: **17.1% Churn Rate**
- **Early-stage Risk**: Churn risk is concentrated in the **first 6 months** of the customer lifecycle (often during onboarding). Once a customer reaches **24 months**, their churn probability drops to **under 10%**.

---

## 👥 Customer Segment Performance

### 🏆 The Most Valuable Customer Segment (Low Risk / High LTV)
- **Profile**: Two-year contract + Automatic billing + Multiple services (e.g. Online Security and Tech Support) + Tenure > 48 months.
- **Churn Rate**: **< 1.5%**
- **Business Action**: Protect this segment. Offer loyalty rewards, upgrade hardware for free, and use them as benchmark profiles for referrals.

### ⚠️ The Highest Risk Customer Segment (High Risk / High Monthly Fees)
- **Profile**: Month-to-month contract + Fiber Optic internet + Electronic check billing + No online security/tech support.
- **Churn Rate**: **55.0% - 60.0%**
- **Business Action**: Target with active marketing campaigns. Offer incentives to transition to paperless credit card auto-pay or bundle a free security trial to increase retention.

---

## 🎯 Strategic Business Recommendations

1. **Auto-Pay Incentives**: Offer a one-time \$5 or \$10 billing credit for transitioning from Electronic Check to Credit Card or Bank Transfer automatic billing.
2. **First-Year Retention Focus**: Introduce an onboarding check-in program (e.g. at month 1, 3, and 6) with proactive customer support outreach to resolve billing or performance friction.
3. **Fiber Optic Support Auditing**: Perform an operational audit on Fiber Optic setup and reliability. Because Fiber Optic customers have high monthly charges, their churn carries the highest revenue loss impact.
4. **Security and Support Bundling**: Bundle "Online Security" and "Tech Support" into baseline packages (or offer them as free add-ons for the first 3 months) to increase service density and make accounts more sticky.
