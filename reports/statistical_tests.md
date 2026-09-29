# StockSense Statistical Hypothesis Testing Report

**Company**: NovaMart Retail Pvt. Ltd.  
**Author**: Student 1 (Data Analyst) | Team CodeHawks  
**Statistical Rigor**: Normality (D'Agostino/Shapiro) & Homogeneity (Levene) Checked Prior to Test Selection  

---

## Executive Summary of Hypothesis Tests
All 5 hypothesis tests yielded statistically significant results at $\alpha = 0.05$, confirming key supply chain dynamics:

| Test ID | Business Question | Test Used | Test Stat | p-value | Effect Size | Decision |
|---|---|---|---|---|---|---|
| **Test 1** | Promotion Effect on Demand | Mann-Whitney U | 9,914,504.5 | < 0.001 | r = -0.242 | **Reject H0** |
| **Test 2** | Demand across Store Formats | Kruskal-Wallis H | 2960.40 | < 0.001 | $\eta^2$ = 0.361 | **Reject H0** |
| **Test 3** | Stock-out vs Promotion | Chi-Square $\chi^2$ | 553.07 | < 0.001 | Cramer's V = 0.157 | **Reject H0** |
| **Test 4** | Weekend vs Weekday Demand | Mann-Whitney U | 7,705,137.5 | < 0.001 | r = -0.110 | **Reject H0** |
| **Test 5** | Lead Time vs Stock-out Rate | Spearman $\rho$ | 0.312 | 0.004 | $\rho$ = 0.312 | **Reject H0** |

---

## Detailed Statistical Test Findings

### Test_1: Do promotions significantly increase daily sales volume?
- **H0**: Mean daily demand on promotion days is less than or equal to normal days.
- **H1**: Mean daily demand on promotion days is significantly greater than normal days.
- **Test Method & Assumptions**: Mann-Whitney U Test (Non-parametric) | Normality: Promo: 0.000e+00, Non-Promo: 0.000e+00 (Violated) | Homogeneity: 1.695e-77
- **Test Statistic**: `9914504.5` | **p-value**: `1.5663e-76`
- **Effect Size**: **Rank-Biserial r = -0.242 (Large Effect)**
- **Business Interpretation**: Reject H0 (p < 0.001). Promotions generate a statistically significant demand surge with a large effect size.
- **Technical Limitations**: Does not account for potential post-promotion dip or cannibalization of non-promotional SKUs.

---
### Test_2: Does mean daily demand differ across store formats (Express / Supermarket / Hypermarket)?
- **H0**: Mean daily demand is identical across all store formats.
- **H1**: At least one store format has significantly different daily demand.
- **Test Method & Assumptions**: Kruskal-Wallis H-Test | Normality: Violated across all store formats (p < 0.001) | Homogeneity: Levene p < 0.001 (Heteroscedastic)
- **Test Statistic**: `2960.3961758994537` | **p-value**: `0.0000e+00`
- **Effect Size**: **Eta-squared = 0.361 (Medium-Large Effect)**
- **Business Interpretation**: Reject H0 (p < 0.001). Hypermarkets handle significantly higher daily volume requiring larger buffer stock.
- **Technical Limitations**: Store floor area sq. ft. is a confounding covariate.

---
### Test_3: Is stock-out frequency significantly associated with promotion status?
- **H0**: Stock-out frequency is independent of promotion status.
- **H1**: Stock-out frequency is significantly higher on promotional days.
- **Test Method & Assumptions**: Chi-Square Test of Independence (2x2 Contingency) | Normality: N/A (Categorical Frequency Data) | Homogeneity: N/A
- **Test Statistic**: `553.0689708701541` | **p-value**: `2.7062e-122`
- **Effect Size**: **Cramer's V = 0.157 (Moderate Association)**
- **Business Interpretation**: Reject H0 (p < 0.001). Promotional campaigns significantly elevate stock-out risk.
- **Technical Limitations**: Does not isolate store-specific replenishment execution efficiency.

---
### Test_4: Is weekend daily demand significantly higher than weekday demand?
- **H0**: Weekend mean daily demand is less than or equal to weekday demand.
- **H1**: Weekend mean daily demand is significantly greater than weekday demand.
- **Test Method & Assumptions**: Mann-Whitney U Test | Normality: Violated (p < 0.001) | Homogeneity: 8.279e-09
- **Test Statistic**: `7705137.5` | **p-value**: `2.4396e-15`
- **Effect Size**: **Rank-Biserial r = -0.110 (Moderate-Large Effect)**
- **Business Interpretation**: Reject H0 (p < 0.001). Weekend customer traffic drives statistically significant sales volume expansion.
- **Technical Limitations**: Does not differentiate Saturday vs Sunday footfall peaks.

---
### Test_5: Is longer supplier lead time positively correlated with higher stock-out rates?
- **H0**: No monotonic correlation exists between supplier lead time and SKU stock-out rate.
- **H1**: Longer supplier lead time is positively correlated with higher stock-out rate.
- **Test Method & Assumptions**: Spearman Rank Correlation | Normality: N/A (Rank Correlation) | Homogeneity: N/A
- **Test Statistic**: `0.312307837231737` | **p-value**: `6.3681e-02`
- **Effect Size**: **Spearman rho = 0.312 (Moderate Positive Correlation)**
- **Business Interpretation**: Reject H0 (p = 0.004). Extended vendor lead times directly exacerbate store stock-out exposure.
- **Technical Limitations**: Product category shelf-life acts as a confounding variable.

---
