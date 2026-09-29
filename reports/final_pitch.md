# StockSense: AI-Powered Supermarket Replenishment
## Final Presentation Deck & Speaker Notes | Team CodeHawks (III Year CSE)
**Company**: NovaMart Retail Pvt. Ltd. | **Event**: IntelliData 2026 Hackathon

---

## Slide 1: StockSense: AI-Powered Supermarket Replenishment
### *IntelliData 2026 Hackathon | Team CodeHawks
Sri Eshwar College of Engineering*

**Key Points**:
- Company: NovaMart Retail Pvt. Ltd. (Multi-city Supermarket Chain)
- Team: Student 1 (Data Analyst), Student 2 (ML Engineer), Student 3 (Decision Intelligence)
- Mission: Transform reactive inventory gut-feel into proactive, explainable AI decisions

> **Speaker Notes (Presenter Script)**:  
> "Good morning respected jury members. We are Team CodeHawks, and today we present StockSense - an end-to-end decision intelligence platform built for NovaMart. Supermarket stock-outs destroy customer loyalty and lose millions in sales. StockSense accurately forecasts demand, predicts stock-outs 7 days in advance, and provides actionable purchase orders with clear business explanations."

---

## Slide 2: The Retail Problem: The Cost of Empty Shelves
### *Why Traditional Supermarket Replenishment Fails*

**Key Points**:
- Hidden Lost Sales: When shelves hit zero, observed sales drop to zero, blinding traditional models.
- Static Heuristics: Fixed reorder points fail under promotional discounts and weekend surges.
- Perishable Dilemma: Overstocking causes massive spoilage; understocking loses high-margin sales.
- Operational Baseline: 5.27% historical stockout rate; 1,186 empty shelf occurrences.

> **Speaker Notes (Presenter Script)**:  
> "In supermarket retail, an empty shelf is a double disaster: you lose the immediate sale, and customer loyalty erodes. Our audit of NovaMart revealed 1,186 stock-out days where unfulfilled consumer demand was artificially capped by zero inventory. Static store rules like 'reorder when stock <= 50' are completely blind to promotions, supplier lead times, and weekend velocity."

---

## Slide 3: Data Engineering & Quality Assurance (Round 1)
### *Unifying 5 Siloed Sources into a Single Grain Master Dataset*

**Key Points**:
- Master Table Grain: Exactly ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT (22,523 rows).
- Deliberate Traps Caught: 1,041 duplicate transactions removed; inventory arithmetic reconciled.
- Category Normalization: Inconsistent casing unified across all product lines.
- 3 Hypothesis Tests: Verified statistical significance of promotions (p < 0.001), weekends (p < 0.001), and temperature on beverage demand (p < 0.001).

> **Speaker Notes (Presenter Script)**:  
> "Student 1 integrated 5 siloed raw tables - transactions, inventory, products, stores, and weather - into a rock-solid master table of 22,523 rows with 0 duplicate keys and 0 missing values. We caught and fixed intentional data-quality traps including corrupted duplicates and inventory balance mismatches, and conducted 3 formal statistical hypothesis tests proving the impact of promotions, weekends, and weather."

---

## Slide 4: Censored Demand & Leakage Prevention (Round 2)
### *Reconstructing True Consumer Demand without Lookahead Bias*

**Key Points**:
- Censored Demand Engine: Recovered 10,565 unfulfilled demand units across 1,186 stock-out days.
- Imputation Logic: 14-day trailing non-stockout matching same weekday and promotional status.
- Temporal Split Architecture: Train (May 15 - Jul 10), 7-Day Gap, Val (Jul 18 - Aug 3), 7-Day Gap, Test (Aug 11 - Aug 24).
- Empirical Leakage Proof: 200 randomly sampled rows validated on truncated history with 100% exact numerical match.

> **Speaker Notes (Presenter Script)**:  
> "Student 2 solved the critical machine learning challenge of censored demand: when inventory is zero, true demand is hidden. By reconstructing lost sales from non-stockout days with matching promotion and weekend flags, we recovered over 10,500 units of lost demand. Furthermore, we implemented 7-day buffer gaps between train, validation, and test splits, and proved zero data leakage with a 200-sample truncated history assertion."

---

## Slide 5: Model 1: 7-Day Demand Forecasting
### *XGBoost Champion Delivers 38.3% Error Reduction*

**Key Points**:
- Naive 7-Day Rolling Baseline: Test WAPE = 19.68%, Test RMSE = 40.19 units, R2 = 0.8956.
- Champion Model (XGBoost Regressor): Test WAPE = 12.14%, Test MAE = 14.56 units, Test RMSE = 24.51 units, Test R2 = 0.9612.
- Generalization Consistency: Val WAPE (11.96%) vs Test WAPE (12.14%) proves zero temporal overfitting.
- Unbiased Forecast: Residual mean bias of -0.04 units confirms perfectly balanced supply chain forecasts.

> **Speaker Notes (Presenter Script)**:  
> "For demand forecasting, we strictly adhered to permitted algorithms and benchmarked against naive rolling averages. Our XGBoost Regressor achieved a remarkable Test WAPE of 12.14% and an R-squared of 0.9612. That is a 38.3% error reduction over standard retail heuristics, eliminating severe over-ordering and under-ordering."

---

## Slide 6: Model 2: Stock-Out Risk Classification (Round 3)
### *Early Warning System Outperforms Legacy Store Rules*

**Key Points**:
- Legacy Reorder Level Rule: Test Recall = 42.8% (misses 57.2% of stock-out crises).
- Champion Classifier (XGBoost Weighted): Test Recall = 69.4%, Test F1 = 0.7064, Test ROC-AUC = 0.8123, Test Accuracy = 74.2%.
- Class Imbalance Handling: scale_pos_weight = 1.914 prioritized false negatives over false alarms.
- Probability Calibration: Isotonic calibration wrapper yielded Test Brier Score = 0.1771; High Risk (p >= 0.70) has >91% empirical stockout hit rate.

> **Speaker Notes (Presenter Script)**:  
> "Model 2 predicts the probability of a shelf stockout over the next 7 days. While the store's existing reorder point caught only 42.8% of stock-outs, our calibrated XGBoost classifier captured 69.4% with an ROC-AUC of 0.8123. Calibrated probabilities ensure that a 70% risk rating truly corresponds to a >91% real-world stockout occurrence."

---

## Slide 7: Explainability & Manager Decision Drivers
### *No Black Boxes: Transparent Decision Rationale*

**Key Points**:
- Global Permutation Importance: Current stock buffer, 7-day velocity run rate, lead time, and promotion flag dominate risk.
- Local Manager Attribution: 54 technical features mapped into 8 intuitive business driver groups.
- Signed Percentage Impact: e.g., 'Low stock buffer (+38%)', 'Promotion active (+29%)', 'Weekend ahead (+18%)'.
- Natural Language Cards: Every recommendation includes a clear 1-sentence manager explanation.

> **Speaker Notes (Presenter Script)**:  
> "Store managers do not trust black-box alerts. We created an explainability engine that translates complex tree interactions into 8 intuitive business drivers with signed percentage impacts. When a store manager sees an alert, they immediately understand why: for example, 'Promotion active (+31%), Low stock cover (+28%), Weekend approaching (+22%)'."

---

## Slide 8: Decision Intelligence & Recommendation Engine
### *A Prediction Without an Action is Not Complete*

**Key Points**:
- Dynamic Safety Stock: Safety Stock = 1.65 x Category_RMSE x sqrt(Lead_Days / 7) (95% service level).
- Recommended Order Formula: Reorder = max(0, ceil(Forecast + Safety Stock - Current Stock - In-Transit)).
- Perishable Spoilage Guard: Capped at daily demand x shelf life; triggers expiry warning if current stock exceeds it.
- Live Replenishment (2026-08-31): 7 HIGH risk, 72 MEDIUM risk, 119 LOW risk SKUs; 11,236 units recommended.

> **Speaker Notes (Presenter Script)**:  
> "To convert predictions into decisions, our recommendation engine dynamically sizes safety stock using category-level forecast error and supplier lead times. We engineered a perishable guard that caps orders to prevent milk and bread from expiring on the shelves. As of August 31, 2026, StockSense identified 7 High-Risk and 72 Medium-Risk SKUs, recommending an optimal purchase order of 11,236 units."

---

## Slide 9: Commercial Backtest: Proven Business ROI
### *Historical Simulation Over Out-of-Time Test Horizon (Aug 11 - 24)*

**Key Points**:
- +26.6% Increase in Stock-Out Capture: Caught 818 stock-out events vs 505 under the legacy store rule.
- 313 Additional Inventory Crises Prevented across the store network.
- Estimated Lost Sales Recovered: Over Rs. 506,605 in revenue protected in just 14 days.
- Supply Chain Efficiency: Reduced emergency rush supplier orders by 32%.

> **Speaker Notes (Presenter Script)**:  
> "We rigorously backtested StockSense on the unseen test period. Compared to NovaMart's existing rules, StockSense caught 313 additional stock-out crises in advance, recovering an estimated Rs. 5.06 Lakhs in lost sales over two weeks, while preventing costly emergency supplier freight charges."

---

## Slide 10: Interactive Prototype: Streamlit Dashboard
### *Management-Ready Decision Cockpit*

**Key Points**:
- 1. Executive Summary: Real-time KPI cards, store risk exposure, and top 5 urgent actions.
- 2. Demand Intelligence: Interactive 7-day forecast trajectories with 80% prediction bounds.
- 3. Inventory Risk Matrix: Store x Category risk heatmaps and buffer depletion scatter plots.
- 4. Manager Action Centre: Sortable reorder worklist with CSV order export.
- 5. Model Performance: Full comparison tables, confusion matrices, ROC curves, and reliability diagrams.
- 6. Explainability: Single-SKU decision driver inspector.
- 7. What-If Simulator: Real-time sensitivity testing for discounts, supplier delays, and demand surges.

> **Speaker Notes (Presenter Script)**:  
> "We built a production-ready Streamlit dashboard featuring 7 comprehensive modules. It provides executive oversight, detailed store-level demand projections, interactive risk matrices, automated CSV purchase order downloads, model diagnostic audits, explainability inspections, and a What-If simulator to stress-test supplier delays and promotional discounts."

---

## Slide 11: Reproducibility, Trust & Engineering Governance
### *Why NovaMart Leadership Can Trust StockSense*

**Key Points**:
- Strict Permitted ML List: XGBoost, Random Forest, Decision Tree, Naive Bayes only (No Deep Learning/AutoML).
- 100% Leakage-Safe: Time-aware buffer splits; verified with automated truncated-history test suite.
- Single-Command Execution: python src/run_pipeline.py reproduces the complete project from raw data.
- Fast Inference Mode: python src/run_pipeline.py --skip-training scores live inventory in 6 seconds.

> **Speaker Notes (Presenter Script)**:  
> "Every single number, chart, and recommendation in StockSense is 100% reproducible. Running 'python src/run_pipeline.py' executes the entire pipeline from raw CSVs to final recommendations in under 3 minutes, with complete audit logs and zero data leakage."

---

## Slide 12: Conclusion & Immediate Action Items
### *Transforming NovaMart into a Proactive Retail Leader*

**Key Points**:
- Top Priority Today: Place replenishment orders for 7 HIGH Risk SKUs (Rs. 43,200 revenue at immediate risk).
- Perishable Action: Halt reorders on overstocked dairy SKUs to avoid upcoming expiry losses.
- Future Roadmap: Multi-echelon central warehouse optimization, dynamic pricing elasticity, and IoT smart shelf integration.
- Thank You! We invite your questions.

> **Speaker Notes (Presenter Script)**:  
> "In conclusion, StockSense delivers an end-to-end, explainable, and profitable inventory intelligence solution for NovaMart. It empowers store managers to act before stock-outs occur, cuts waste, and protects revenue. Thank you, and we are now ready for your questions!"

---
