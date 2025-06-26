# 📊 Survival Analysis for Customer Churn

## 🔥 Motivation
Customer churn is a major business problem in subscription services, banking, telecom, and e-commerce. Instead of simply predicting **if** a customer will churn, **Survival Analysis** helps predict **when** they are likely to churn, enabling more targeted retention strategies.

This project uses **survival models** like Kaplan-Meier and Cox Proportional Hazards to understand customer lifetime and predict churn timelines.

---

## 🧠 Introduction

Survival Analysis is a branch of statistics that deals with **time-to-event data**, where the event could be:
- Customer churn
- Subscription cancellation
- Last order activity
- System failure, etc.

In this project, we apply **lifelines** (Python library) to model **seller/customer churn** over time, estimating survival probabilities and identifying key factors that increase churn risk.

---

## 📦 Project Description

This project includes the following key components:

- ✅ Data preprocessing (e.g., `Tenure`, censoring, feature engineering)
- 📈 Kaplan-Meier estimation for group-level survival curves
- ⚙️ Cox Proportional Hazards Model for multivariate risk prediction
- 📊 Individual-level predictions: survival probabilities, churn risk scores, and expected lifetime
- 🧪 Model validation with concordance index and AIC

---

## 💼 Use Cases

| Use Case                        | Description |
|---------------------------------|-------------|
| 🧑‍💼 Churn prediction            | Predict **when** a customer is likely to churn |
| 🧠 Risk segmentation             | Score customers by churn risk (high vs low) |
| ⏳ Lifetime value estimation     | Estimate remaining lifespan of a customer |
| 📊 Retention strategy           | Identify who needs retention efforts now |

---

## 📥 Input

Expected input data should include:

| Column Name         | Description                         |
|---------------------|-------------------------------------|
| `CustomerId`        | Unique customer identifier          |
| `Tenure`            | Number of years customer has stayed |
| `Exited`            | 1 if churned, 0 otherwise (censored)|
| `EstimatedSalary`   | Numeric feature                     |
| `Gender`            | Categorical feature                 |
| `Geography`         | Categorical feature                 |
| `...`               | Additional behavioral attributes    |

---

## 📤 Output

The output includes:

| Column Name            | Description                                             |
|------------------------|---------------------------------------------------------|
| `risk_score`           | Cox model partial hazard score (higher = more risky)    |
| `median_survival_time` | Predicted time until 50% chance of churn                |
| `churn_5years`         | Probability that customer will churn within 5 years     |
| `churn_70%`            | Estimated year when customer hits 70% churn probability |
| `live_0year`, `live_5years` | Survival probabilities at given time points       |

You can also generate:
- Kaplan-Meier plots for different groups
- Feature importance from Cox model
- Risk-based segmentation dashboards

---

## 📚 Future Work
- Time-varying covariates (dynamic behavior tracking)
- Incorporate business events (e.g., promotions, support tickets)
- Integrate with dashboards for real-time decision-making

---

## 🔧 Requirements
- Python ≥ 3.8  
- pandas, numpy  
- lifelines  
- matplotlib, seaborn  
- scikit-learn (optional)

---

## 📬 Contact
For questions or contributions, feel free to reach out to `yourname@email.com`.

