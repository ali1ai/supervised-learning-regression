#  Titanic Survival Prediction — Logistic Regression Project
### Repository: `supervised-learning-regression`

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine--Learning-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

##  Overview
This project predicts **Titanic passenger survival** using **Logistic Regression**.  
It demonstrates a **complete machine learning workflow** including:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training using Scikit-Learn Pipelines
- Model Evaluation and Cross-Validation
- Interpretability using Odds Ratios and SHAP Values

---

##  Model Used
**Model:** Logistic Regression (Binary Classification)  
**Framework:** Scikit-Learn  
**Goal:** Predict survival probability of Titanic passengers (`survived` = 0 or 1).

---

##  Project Files

| File | Description |
|------|--------------|
| `Titanic_Logistic_Regression_Project.ipynb` | Main Jupyter Notebook with full workflow (EDA → Modeling → SHAP) |
| `titanic_logistic_pipeline.pkl` | Trained Logistic Regression pipeline (serialized with Joblib) |
| `LICENSE` | MIT License |
| `README.md` | Project overview and documentation |

*(Optional: You can later add `requirements.txt` and a `/notebooks` folder for organization.)*

---

##  Features & Techniques
- Cleaned missing values (Age, Embarked, Deck)  
- Engineered new features: `family_size`, `is_alone`, `fare_per_person`, `age_bin`  
- Scaled numerical features and one-hot encoded categorical ones  
- Created a `Pipeline` for reproducible ML workflow  
- Used cross-validation (F1-score) for evaluation  
- Explained predictions using **SHAP summary plots** and **odds ratios**

---

##  Exploratory Data Analysis
- **Visualizations:** Seaborn plots for survival rates by gender, class, age, and title  
- **Correlation Heatmap:** Relationships among numeric variables  
- **Violin & Bar Plots:** Survival patterns by social and demographic groups  

---

##  Results Summary
| Metric | Value (approx.) |
|--------|-----------------|
| Accuracy | ~0.80 |
| F1-score (5-fold CV) | ~0.78 |
| Precision | ~0.79 |
| Recall | ~0.76 |

**Top Predictors of Survival:**
- Passenger gender (`sex`)
- Passenger class (`pclass`)
- Family status (`is_alone`, `family_size`)
- Ticket fare per person (`fare_per_person`)

---

##  Interpretability
- **Odds Ratios:** Explain the effect of each feature on survival odds.  
- **SHAP Values:** Visualize global feature importance and individual impact on predictions.

---

##  Installation & Usage

### Clone the repository:
```bash
git clone https://github.com/yourusername/supervised-learning-regression.git
cd supervised-learning-regression
