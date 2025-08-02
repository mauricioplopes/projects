# Neutropenia Analysis Using MIMIC Data

This repository contains the code and report from **Practice 03** of the course _Data Science and Visualization in Healthcare – MO413_, offered by the Institute of Computing at UNICAMP during the first semester of 2023. The project was developed by **Maurício Pereira Lopes (RA 225242)** under the supervision of Professor **André Santanchè**.

- [Final report](REPORT.MD)
- [Code](Pratica3.ipynb)

## Objective

The main goal is to assess, for patients with neutropenia, the association between clinical and laboratory features and two outcomes:
- **Death**
- **Length of hospital stay (in days)**

The analysis separates the patients into two groups:
- Those diagnosed with cancer
- Those without a cancer diagnosis

## Data Source

The data was extracted from the **MIMIC** (Medical Information Mart for Intensive Care) database and provided in four CSV files:

- `neutro_admissions.csv`: patient admission data.
- `neutro_diagnoses.csv`: diagnoses related to each admission.
- `neutro_itemid_loinc.csv`: catalog of laboratory test types.
- `neutro_labevents_min.csv`: laboratory test results.

⚠️ **Note**: Access to MIMIC data requires official authorization.

## Methodology

The analysis was divided into three main stages:

### 1. Descriptive Analysis

- General characterization of the dataset: race distribution, admission types, hospital stay durations.
- Death frequencies related to admission durations.
- Frequencies of diagnoses and lab tests in death cases.

### 2. Correlation Analysis

- Association analysis using the **Dython** library.
- Low correlations found overall:
  - Maximum correlation of 23% between death and length of stay.
  - Diagnoses showed up to 37% correlation with the death outcome.

### 3. Predictive Modeling

#### Data Preparation:
- Selected only the most recent admission per patient.
- Included only the first occurrence of each lab test.
- Categorical variables were one-hot encoded.
- Resulting dataset: 4,209 rows × 814 features.

#### Models Used:
- **Logistic Regression**
- **Random Forest Classifier**

#### Techniques:
- Class balancing via class weights.
- Generation of synthetic data using **ADASYN**.

#### Results:
- Severe class imbalance: only 7% of samples represent death outcomes.
- Logistic Regression:
  - Accuracy: 58.4%
  - F1 Score: 11.5%
- Random Forest:
  - Accuracy: 93.7%
  - F1 Score: 0.0% (no true positives predicted)

Even with synthetic balancing, the models performed poorly on the test set, failing to generalize.

## Conclusion

The analysis proved challenging due to:
- High data variability and redundancy (e.g., repeated exams and admissions).
- Weak correlations between features and outcomes.
- Severe class imbalance in the prediction task.

Even when isolating cancer patients, the models struggled to produce meaningful predictions. Although synthetic data helped in training and validation, this did not translate to improved performance on the test data.

Future efforts could include:
- More precise clinical variables captured at admission.
- Selecting only specific lab tests relevant to neutropenia.
- Improved diagnosis classification.

These adjustments may lead to better predictive models and insights.


⚠️ This project does not include the original MIMIC data files due to access restrictions. To reproduce the results, obtain access to the MIMIC database and prepare the datasets as described in the report.