# Vehicle Insurance Premium Predictor

A machine learning project that predicts whether a vehicle insurance premium is **above 300** or **300 and below** using customer, driver, and vehicle-related features.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insurance-predictor-app-coo.streamlit.app)

*Tip: Cmd+Click to open the demo in a new tab.*

## Overview

This project transforms the insurance premium prediction task into a binary classification problem by creating a target variable:

- `1` → Premium > 300
- `0` → Premium ≤ 300

Several classification models were trained and evaluated, and a **Random Forest classifier** was selected as the final model. To improve usability in deployment, the final version of the app uses only the **top 10 most important features**, which produced performance nearly identical to the full-feature model.

The final model was deployed in a **Streamlit app** for both:
- single-record prediction
- batch CSV prediction

---

## Problem Statement

Insurance companies often need fast ways to assess whether a policy premium is likely to fall above or below a certain threshold. This project builds a classification model that helps estimate that outcome based on available policyholder and vehicle information.

---

## Dataset Features

Kaggle Link - https://www.kaggle.com/datasets/mexwell/motor-vehicle-insurance-portfolio

The original dataset contains variables related to:

- policy details
- driver profile
- vehicle characteristics
- premium amount

Examples of fields used in the project include:

- `Seniority`
- `Payment`
- `Power`
- `Cylinder_capacity`
- `Value_vehicle`
- `Length`
- `Weight`
- `age`
- `year_licensed`
- `car_age`

---

## Feature Engineering

The original `Premium` column was converted into a binary target:

```python
target_300 = (Premium > 300).astype(int)
```

## Run Locally

1. Clone the repository
2. Install dependencies
3. Run the Streamlit app
