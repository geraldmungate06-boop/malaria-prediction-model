# Malaria Incidence Prediction — Zimbabwe

A machine-learning project that forecasts Zimbabwe's national yearly **malaria incidence** (estimated cases per 1,000 population at risk) from WHO malaria indicators, with an interactive Streamlit app.

> **Note:** this is a student learning project. It works at national, yearly level using a very small dataset. It does **not** diagnose or detect malaria in individual patients and must not be used for medical or public-health decisions.

## Project overview

| | |
|---|---|
| **Goal** | Predict estimated malaria incidence (per 1,000 population at risk) for Zimbabwe |
| **Data** | WHO Global Health Observatory malaria indicators for Zimbabwe, 2000–2024 |
| **Model** | Random Forest Regressor (scikit-learn, 100 trees) |
| **Evaluation** | Chronological split: train on 2000–2015 (16 years), test on 2016–2024 (9 years) |
| **App** | Streamlit web app (`app.py`) |

## Repository contents

| File | Description |
|---|---|
| `malaria_model_corrected.ipynb` | Notebook: data cleaning, exploratory analysis, feature engineering, training, evaluation and saving the model |
| `malaria_indicators_zwe.csv` | WHO malaria indicators for Zimbabwe |
| `malaria_model.pkl` | Saved model, scaler, feature list and feature medians |
| `app.py` | Streamlit app that loads the saved model and returns a prediction |
| `README.md` | This file |

## Method

1. **Data cleaning:** kept rows with a numeric value and pivoted the indicators into one row per year (duplicate year/indicator pairs averaged).
2. **Indicators used:** estimated incidence (target), estimated mortality rate, confirmed cases, suspected cases, RDT-positive tests and imported cases.
3. **Feature engineering:** 1- and 2-year lags of each indicator, plus year encodings. Missing values were filled with the median.
4. **Scaling:** `StandardScaler`.
5. **Training:** Random Forest Regressor with a time-based train/test split, so the model is never tested on years earlier than those it trained on.
6. **Saving:** the model, scaler, feature list and medians are stored together in `malaria_model.pkl` so the app applies exactly the same preprocessing as training.

## Results

Performance on the held-out test years (2016–2024):

| Metric | Value |
|---|---|
| MAE | 20.55 per 1,000 |
| RMSE | 26.90 per 1,000 |
| R² | 0.25 |

The model performs better than two simple baselines I checked on the same test years: predicting the training average (R² = −1.53) and predicting the previous year's value (R² = −0.63). An R² of 0.25 is still modest, and with only 9 test years it should be read as indicative only.

## Limitations

- **Very small dataset:** only 25 yearly observations.
- **Missing indicators in early years:** confirmed cases, suspected cases, RDT positives and imported cases are only available from 2015, so they are mostly median-filled in the training years.
- **Estimated, not observed, values:** incidence and mortality are WHO model estimates. Incidence is a constant 102.6 for 2000–2006, which looks like a placeholder, and mortality is estimated alongside incidence, so the model is partly predicting one estimate from another. Mortality carries most of the feature importance.
- **No extrapolation:** Random Forests cannot predict outside the range of values seen in training, so predictions for inputs far from the data (or for years beyond 2024) are unreliable.
- **Fixed lag features in the app:** the app only collects seven inputs; the lag features are filled with training medians.
- **National level only:** no regional, climate or patient-level data is used.

## Run the app locally

1. Install the requirements:

   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```

2. Keep `app.py` and `malaria_model.pkl` in the same folder, then start the app:

   ```bash
   streamlit run app.py
   ```

3. Enter the indicators in the form and click **Predict**.

To retrain, open `malaria_model_corrected.ipynb` in Jupyter or Google Colab with `malaria_indicators_zwe.csv` in the same folder and run all cells. This regenerates `malaria_model.pkl`.

> **Security note:** `.pkl` files can execute code when loaded. Only load model files from sources you trust.

## Possible improvements

- Remove the 4-year `sin`/`cos` year encoding, which only produces three distinct values.
- Fit the scaler and median-fill on the training years only.
- Train a second model without the mortality indicator to see how much it drives the result.
- Use the time-series cross-validation folds for evaluation and tuning.
- Add regional, climate and rainfall data.

## Data source

World Health Organization, Global Health Observatory (GHO), malaria indicators for Zimbabwe.

## Author

Gerald Mungate — BComm Honours Data Science and Informatics, Midlands State University
