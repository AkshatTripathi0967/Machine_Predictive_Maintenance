# Predictive Maintenance Dashboard

Deployed Link :- https://machinepredictivemaintenance-qnpp6mmmychrccd4qhbvxh.streamlit.app/
A Streamlit dashboard that uses a trained Random Forest model to estimate the risk that each machine in a fleet will fail within the next 6 hours, so maintenance can be scheduled before a breakdown.

The data comes from the Microsoft Azure Predictive Maintenance dataset: 100 machines, hourly sensor readings (`volt`, `rotate`, `pressure`, `vibration`), error logs, maintenance history and failure records.

## Project structure

```
pdm_app/
├── app.py                    # The Streamlit dashboard
├── random_forest_model.pkl   # Your trained model (add this yourself)
├── holdout.csv.gz            # Held-out data the dashboard scores
├── train_model.py            # Optional: rebuilds holdout.csv.gz (and a fresh model)
├── requirements.txt
└── .streamlit/
    └── config.toml           # Colour theme
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints (usually http://localhost:8501).

Requirements:
- Python 3.9 or newer.
- `random_forest_model.pkl` in the same folder as `app.py`.
- A scikit-learn version close to the one that trained the model. Pickled models can fail to load across very different versions. If you see a load error, install the version you trained with.

The first launch is slow, because the app scores about 175,000 rows with your model. The result is cached, so later interactions are fast.

## What the dashboard shows

**Fleet monitor.** A header badge shows how many machines need inspection. The plant floor board has one tile per machine (grey is normal, amber is watch, red is alert), next to an inspection priority list ranked by risk. Use the Replay hour slider in the sidebar to move through the test period.

**Machine detail.** The risk over time for one machine, plotted against the alert threshold, with vertical lines at real failures. You can overlay sensor rolling averages.

**What-if prediction.** Start from a real record, edit any feature value, and see the risk change on a gauge. A global feature importance chart is included.

**Batch upload.** Upload a CSV of feature rows and download predictions. A template with the right column names is available from the tab.

## Sidebar controls

| Control | What it does |
|---|---|
| Compare alerts against | Chooses the ground truth: "failure exactly 6h ahead" (your notebook's `shift(-6)` target) or "any failure in the next 6h". |
| Alert threshold | Probability at which a machine is flagged. Default 0.085. Lower values catch more failures but raise more false alarms. |
| Replay hour | The hour of the test period shown on the fleet board. |
| Test-period accuracy | Recall, precision, F1 and PR-AUC at the current threshold, computed live. |

"Watch" is any machine with a risk above half the alert threshold.

## How it works

1. `holdout.csv.gz` holds the last 20% of each machine's history. Your model trained on the first 80%, so these rows are unseen data.
2. At startup the app reads the feature names from the model (`feature_names_in_`), checks the holdout file has all of them, and scores every row once.
3. The dashboard shows those scores alongside what actually happened, so you can judge the alerts.

The holdout file is a replay of past data for demonstration. It is not needed in a real deployment. There, the app would receive live feature rows instead (see below).

## Known data issues

These came up while building the app, and they affect how you should read the results.

- **Corrupted `datetime` column.** In the notebook, the line `df9["datetime"] = pd.to_datetime(df["datetime"])` copies datetimes from the failures table by row number, so almost every row has no datetime, and the first 761 rows of machine 1 are reordered. The app therefore uses an hour index instead of dates, and `train_model.py` drops those 761 rows. Fix this at the source by deleting that line.
- **Duplicate rows.** Merging the error and maintenance tables duplicated some machine-hours. `train_model.py` keeps one row per machine-hour.
- **Target definition.** `shift(-6)` marks a failure exactly 6 hours ahead, not any failure within 6 hours. For the "any failure in the next 6h" objective, use a forward 6-hour window and drop the last 6 rows per machine instead of labelling them 0.
- **Threshold selection.** Choose the threshold on a separate chronological validation slice, not the test set. `train_model.py` shows one way to do this (60/20/20 split per machine, with a 6-row gap at each boundary).

## Retraining

To rebuild the holdout file or train a fresh model from the final CSV:

```bash
python train_model.py path/to/finalcsv
```

This writes `holdout.csv.gz`, `model.joblib` and `metrics.json`. To use the new model in the dashboard, save it with `joblib.dump(model, "random_forest_model.pkl")` from a model fitted on a pandas DataFrame, so the feature names are stored.

## Moving toward a real deployment

The dashboard replays recorded data. In a factory, the feature rows would come from live systems:

- Sensors from PLCs or a data historian, via OPC UA, MQTT or a time-series database.
- Error logs from the machine controllers.
- Maintenance history from the CMMS or ERP.

An hourly job would aggregate each machine's data, compute the same rolling and maintenance features used in training (use one shared function for both, to avoid training-serving skew), score every machine and store the risk scores for the dashboard to read. The Batch upload tab is a manual stand-in for that step.

## Limitations

- Row-level metrics overstate how many separate failures were caught, because one failure produces several adjacent positive rows. Event-level measures (the share of failures with an alert beforehand, false alarms per machine per week, lead time) would be more useful operationally.
- Random Forest probabilities are not well calibrated. A threshold like 0.085 is a ranking cut-off, not a literal 8.5% chance of failure.
- With only a few hundred failures in the test period, metrics are noisy. Treat small differences between models or thresholds with caution.
- Pickle files can run code when loaded. Only load model files you trained or trust.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: random_forest_model.pkl` | Place the model file next to `app.py`, or change `MODEL_PATH` at the top of `app.py`. |
| Error about missing columns in `holdout.csv.gz` | Your model expects different features than the holdout file has. Regenerate the holdout with `train_model.py`, or retrain the model on the same columns. |
| Model fails to unpickle | Install the scikit-learn version used to train it. |
| Slow first load | Expected. Scoring is cached after the first run. |
| Font looks different offline | The IBM Plex Sans font loads from Google Fonts and falls back to a system sans-serif without internet. |
