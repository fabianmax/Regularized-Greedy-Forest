import json
import pandas as pd
import numpy as np

# Path to model json files
model_path_rgf = "../models/2018-05-18_00-49-00_rgf_model.json"
model_path_xgb = "../models/2018-05-18_18-39-01_xgb_model.json"

# Open model results
with open(model_path_rgf) as infile:
    mod_rgf = json.load(infile)

with open(model_path_xgb) as infile:
    mod_xgb = json.load(infile)

# Extract infos
results_xgb = pd.DataFrame({
    "model": np.repeat("XGBoost", len(mod_xgb)),
    "dataset": [l["test"][13:] for l in mod_xgb],
    "test_score": [l["test_score"] for l in mod_xgb],
    "task": [l["task"] for l in mod_xgb],
    "time": [l["time"] for l in mod_xgb]
})

results_rgf = pd.DataFrame({
    "model": np.repeat("RGF", len(mod_rgf)),
    "dataset": [l["test"][13:] for l in mod_rgf],
    "test_score": [l["test_score"] for l in mod_rgf],
    "task": [l["task"] for l in mod_rgf],
    "time": [l["time"] for l in mod_rgf]
})

# And combine
results = pd.concat([results_xgb, results_rgf]).sort_values(by="dataset")





