"""Stores important paths."""

import os

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"data")
MODELS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"models")
RUNS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"experiment_logs","runs")
RESULTS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"experiment_logs","results")

METADATA_FILENAME = "Metadata.pkl"      #must be .pkl
RESULTS_FILENAME = "Results.feather"    #must be .feather

print(f"[INFO] Main directory {MAIN_DIRECTORY}")