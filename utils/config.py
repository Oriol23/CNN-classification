"""Stores important paths."""

import os

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"data")
MODELS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"models")
RUNS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"experiment_logs","runs")
RUNS_METADATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"experiment_logs",
                                       "runs_metadata")
METADATA_FILENAME = "Metadata.pkl"
RESULTS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"experiment_logs","results")
