"""Stores important paths."""

import os

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"data")
RUNS_DIRECTORY = os.path.join(MAIN_DIRECTORY,"runs")
RUNS_METADATA_DIRECTORY = os.path.join(MAIN_DIRECTORY,"runs_metadata")
METADATA_FILENAME = "Metadata.pkl"