"""Validate and persist a curated source table."""

import argparse

from protein_state_router.config import load_config
from protein_state_router.data.catalog import build_catalog

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
build_catalog(config["input_path"], config["output_catalog"])
