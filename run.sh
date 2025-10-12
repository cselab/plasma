#!/bin/sh

. ~/.venv/torax/bin/activate &&
python -m coverage  run run.py &&
python -m coverage html &&
diff run.raw ~/run.raw
