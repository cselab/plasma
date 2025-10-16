#!/bin/sh

. ~/.venv/torax/bin/activate &&
    python -m coverage  run run.py &&
    python plot.py run.raw &&
    python -m coverage html &&
    python3 diff.py run.raw ~/run.raw
