#!/bin/sh

python -m coverage  run --source . run.py &&
    python -m coverage html
