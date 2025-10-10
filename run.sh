#!/bin/sh

python -m coverage  run run.py &&
    python -m coverage html
