#!/bin/bash

python3 src/bok_minutes_download.py &&
python3 src/bok_minutes_preprocessing.py &&
python3 src/bok_minutes_tone_analytics.py  # &&
# python3 src/bok_minutes_send_analytics.py
