#!/bin/bash
# CML model build script — installs Python dependencies before the model container is built.
set -e
pip3 install --no-cache-dir -r requirements.txt
