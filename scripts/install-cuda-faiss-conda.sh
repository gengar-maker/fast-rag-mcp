#!/usr/bin/env bash
set -euo pipefail

# Recommended when you want FAISS GPU. The pip faiss-cpu wheel is CPU-only;
# FAISS GPU wheels are usually consumed from conda packages.
conda install -y -c pytorch -c nvidia faiss-gpu
