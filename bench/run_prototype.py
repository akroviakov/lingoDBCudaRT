import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import argparse
import subprocess
import sys
import shutil
import argparse
import re
import glob
import plot_prototype

def run_command(command, cwd=None, stdout=None):
    """Run a shell command and capture its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, stdout=stdout)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to SSB data set')
    parser.add_argument('--sf', type=str, required=True, help='SSB scale factor')
    args = parser.parse_args()

    data_dir = "Data/Prototype"
    os.makedirs(data_dir, exist_ok=True)
    plot_dir = "Plots"
    os.makedirs(plot_dir, exist_ok=True)
    metrics_output_file = os.path.join(data_dir, f"metrics_SF_{args.sf}")
    if os.path.exists(metrics_output_file):
        os.remove(metrics_output_file)

    metrics_string = ",".join(plot_prototype.metricToSemantics.keys())
    run_command("make Bench -C ../build")
    ncu_command = (
        f"ncu --metrics {metrics_string} -f --export {metrics_output_file} ../build/lingodbDSA/Bench {args.dataset_path} {args.sf}"
    )
    run_command(ncu_command)
    run_command(f"ncu --import {metrics_output_file}.ncu-rep --csv > {metrics_output_file}.csv")
    os.remove(f"{metrics_output_file}.ncu-rep")
    plot_prototype.plot_prototype(f"{metrics_output_file}.csv", plot_dir)