import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import argparse

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
byte_order = ['bytes', 'KB', 'MB', 'GB']
linestyles = ['-', '--', ':', '-.'] 
discrete_colors = ['blue', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'] # 'red'
hatch_patterns = ['/', '+', 'o', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

def bytes_to_human_readable(num_bytes):
    for unit in byte_order:
        if num_bytes < 1000:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1000

def map_to_numeric(item):
    value, unit = item.split()
    multiplier = 1
    # not KiB
    if unit == 'KB':
        multiplier = 1000
    elif unit == 'MB':
        multiplier = 1000 * 1000
    elif unit == 'GB':
        multiplier = 1000 * 1000 * 1000
    return int(value) * multiplier

def plot_time_vs_selectivity_per_data_size(filename, plot_dir):
    df = pd.read_csv(filename)
    grouped_df = df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity', 'Grid Size', 'Block Size']).mean().reset_index()
    minimal_time_df = grouped_df.loc[grouped_df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity'])['Kernel Time (ms)'].idxmin()]
    unique_num_rows = minimal_time_df['Num rows'].unique()
    
    fig, axes = plt.subplots(len(unique_num_rows), 1, figsize=(6, 3 * len(unique_num_rows)), sharex=True)
    if len(unique_num_rows) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    color_map = {version: discrete_colors[i % len(discrete_colors)] for i, version in enumerate(minimal_time_df['Version'].unique())}
    style_map = {locality: linestyles[i % len(linestyles)] for i, locality in enumerate(minimal_time_df['Locality Level'].unique())}
    
    for i, num_rows in enumerate(unique_num_rows):
        ax = axes[i]
        subset = minimal_time_df[minimal_time_df['Num rows'] == num_rows]
        
        prefix_sum_df = subset[subset['Locality Level'] == 'PrefixSum']
        if not prefix_sum_df.empty:
            ax.plot(prefix_sum_df['Selectivity'], prefix_sum_df['Kernel Time (ms)'], 
                    color='red', linestyle='-', marker='o', 
                    label='PrefixSum', linewidth=2)

        for (version, locality), group in subset.groupby(['Version', 'Locality Level']):
            if locality == 'PrefixSum':
                continue  # Skip PrefixSum since it's already plotted
            color = color_map[version]
            style = style_map[locality]
            ax.plot(group['Selectivity'], group['Kernel Time (ms)'], 
                    marker='X', color=color, linestyle=style, 
                    label=f'{version}::{locality}')

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.set_title(f'Num rows: {num_rows}')
        ax.set_ylabel('Kernel Time (ms)')
        ax.set_xlabel('Selectivity')
        ax.set_yscale('log')
        ax.grid(True)
    
    fig.tight_layout()  # Adjust the right margin to make space for the legend
    fig.savefig(f'{plot_dir}/time_vs_selectivity_per_data_size.png', dpi=300)
    fig.savefig(f"{plot_dir}/time_vs_selectivity_per_data_size.pdf", dpi=300)

def plot_time_vs_data_size_per_selectivity(filename, plot_dir):
    df = pd.read_csv(filename)
    grouped_df = df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity', 'Grid Size', 'Block Size']).mean().reset_index()
    minimal_time_df = grouped_df.loc[grouped_df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity'])['Kernel Time (ms)'].idxmin()]
    unique_selectivities = minimal_time_df['Selectivity'].unique()
    
    fig, axes = plt.subplots(len(unique_selectivities), 1, figsize=(12, 6 * len(unique_selectivities)), sharex=True)
    if len(unique_selectivities) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    color_map = {version: discrete_colors[i % len(discrete_colors)] for i, version in enumerate(minimal_time_df['Version'].unique())}
    style_map = {locality: linestyles[i % len(linestyles)] for i, locality in enumerate(minimal_time_df['Locality Level'].unique())}
    for i, selectivity in enumerate(unique_selectivities):
        ax = axes[i]
        subset = minimal_time_df[minimal_time_df['Selectivity'] == selectivity]
        
        prefix_sum_df = subset[subset['Locality Level'] == 'PrefixSum']
        if not prefix_sum_df.empty:
            ax.plot(prefix_sum_df['Num rows'], prefix_sum_df['Kernel Time (ms)'], 
                    color='red', linestyle='-', marker='', 
                    label='PrefixSum', linewidth=2)
        
        for (version, locality), group in subset.groupby(['Version', 'Locality Level']):
            if locality == 'PrefixSum':
                continue  # Skip PrefixSum since it's already plotted
            color = color_map[version]
            style = style_map[locality]
            ax.plot(group['Num rows'], group['Kernel Time (ms)'], 
                    marker='X', color=color, linestyle=style, 
                    label=f'{version}::{locality}')
        
        ax.set_title(f'Selectivity: {selectivity}')
        ax.set_ylabel('Kernel Time (ms)')
        ax.set_xlabel('Num rows')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Finalize the layout and save the plot
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/time_vs_data_size_per_selectivity.png', dpi=300)
    fig.savefig(f"{plot_dir}/time_vs_data_size_per_selectivity.pdf", dpi=300)

def plot_barplot_bytes_with_baseline(filename, plot_dir):
    df = pd.read_csv(filename)
    
    grouped_df = df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity', 'Grid Size', 'Block Size']).mean().reset_index()
    grouped_df = grouped_df[(grouped_df["Version"] == "Baseline") & (grouped_df['Locality Level'] != "PrefixSum")]

    max_num_rows = grouped_df['Num rows'].max()
    max_selectivity = grouped_df['Selectivity'].max()
    grouped_df = grouped_df[(grouped_df['Num rows'] == max_num_rows) & (grouped_df['Selectivity'] == max_selectivity)]
    min_time_df = grouped_df.loc[grouped_df.groupby('Locality Level')['Kernel Time (ms)'].idxmin()]

    kernel_types = min_time_df['Locality Level']
    allocated_bytes = min_time_df['AllocatedResultBytes']
    true_bytes = min_time_df['TrueResultBytes'].mean()  # Assuming the same value for 'TrueResultBytes' across all rows
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(kernel_types, allocated_bytes, color='steelblue', edgecolor='black', label='Allocated Result Bytes')
    ax.axhline(y=true_bytes, color='red', linestyle='--', label='True Result Bytes (Ground Truth)')
    ax.set_ylabel('Result Bytes')
    ax.legend()
    ax.set_xticks(kernel_types)
    ax.set_xticklabels(kernel_types, ha='right')

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/barplot_result_bytes.png", dpi=300)
    fig.savefig(f"{plot_dir}/barplot_result_bytes.pdf", dpi=300)


def plot_barplot_malloc_count(filename, plot_dir):
    df = pd.read_csv(filename)
    grouped_df = df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity', 'Grid Size', 'Block Size']).mean().reset_index()
    grouped_df = grouped_df[grouped_df["Version"] == "Baseline"]
    grouped_df = grouped_df[grouped_df['Locality Level'] != "PrefixSum"]

    max_num_rows = grouped_df['Num rows'].max()
    max_selectivity = grouped_df['Selectivity'].max()
    max_blk = grouped_df['Block Size'].max()
    max_grid = grouped_df['Grid Size'].max()
    grouped_df = grouped_df[grouped_df['Num rows'] == max_num_rows]
    grouped_df = grouped_df[grouped_df['Selectivity'] == max_selectivity]
    grouped_df = grouped_df[grouped_df['Block Size'] == max_blk]
    grouped_df = grouped_df[grouped_df['Grid Size'] == max_grid]
    print(grouped_df)
    kernel_types = grouped_df['Locality Level']
    malloc_count_buffer = grouped_df['Malloc Count (buffer)']
    malloc_count_vec = grouped_df['Malloc Count (Vec)']


    fig, ax = plt.subplots(figsize=(5, 3))
    # ax.set_yscale('log')
    ax.bar(kernel_types, malloc_count_buffer, 
                   color="red", hatch=hatch_patterns[0], 
                   edgecolor='black',
                   label='Malloc Count (buffer)')
    ax.bar(kernel_types, malloc_count_vec, 
                   color="steelblue", hatch=hatch_patterns[1], 
                   edgecolor='black',
                   bottom=malloc_count_buffer, 
                   label='Malloc Count (Vec)')
    
    ax.set_ylabel('# of malloc() calls')
    # ax.set_xlabel('Kernel type')
    # ax.set_title('Malloc calls for different kernel versions')
    ax.legend()
    ax.set_xticks(kernel_types)
    ax.set_xticklabels(kernel_types, ha='center')
    
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/barplot_malloc_count.png", dpi=300)
    fig.savefig(f"{plot_dir}/barplot_malloc_count.pdf", dpi=300)


def plot_allocated_vs_true_result_bytes(filename, plot_dir):
    df = pd.read_csv(filename)

    grouped_df = df.groupby(['Version', 'Locality Level', 'Num rows', 'Selectivity', 'Grid Size', 'Block Size']).mean().reset_index()
    grouped_df = grouped_df[grouped_df['Locality Level'] != "PrefixSum"]
    
    max_selectivity = grouped_df['Selectivity'].max()
    max_selectivity_df = grouped_df[grouped_df['Selectivity'] == max_selectivity]

    baseline_df = max_selectivity_df[max_selectivity_df['Version'] == 'Baseline']
    min_time_df = baseline_df.loc[baseline_df.groupby(['Locality Level', 'Num rows'])['Kernel Time (ms)'].idxmin()]
    localities = min_time_df['Locality Level'].unique()
    color_map = {locality: discrete_colors[i % len(discrete_colors)] for i, locality in enumerate(localities)}
    linestyle_map = {locality: linestyles[i % len(linestyles)] for i, locality in enumerate(localities)}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)
    true_result_df = baseline_df[['Num rows', 'TrueResultBytes']].drop_duplicates()
    ax.plot(true_result_df['Num rows'], true_result_df['TrueResultBytes'], 
            color='red', linestyle='-', marker='', 
            label='True Result Bytes')

    for (version, locality), group in min_time_df.groupby(['Version', 'Locality Level']):
        color = color_map[locality]
        linestyle = linestyle_map[locality]
        ax.plot(group['Num rows'], group['AllocatedResultBytes'], 
                marker='X', color=color, linestyle=linestyle, 
                label=f'Locality: {locality}')
    
    # ax.set_title('')
    ax.set_xlabel('Num Rows')
    ax.set_ylabel('Result Bytes')
    ax.legend()


    fig.tight_layout()
    fig.savefig(f'{plot_dir}/allocated_vs_true_result_bytes.png', dpi=300)
    fig.savefig(f"{plot_dir}/allocated_vs_true_result_bytes.pdf", dpi=300)


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV data file')
    args = parser.parse_args()
    plot_dir = "Plots"
    os.makedirs(plot_dir, exist_ok=True)

    plot_time_vs_selectivity_per_data_size(args.csv_path, plot_dir)
    plot_barplot_malloc_count(args.csv_path, plot_dir)
    plot_time_vs_data_size_per_selectivity(args.csv_path, plot_dir)
    plot_barplot_bytes_with_baseline(args.csv_path, plot_dir)
    plot_allocated_vs_true_result_bytes(args.csv_path, plot_dir)
