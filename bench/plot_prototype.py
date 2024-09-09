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

metricToSemantics = {
    # "lts__t_sectors.avg.pct_of_peak_sustained_elapsed" : "L2 requests (of peak)",
    # "lts__t_sectors_lookup_hit.sum" : "L2 hits",
    # "lts__t_sectors_lookup_miss.sum" : "L2 misses",
    # "lts__t_sector_hit_rate.pct" : "L2 hit rate",
    # "lts__t_sectors_srcunit_tex_op_read.sum.per_second" : "L2->L1 sectors (per second)",

    # "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum.pct_of_peak_sustained_elapsed" : "L2->L1 bandwidth(of peak)",
    # "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum" : "L2->L1 sectors",
    # "l1tex__lsu_writeback_active_mem_lg.sum.pct_of_peak_sustained_elapsed" : "L1 utilization (of peak)",
    # "l1tex__t_sector_hit_rate.pct" : "L1 hit rate",
    # "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" : "L1 sectors loaded",
    # "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" : "L1 load requests",
    # "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum" : "L1 sectors written",
    # "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum" : "L1 store requests",
    # "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum" : "Num. warps hit L1",

    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed" : "Cycles with work",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" : "Global memory stalls pct",
    "smsp__warps_issue_stalled_long_scoreboard.avg" : "Global memory stalls", #total
    "smsp__average_warp_latency_per_inst_issued.ratio" : "Instruction latency",
    "smsp__warps_eligible.avg.per_cycle_active" : "Eligible warps per cycle",
    "smsp__inst_executed.sum" : "Executed instructions",
    "smsp__warps_issue_stalled_lg_throttle.avg" : "LSU throttle stalls",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct" : "LSU throttle stalls pct",

    "smsp__warps_launched.sum" : "Num. launched warps",

    "dram__bytes_read.sum.per_second" : "Read throughput",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed" : "Read throughput of peak",
    "dram__bytes.sum" : "Total DRAM traffic",

    "gpu__time_duration.sum" : "Kernel duration"
}

hatch_patterns = ['\\', '-', '+', 'x', 'o', 'O', '.', '*', '//', '||']

def convertMetrics(metric_name):
    found = metricToSemantics.get(metric_name)
    # print(f"{metric_name}  - > {found}")
    return metric_name if found == None else found 

def extract_kernel_name(name):
    fill = name.find("(FillVariant")
    if(fill != -1):
        return name[:fill]
    end = name.find('(') 
    return name[:end]

def convert_metric_value(value):
    if not isinstance(value, str):
        return value
    value = value.replace('.', '')
    value = value.replace(',', '.')
    return pd.to_numeric(value, errors='coerce')


def readPreprocess(file_path):
    df = pd.read_csv(file_path)
    df['Metric Name'] = df['Metric Name'].apply(convertMetrics)
    df['Metric Value'] = df['Metric Value'].apply(convert_metric_value)
    df['Kernel Name'] = df['Kernel Name'].apply(extract_kernel_name)
    df['Metric Unit'].fillna('ratio', inplace=True)
    pivot_df = df.pivot_table(
        index=['Kernel Name'],
        columns=['Metric Name', 'Metric Unit'],
        values='Metric Value',
        aggfunc='median'
    )
    mask = ~pivot_df.index.get_level_values('Kernel Name').str.contains('gallatin::', case=False, na=False)
    mask = mask & ~pivot_df.index.get_level_values('Kernel Name').str.contains('print', case=False, na=False)
    pivot_df = pivot_df[mask]
    return pivot_df

def reduceGBUnits(dataframe, column_name):
    conversion_map = {
        'Gbyte': 1024**3, 
        'Mbyte': 1024**2,  
        'Kbyte': 1024   
    }
    for metric_unit in conversion_map:
        column = (column_name, metric_unit)
        if column in dataframe.columns:
            dataframe[(column_name, 'Byte')] = dataframe.get((column_name, 'Byte'), 0) + dataframe[column].fillna(0) * conversion_map[metric_unit]
    columns_to_drop = [((column_name, metric)) for metric in conversion_map.keys() if (column_name, metric) in dataframe.columns]
    dataframe = dataframe.drop(columns=columns_to_drop)
    dataframe = dataframe[[col for col in [(column_name, 'Byte')] + [c for c in dataframe.columns if c != (column_name, 'Byte')]]]
    return dataframe

def reduceGBPerSUnits(dataframe, column_name):
    conversion_map = {
        'Gbyte/second': 1024**3, 
        'Mbyte/second': 1024**2,  
        'Kbyte/second': 1024   
    }
    for metric_unit in conversion_map:
        column = (column_name, metric_unit)
        if column in dataframe.columns:
            dataframe[(column_name, 'Byte/s')] = dataframe.get((column_name, 'Byte/s'), 0) + dataframe[column].fillna(0) * conversion_map[metric_unit]
    columns_to_drop = [((column_name, metric)) for metric in conversion_map.keys() if (column_name, metric) in dataframe.columns]
    dataframe = dataframe.drop(columns=columns_to_drop)
    dataframe = dataframe[[col for col in [(column_name, 'Byte/s')] + [c for c in dataframe.columns if c != (column_name, 'Byte/s')]]]
    return dataframe

def reduceTimeUnits(dataframe, column_name):
    conversion_map = {
        'usecond': 1 / 1000
    }
    dataframe[(column_name, 'msecond')] = dataframe[(column_name, 'msecond')].fillna(0) 
    for metric_unit in conversion_map:
        column = (column_name, metric_unit)
        if column in dataframe.columns:
            dataframe[(column_name, 'msecond')] += dataframe[column].fillna(0) * conversion_map[metric_unit]
    columns_to_drop = [(column_name, metric) for metric in conversion_map if (column_name, metric) in dataframe.columns]
    dataframe = dataframe.drop(columns=columns_to_drop)
    dataframe = dataframe[[col for col in [(column_name, 'msecond')] + [c for c in dataframe.columns if c != (column_name, 'msecond')]]]
    return dataframe

def plot_prototype(file_path, plot_dir, SF=1):
    df = readPreprocess(file_path)
    df = df[["Kernel duration"]]
    new_order = [
        'void growingBufferFillTB<(TABLE)0,', 
        'void growingBufferFillTB<(TABLE)1,', 
        'void growingBufferFillTB<(TABLE)2,', 
        'void growingBufferFillTB<(TABLE)3,', 
        'buildHashIndexedViewSimple', 
        'buildPreAggregationHashtableFragments', 
        'mergePreAggregationHashtableFragments', 
        'freeFragments', 
        'freeKernel'
    ]
    df = df.reindex(new_order)
    df= reduceTimeUnits(df, "Kernel duration")
    df.loc['buildHashIndexedViewSimple', ('Kernel duration', 'msecond')] *= 4

    fig, ax = plt.subplots(figsize=(10, 3))
    colors = plt.cm.tab10(range(len(df)))

    left = 0  
    for i, (value, name) in enumerate(zip(df[('Kernel duration', 'msecond')], df.index)):
        ax.barh(0, value, left=left, color=colors[i], edgecolor='black', label=name)[0].set_hatch(hatch_patterns[i])  
        left += value  
    ax.set_xlim(0, left)
    ax.set_yticks([])
    ax.set_title('Device execution timeline (ms)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=3, title="ShortName")

    fig.tight_layout()

    fig.savefig(f'{plot_dir}/Q41_timeline.png', dpi=300)
    fig.savefig(f"{plot_dir}/Q41_timeline.pdf", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True, help='Path to csv data file')
    args = parser.parse_args()
    plot_dir = "Plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_prototype(args.csv_path, plot_dir)

