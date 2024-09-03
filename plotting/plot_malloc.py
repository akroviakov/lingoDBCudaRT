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
    grouped_df = grouped_df[grouped_df['Num rows'] == max_num_rows]
    grouped_df = grouped_df[grouped_df['Selectivity'] == max_selectivity]
    min_time_df = grouped_df.loc[grouped_df.groupby('Locality Level')['Kernel Time (ms)'].idxmin()]

    kernel_types = min_time_df['Locality Level']
    malloc_count_buffer = min_time_df['Malloc Count (buffer)']
    malloc_count_vec = min_time_df['Malloc Count (Vec)']


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


# def plot_X_by_Subplots(df, X_colname, subplot_colname, lines_colname, line_color_colname, dict_filter, xlogbase, plot_dir, plotname):
#     # print(f"Plotting x = {X_colname}, lines = {lines_colname}, color = {line_color_colname}, filter = {dict_filter}, subplots by = {subplot_colname}")
#     selection_mask = True
#     for colname, val in dict_filter.items():
#         selection_mask &= (df[colname] == val)
#     df = df[selection_mask]
#     subplots = df[subplot_colname].unique()
#     num_plots = len(subplots)
#     num_cols = math.ceil(math.sqrt(num_plots))
#     num_rows = math.ceil(num_plots / num_cols)
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 10))
#     group_cols = set(['Kernel type', X_colname, subplot_colname, lines_colname, line_color_colname])
#     group_cols.update(dict_filter.keys())
#     print(list(group_cols))
#     df = df.groupby(list(group_cols))['Kernel Time'].mean().reset_index().dropna()
#     # with pd.option_context('display.max_columns', None):
#     #     with pd.option_context('display.max_rows', None):
#     #         print(df)
#     # exit(-1)
#     if(num_cols > 1):
#         axs = axs.flatten()
#     lines = df[lines_colname].unique()
#     line_types = {line: linestyle for line, linestyle in zip(lines, linestyles)}
#     line_color_values = df[line_color_colname].unique()
#     line_colors = {color: discrete_colors[i % len(discrete_colors)] for i, color in enumerate(line_color_values)}

#     for idx, subplot_filter_val in enumerate(subplots):
#         subset_df = df[(df[subplot_colname] == subplot_filter_val)]
#         ax = axs[idx] if num_cols > 1 or num_rows > 1 else axs
 
#         if lines_colname != line_color_colname:
#             for (line_color, line_type), group in subset_df.groupby([line_color_colname, lines_colname]):
#                 linestyle = line_types[line_type]
#                 color = line_colors[line_color]
#                 ax.plot(group[X_colname], group["Kernel Time"], marker='o', linestyle=linestyle, color=color, label=f'{line_color_colname} ({lines_colname})')
#         else:
#             for line_type, group in subset_df.groupby(lines_colname):
#                 linestyle = line_types[line_type]
#                 color = line_colors[line_type]
#                 ax.plot(group[X_colname], group["Kernel Time"], marker='o', linestyle=linestyle, color=color, label=f'{line_color_colname} ({lines_colname})')
        
#         ax.set_yscale('log')
#         ax.set_xscale('log', base=xlogbase)
#         ax.set_xlabel(X_colname)
#         ax.set_ylabel('Kernel Time (ms)')
#         ax.set_title(f'{subplot_colname}: {subplot_filter_val}')
#         ax.set_xticks(subset_df[X_colname].unique())
#         if(X_colname == "Num bytes"):
#             ax.set_xticklabels([bytes_to_human_readable(i) for i in subset_df[X_colname].unique()])
#         else:
#             ax.set_xticklabels(subset_df[X_colname].unique())

#         ax.grid(True)

#     if(line_color_colname == "Num bytes"):
#         sorted_labels = sorted(df[line_color_colname].unique(), key=lambda x: map_to_numeric(x))
#     else: 
#         sorted_labels = sorted(df[line_color_colname].unique())

#     color_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in discrete_colors[:len(line_color_values)]]
#     color_labels = [f'{line_color_colname}: {color}' for color in sorted_labels]
#     handles = color_handles
#     labels = color_labels
#     if lines_colname != line_color_colname:
#         linestyle_handles = [plt.Line2D([0], [0], color='black', linestyle=line_types[line]) for line in line_types]
#         linestyle_labels = [f'{lines_colname}: {line}' for line in line_types]
#         handles += linestyle_handles
#         labels += linestyle_labels

#     if(num_cols > 1 or num_rows > 1):
#         first_ax = axs[0]
#     else:
#         first_ax = axs
#     first_ax.legend(handles, labels, title="Legend", loc='upper right', bbox_to_anchor=(-0.15, 1))

#     fig.tight_layout()
#     os.makedirs(plot_dir, exist_ok=True)
#     formatted_parts=[]
#     for key, value in dict_filter.items():
#         formatted_parts.append(f'__{key.replace(" ", "_")}_{value}')
#     suffix = ''.join(formatted_parts)
#     fig.savefig(f'{plot_dir}{plotname}{suffix}.png', dpi=300)

# def plot_BEST_X_by_Subplots(df, X_colname, subplot_colname, lines_colname, line_color_colname, xlogbase, plot_dir, plotname):
#     # Find the combination of filter values that gives the lowest Kernel Time
#     group_cols = set([X_colname, subplot_colname, lines_colname, line_color_colname])
#     grouped_df = df.groupby(list(group_cols))['Kernel Time'].min().reset_index()
#     subplots = grouped_df[subplot_colname].unique()
#     num_plots = len(subplots)
#     num_cols = math.ceil(math.sqrt(num_plots))
#     num_rows = math.ceil(num_plots / num_cols)
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 10))
#     merged_df = pd.merge(grouped_df, df, on=list(group_cols) + ['Kernel Time'], how='inner')
#     # print(merged_df)
#     # Flatten axs if it's a 2D array, otherwise make it a list
#     if isinstance(axs, (list, np.ndarray)) and len(axs.shape) == 2:
#         axs = axs.flatten()
#     else:
#         axs = [axs]
    
#     lines = grouped_df[lines_colname].unique()
#     line_types = {line: linestyle for line, linestyle in zip(lines, linestyles)}
#     line_color_values = grouped_df[line_color_colname].unique()
#     line_colors = {color: discrete_colors[i % len(discrete_colors)] for i, color in enumerate(line_color_values)}
    
#     for idx, subplot_filter_val in enumerate(subplots):
#         subset_df = grouped_df[grouped_df[subplot_colname] == subplot_filter_val]
#         ax = axs[idx]
        
#         if lines_colname != line_color_colname:
#             for (line_color, line_type), group in subset_df.groupby([line_color_colname, lines_colname]):
#                 linestyle = line_types[line_type]
#                 color = line_colors[line_color]
#                 ax.plot(group[X_colname], group["Kernel Time"], marker='o', linestyle=linestyle, color=color, label=f'{line_color_colname} ({lines_colname})')
#         else:
#             for line_type, group in subset_df.groupby(lines_colname):
#                 linestyle = line_types[line_type]
#                 color = line_colors[line_type]
#                 ax.plot(group[X_colname], group["Kernel Time"], marker='o', linestyle=linestyle, color=color, label=f'{line_color_colname} ({lines_colname})')
        
#         ax.set_yscale('log')
#         ax.set_xscale('log', base=xlogbase)
#         ax.set_xlabel(X_colname)
#         ax.set_ylabel('Kernel Time (ms)')
#         ax.set_title(f'{subplot_colname}: {subplot_filter_val}')
#         ax.set_xticks(subset_df[X_colname].unique())
        
#         if X_colname == "Num bytes":
#             ax.set_xticklabels([bytes_to_human_readable(i) for i in subset_df[X_colname].unique()])
#         else:
#             ax.set_xticklabels(subset_df[X_colname].unique())
        
#         ax.grid(True)
    
#     # Hide any empty subplots
#     for i in range(num_plots, len(axs)):
#         fig.delaxes(axs[i])
    
#     if line_color_colname == "Num bytes":
#         sorted_labels = sorted(grouped_df[line_color_colname].unique(), key=lambda x: map_to_numeric(x))
#     else:
#         sorted_labels = sorted(grouped_df[line_color_colname].unique())
    
#     color_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in discrete_colors[:len(line_color_values)]]
#     color_labels = [f'{line_color_colname}: {color}' for color in sorted_labels]
#     handles = color_handles
#     labels = color_labels
    
#     if lines_colname != line_color_colname:
#         linestyle_handles = [plt.Line2D([0], [0], color='black', linestyle=line_types[line]) for line in line_types]
#         linestyle_labels = [f'{lines_colname}: {line}' for line in line_types]
#         handles += linestyle_handles
#         labels += linestyle_labels
    
#     first_ax = axs[0] if num_cols > 1 or num_rows > 1 else axs
#     first_ax.legend(handles, labels, title="Legend", loc='upper right', bbox_to_anchor=(-0.15, 1))
    
#     fig.tight_layout()
#     os.makedirs(plot_dir, exist_ok=True)
#     fig.savefig(f'{plot_dir}{plotname}_best_points.png', dpi=300)

# def plot_ALL_threads_by_init_buf_sizes(dataframe):
#     unique_num_cols = dataframe['Num cols'].unique()
#     unique_num_blocks = dataframe['Num Blocks'].unique()
#     for num_blocks in unique_num_blocks:
#         for num_cols in unique_num_cols:
#             plot_X_by_Subplots(dataframe, 
#                 X_colname="Num threads", 
#                 subplot_colname='Init buffer size',
#                 lines_colname="Kernel type", 
#                 line_color_colname="Num bytes",
#                 dict_filter={"Num cols": num_cols, "Num Blocks": num_blocks}, 
#                 xlogbase=2, 
#                 plot_dir=f"{CURR_DIR}/Plots/threads_by_init_buf_sizes/", 
#                 plotname="threads_by_init_buf_sizes")

# def plot_ALL_blocks_by_init_buf_sizes(dataframe):
#     unique_num_cols = dataframe['Num cols'].unique()
#     for num_cols in unique_num_cols:
#         plot_X_by_Subplots(dataframe, 
#             X_colname="Num Blocks", 
#             subplot_colname='Init buffer size',
#             lines_colname="Kernel type", 
#             line_color_colname="Num bytes",
#             dict_filter={"Num cols": num_cols, "Num threads": 256}, 
#             xlogbase=2, 
#             plot_dir=f"{CURR_DIR}/Plots/blocks_by_init_buf_sizes/", 
#             plotname="blocks_by_init_buf_sizes")
            
# def plot_ALL_cols_by_num_bytes(dataframe):
#     unique_num_blocks = dataframe['Num Blocks'].unique()
#     unique_init_sizes = dataframe['Init buffer size'].unique()

#     for num_blocks in unique_num_blocks:
#         for init_size in unique_init_sizes:
#             plot_X_by_Subplots(dataframe, 
#                 X_colname="Num cols", 
#                 subplot_colname='Num bytes',
#                 lines_colname="Kernel type", 
#                 line_color_colname="Kernel type",
#                 dict_filter={"Init buffer size": init_size, "Num Blocks": num_blocks, "Num threads": 256}, 
#                 xlogbase=10, 
#                 plot_dir=f"{CURR_DIR}/Plots/cols_by_num_bytes/", 
#                 plotname="cols_by_num_bytes")

# def plot_ALL_bytes_by_num_cols(dataframe):
#     dataframe['Num bytes'] = dataframe['Num bytes'].apply(map_to_numeric).astype(int)
#     plot_BEST_X_by_Subplots(dataframe, 
#                 X_colname="Num bytes", 
#                 subplot_colname='Num cols',
#                 lines_colname="Kernel type", 
#                 line_color_colname="Kernel type",
#                 xlogbase=10, 
#                 plot_dir=f"{CURR_DIR}/Plots/size_by_cols/", 
#                 plotname="size_by_cols")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV data file')
    args = parser.parse_args()

    plot_time_vs_selectivity_per_data_size(args.csv_path, "Plots")
    plot_barplot_malloc_count(args.csv_path, "Plots")
    plot_time_vs_data_size_per_selectivity(args.csv_path, "Plots")
    plot_barplot_bytes_with_baseline(args.csv_path, "Plots")
    plot_allocated_vs_true_result_bytes(args.csv_path, "Plots")

    # df = pd.read_csv(args.csv_path)
    # df['Num bytes'] = df['Num bytes'].apply(bytes_to_human_readable)
    # df['Num threads'] = df['Num threads'].astype('category')
    # # plot_ALL_cols_by_num_bytes(df.copy())
    # plot_ALL_bytes_by_num_cols(df.copy())
    # plot_ALL_threads_by_init_buf_sizes(df.copy())
    # plot_ALL_blocks_by_init_buf_sizes(df.copy())
