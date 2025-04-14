import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_log_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Log file not found at '{filepath}'")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise IOError(f"Error reading log file '{filepath}': {e}")


def parse_logs(log_content):
    """Parses the log string to extract model names and generation times."""
    data = []
    current_model = None
    # Regex to find the start of a model test block
    model_start_regex = re.compile(r"^--- Running generation test with (\S+) ---$")
    # Regex to find the generation time line
    time_regex = re.compile(r"^Generation finished in (\d+\.?\d*) seconds\.$")
    # Regex to extract the prompt (optional, but good for context)
    prompt_regex = re.compile(r"^\s+Prompt: '(.*)'$")
    current_prompt = "N/A"

    log_stream = io.StringIO(log_content) # Treat string as a file

    for line in log_stream:
        line = line.strip()

        # Check if a new model block starts
        model_match = model_start_regex.match(line)
        if model_match:
            current_model = model_match.group(1)
            # print(f"Found model block: {current_model}") # Debug print
            continue # Move to next line

        # Check for prompt line
        prompt_match = prompt_regex.match(line)
        if prompt_match:
            current_prompt = prompt_match.group(1)
            # print(f"Found prompt: {current_prompt}") # Debug print

        # If we are inside a model block, look for the time line
        if current_model:
            time_match = time_regex.match(line)
            if time_match:
                time_taken = float(time_match.group(1))
                data.append({"Model": current_model, "Time": time_taken, "Prompt": current_prompt})
                # print(f"Found time: {time_taken} for model {current_model}") # Debug print
                # Reset prompt after finding time for it
                current_prompt = "N/A"


    if not data:
         print("Warning: No generation time data found in logs.")
         return pd.DataFrame(columns=["Model", "Time", "Prompt"]) # Return empty DataFrame

    return pd.DataFrame(data)

def generate_stats_and_plots(df, output_dir="log_analysis_plots"):
    """Calculates statistics and generates plots from the parsed data."""
    if df.empty:
        print("DataFrame is empty, skipping statistics and plots.")
        return

    print("\n--- Generation Time Statistics ---")

    # Calculate statistics per model
    stats = df.groupby('Model')['Time'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
    stats = stats.round(3) # Round for readability

    print(stats.to_string(index=False)) # Print stats table

    # --- Create Plots ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nSaving plots to directory: '{output_dir}'")

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Order models for plotting (optional, based on typical performance or name)
    model_order = sorted(df['Model'].unique()) # Simple alphabetical order
    # Or define a custom order:
    # model_order = ["SD15", "SD21", "SD21Turbo", "SDXL", "SDXLTurbo", "SD3Medium"]


    # 1. Bar Plot of Average Generation Time
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Model', y='mean', data=stats, order=model_order, palette="viridis")
    plt.title('Average Generation Time per Model (Seconds)')
    plt.xlabel('Model')
    plt.ylabel('Average Time (s)')
    plt.xticks(rotation=45, ha='right')
    # Add text labels for average times
    for index, row in stats.iterrows():
        model_pos = model_order.index(row['Model']) # Get the position in the ordered list
        barplot.text(model_pos, row['mean'], f"{row['mean']:.2f}", color='black', ha="center", va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_generation_time_bar.png"))
    print(f"Saved: average_generation_time_bar.png")
    # plt.show() # Uncomment to display plot immediately

    # 2. Box Plot of Generation Time Distribution
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Model', y='Time', data=df, order=model_order, palette="coolwarm")
    plt.title('Distribution of Generation Times per Model (Seconds)')
    plt.xlabel('Model')
    plt.ylabel('Generation Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_time_distribution_boxplot.png"))
    print(f"Saved: generation_time_distribution_boxplot.png")
    # plt.show() # Uncomment to display plot immediately

    # 3. Optional: Violin Plot (shows distribution shape better)
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Model', y='Time', data=df, order=model_order, palette="plasma", inner="quartile") # inner="stick" or inner="point" are alternatives
    plt.title('Distribution Shape of Generation Times per Model (Seconds)')
    plt.xlabel('Model')
    plt.ylabel('Generation Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_time_distribution_violinplot.png"))
    print(f"Saved: generation_time_distribution_violinplot.png")
    # plt.show() # Uncomment to display plot immediately

    # Close all plot figures
    plt.close('all')

# --- Main Execution ---
if __name__ == "__main__":
    
    log_data = load_log_file('/ghome/c5mcv07/C5_G7_MCV/Week_5/Models_Exploration/out/73233.out')
    parsed_data = parse_logs(log_data)

    if not parsed_data.empty:
        generate_stats_and_plots(parsed_data)
    else:
        print("Could not extract data from logs.")