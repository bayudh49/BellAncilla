import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = r"Compound_noise_teleportation_results.xlsx"  # Change if located in a different folder
xls = pd.ExcelFile(file_path)

# Separate GHZ and Ancilla sheets
ghz_sheets = [s for s in xls.sheet_names if s.lower().startswith('ghz')]
anc_sheets = [s for s in xls.sheet_names if s.lower().startswith('anc')]

# Function to compute absolute robustness metrics
def compute_absolute_robustness(xls, sheets, scheme_name):
    records = []
    for sheet in sheets:
        df = xls.parse(sheet)
        if 'Fidelity' not in df.columns or 'Input' not in df.columns:
            continue
        grouped = df.groupby('Input')['Fidelity']
        summary = grouped.agg(
            Min_Fidelity='min',
            Mean_Fidelity='mean',
            Std_Fidelity='std',
            Above_Threshold=lambda x: (x >= 0.9).sum(),
            Total='count'
        ).reset_index()
        summary['Noise Pair'] = sheet.split('_', 1)[-1]
        summary['Threshold_Robustness'] = summary['Above_Threshold'] / summary['Total']
        summary['Scheme'] = scheme_name
        records.append(summary)
    return pd.concat(records, ignore_index=True)

# Compute for GHZ and Ancilla
ghz_result = compute_absolute_robustness(xls, ghz_sheets, 'GHZ')
anc_result = compute_absolute_robustness(xls, anc_sheets, 'Ancilla')
robust_comparison = pd.concat([ghz_result, anc_result], ignore_index=True)

# Average metrics per Noise Pair and Scheme
summary_metrics = robust_comparison.groupby(['Scheme', 'Noise Pair']).agg(
    Avg_Fidelity=('Mean_Fidelity', 'mean'),
    Min_Fidelity=('Min_Fidelity', 'mean'),
    Std_Fidelity=('Std_Fidelity', 'mean'),
    Threshold_Robustness=('Threshold_Robustness', 'mean')
).reset_index()

# Visualization: bar chart for each metric
metrics_info = {
    'Avg_Fidelity': 'Average Fidelity',
    'Min_Fidelity': 'Minimum Fidelity',
    'Std_Fidelity': 'Fidelity Standard Deviation',
    'Threshold_Robustness': 'Proportion of High Fidelity (F ≥ 0.9)'
}

for metric, label in metrics_info.items():
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_metrics, x='Noise Pair', y=metric, hue='Scheme')
    plt.xticks(rotation=90)
    plt.title(f'{label} by Scheme and Noise Pair')
    plt.ylabel(label)
    plt.xlabel('Noise Pair')
    plt.tight_layout()
    plt.savefig(f'bar_{metric}.png', dpi=300)
    plt.close()
