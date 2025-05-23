import os
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re

def sanitize_filename(text):
    # Remove illegal characters for filenames
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    
    # Manually map input state to safe filename strings
    text = (
        text.replace('|0>', 'ket0')
            .replace('|1>', 'ket1')
            .replace('|+>', 'ketplus')
            .replace('|->', 'ketminus')
            .replace('|+i>', 'ketplusi')
            .replace('|-i>', 'ketminusi')
            .replace('⟩', '')  # if there is a unicode ket symbol
            .replace('−', '-')  # convert unicode minus to ASCII dash
    )
    
    text = (
        text.replace('0', 'ket0')
            .replace('1', 'ket1')
            .replace('+', 'ketplus')
            .replace('-', 'ketminus')
            .replace('+i', 'ketplusi')
            .replace('-i', 'ketminusi')
            .replace('⟩', '')  # if there is a unicode ket symbol
            .replace('−', '-')  # convert unicode minus to ASCII dash
    )

    return text

# Re-uploaded file assumed
file_path = 'comparison_of_fidelity.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Create output directory
output_dir = 'piecharts_per_noise_input_scheme'
os.makedirs(output_dir, exist_ok=True)

# Define colors
colors = {'Anc Unggul': 'green', 'GHZ Unggul': 'red', 'Netral': 'gold'}

# Get unique noise pairs and input states
noise_pairs = df['Noise Pair'].unique()
input_states = df['Input State'].unique()

# Generate pie chart for each noise pair and input state
output_files = []
label_map = {
    'Anc Unggul': 'Anc',
    'GHZ Unggul': 'GHZ',
    'Netral': 'Neutral'
}

for noise in noise_pairs:
    for input_state in input_states:
        subset = df[(df['Noise Pair'] == noise) & (df['Input State'] == input_state)]
        status_counts = subset['Status'].value_counts().reindex(['Anc Unggul', 'GHZ Unggul', 'Netral']).fillna(0)

        if status_counts.sum() == 0:
            continue

        # Take only labels with counts > 0
        nonzero_status_counts = status_counts[status_counts > 0]
        nonzero_labels = [label_map.get(k, k) for k in nonzero_status_counts.index]
        nonzero_colors = [colors[k] for k in nonzero_status_counts.index]

        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            nonzero_status_counts,
            labels=nonzero_labels,
            autopct='%1.0f%%',
            colors=nonzero_colors,
            startangle=90,
            textprops={'fontsize': 15}
        )

        ax.axis('equal')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        safe_noise = sanitize_filename(noise.replace(" ", ""))
        safe_input = sanitize_filename(input_state)
        filename = f'{output_dir}/pie_{safe_noise}_{safe_input}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        output_files.append(filename)

output_files[:10]  # Preview first 10 generated files
