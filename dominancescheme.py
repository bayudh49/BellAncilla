import pandas as pd
import os

# Load the Excel file
file_path = "Compound_noise_teleportation_results.xlsx"
xlsx = pd.ExcelFile(file_path)

# Separate GHZ and Ancilla sheet names based on prefix
ghz_sheets = [sheet for sheet in xlsx.sheet_names if sheet.startswith("GHZ_")]
anc_sheets = [sheet for sheet in xlsx.sheet_names if sheet.startswith("Anc_")]

# Store final results in a list
results = []

# Loop through all GHZ and Ancilla sheet pairs
for ghz_sheet in ghz_sheets:
    # Get the matching Ancilla sheet (must have the same suffix)
    suffix = ghz_sheet.replace("GHZ_", "")
    anc_sheet = f"Anc_{suffix}"
    
    if anc_sheet not in anc_sheets:
        continue  # Skip if no corresponding Ancilla sheet

    # Read data
    df_ghz = xlsx.parse(ghz_sheet)
    df_anc = xlsx.parse(anc_sheet)

    # Normalize column names
    df_ghz.columns = df_ghz.columns.str.strip()
    df_anc.columns = df_anc.columns.str.strip()

    # Loop through all combinations of input-p1-p2 in GHZ
    for _, ghz_row in df_ghz.iterrows():
        try:
            input_state = ghz_row['Input']
            p1 = round(ghz_row['p1'], 6)
            p2 = round(ghz_row['p2'], 6)
            fid_ghz = ghz_row['Fidelity']
            print(f"Processing GHZ sheet: {ghz_sheet}, Input: {input_state}, p1: {p1}, p2: {p2}")
            
            # Filter ANC by 3 criteria
            matched = df_anc[
                (df_anc['Input'] == input_state) &
                (df_anc['p1'].round(6) == p1) &
                (df_anc['p2'].round(6) == p2)
            ]

            if not matched.empty:
                best_row = matched.loc[matched['Fidelity'].idxmax()]
                fid_anc = best_row['Fidelity']
                theta = best_row['theta']
                phi = best_row['phi']
                fid_diff = fid_ghz - fid_anc

                if fid_diff > 1e-3:
                    status = "GHZ Superior"
                elif fid_diff < -1e-3:
                    status = "Ancilla Superior"
                else:
                    status = "Neutral"

                results.append({
                    "Noise Pair": suffix,
                    "Input State": input_state,
                    "p1": p1,
                    "p2": p2,
                    "Fid_max_GHZ": fid_ghz,
                    "Fid_max_anc": fid_anc,
                    "theta_max": theta,
                    "phi_max": phi,
                    "Fid_diff": fid_diff,
                    "Status": status
                })
        except Exception as e:
            continue

# Convert to DataFrame and display
df_result = pd.DataFrame(results)
print(df_result)  # Display in terminal

# Or save to Excel/CSV file
df_result.to_excel("comparison_of_fidelity.xlsx", index=False)
