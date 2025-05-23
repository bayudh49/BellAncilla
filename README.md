# BellAncilla
This repository contains the dataset and all source code used in the research on {Ancilla-Assisted Quantum Teleportation under Compound Hybrid Noise Model---Comparative Analysis with Bell and GHZ Protocols. It supports the results and simulations presented in our study. Supplementary materials are also provided for reproducibility.

The steps we followed in our data analysis process are as follows:

1. Numerical Simulation: We performed numerical simulations using numericalsimulation.py, which generated the output file Compound_noise_teleportation_results.xlsx.

2. Symbolic Verification: We cross-checked the symbolic mathematical derivation using Python with the script symbolicsimulation.py. The symbolic outputs are stored in the outputlatex folder.

3. Dominance Statistics: We analyzed statistical dominance between the GHZ and Ancilla schemes using dominanceshceme.py, producing the file comparison_of_fidelity.xlsx.

4. Visualization by Input and Noise: We visualized the dominance of each scheme per input state and per noise pair using the script piechart.py, with the output available in the folder piecharts_per_noise_input_scheme.

5. Robustness Histogram: We visualized the average fidelity, minimum fidelity, standard deviation, and proportion of high fidelity (F ≥ 0.9) using the script histogram.py.

These constitute the complete data processing steps used in this research.


