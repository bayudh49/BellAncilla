import numpy as np
import pandas as pd
from qutip import *
from itertools import combinations
from pathlib import Path

# === Operator Kraus Noise ===
def kraus_bitflip(p): return [np.sqrt(1 - p) * qeye(2), np.sqrt(p) * sigmax()]
def kraus_phaseflip(p): return [np.sqrt(1 - p) * qeye(2), np.sqrt(p) * sigmaz()]
def kraus_bitphaseflip(p): return [np.sqrt(1 - p) * qeye(2), np.sqrt(p) * sigmay()]
def kraus_depolarizing(p):
    return [np.sqrt(1 - p) * qeye(2),
            np.sqrt(p/3) * sigmax(),
            np.sqrt(p/3) * sigmay(),
            np.sqrt(p/3) * sigmaz()]
def kraus_amplitude_damping(p):
    return [Qobj([[1, 0], [0, np.sqrt(1 - p)]]),
            Qobj([[0, np.sqrt(p)], [0, 0]])]
def kraus_phase_damping(p):
    return [Qobj([[1, 0], [0, np.sqrt(1 - p)]]),
            Qobj([[0, 0], [0, np.sqrt(p)]])]

noise_kraus_map = {
    "BitFlip": kraus_bitflip,
    "PhaseFlip": kraus_phaseflip,
    "BitPhaseFlip": kraus_bitphaseflip,
    "Depolarizing": kraus_depolarizing,
    "AmplitudeDamping": kraus_amplitude_damping,
    "PhaseDamping": kraus_phase_damping
}
# 3. Apply hybrid noise only to channel (qubit 1 and 2)
def apply_kraus_to_qubit(rho, kraus_ops, target_idx, N):
        out = Qobj(np.zeros(rho.shape), dims=rho.dims)
        for K in kraus_ops:
            ops = [qeye(2)] * N
            ops[target_idx] = K
            E = tensor(ops)
            out += E * rho * E.dag()
        return out

def hybrid_noise_general(rho, type1, p1, type2, p2):
    E1_list = noise_kraus_map[type1](p1)
    E2_list = noise_kraus_map[type2](p2)
    out = Qobj(np.zeros(rho.shape), dims=rho.dims)
    for E1 in E1_list:
        for E2 in E2_list:
            out += E1 * E2 * rho * E2.dag() * E1.dag()
    return out
def pauli_correction_by_bell_and_charlie(bell_index, charlie_z):
    """
    Mapping hasil pengukuran Bell (0–3) dan hasil pengukuran Charlie (0/1)
    ke koreksi Pauli yang harus diterapkan di Bob.
    """
    I2 = qeye(2)
    X  = sigmax()
    Y  = sigmay()
    Z  = sigmaz()

    mapping = {
        # (Bell outcome, Charlie result) : Correction
        (0, 0): I2,  # Φ+
        (0, 1): Z,   # Φ+ & z=1
        (1, 0): Z,   # Φ−
        (1, 1): I2,  # Φ− & z=1
        (2, 0): X,   # Ψ+
        (2, 1): Y,   # Ψ+ & z=1
        (3, 0): Y,   # Ψ−
        (3, 1): X    # Ψ− & z=1
    }

    return mapping.get((bell_index, charlie_z), I2)


def ghz_pauli_correction(bell_index, bob_result):
    I2 = qeye(2)
    X = sigmax()
    Y = sigmay()
    Z = sigmaz()

    table = {
        (0, 0): I2,
        (0, 1): Z,
        (1, 0): Z,
        (1, 1): I2,
        (2, 0): X,
        (2, 1): Y,
        (3, 0): Y,
        (3, 1): X,
    }

    # fallback if bob_result isn't valid
    if (bell_index, bob_result) not in table:
        return I2  
    return table[(bell_index, bob_result)]

# === GHZ Teleportation ===

def teleport_with_ghz_hybrid_realistic(alpha, beta, p1, p2, type1, type2):
    # 1. Make Input State
    psi_in = (alpha * basis(2, 0) + beta * basis(2, 1)).unit()
    rho_in = ket2dm(psi_in)

    # 2. GHZ Channel State
    psi_ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) +
               tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
    rho_channel = ket2dm(psi_ghz)

    # 3. Apply noise to qubit 1 dan 2 of channel state (index 0 dan 1)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type1](p1), 0, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type2](p2), 0, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type1](p1), 1, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type2](p2), 1, 3)

    # 4. join input + channel → total 4 qubit: (0=input, 1=GHZ1, 2=GHZ2, 3=GHZ3)
    rho_total = tensor(rho_in, rho_channel)

    # 5. Bell measurement on qubit (0,1)
    bell_basis = [bell_state(s) for s in ['00', '01', '10', '11']]
    bell_projs = [tensor(b * b.dag(), qeye(2), qeye(2)) for b in bell_basis]

    # 6. Looping for all Bell's and Charlie's measurement posssible outcome
    rho_bob = Qobj(np.zeros((2, 2)), dims=[[2], [2]])
    
    F_total=0

    for i, P in enumerate(bell_projs):
        rho_post_bell = P * rho_total * P.dag()
        prob_bell = rho_post_bell.tr()
        if prob_bell == 0: continue
        rho_post_bell = rho_post_bell / prob_bell

        # Z-projection on qubit 3 (Charlie)
        for z in [0, 1]:
            proj_z = basis(2, z) * basis(2, z).dag()
            Pz = tensor(qeye(2), qeye(2), qeye(2), proj_z)
            rho_proj = Pz * rho_post_bell * Pz.dag()
            prob_z = rho_proj.tr()
            if prob_z == 0: continue
            rho_proj = rho_proj / prob_z

            # Get reduced state of Bob (qubit 2)
            rho_b = rho_proj.ptrace(2)

            # Correction based on Bell and Charlie's outcomes
            correction = ghz_pauli_correction(i, z)
            rho_corr = correction * rho_b * correction.dag()
            rho_corr = rho_corr / rho_corr.tr()
            
            F_i = fidelity(rho_corr, rho_in)

            # Weighted by joint probability
            F_total += prob_bell * prob_z * F_i

            # Bob receives the corrected version according to joint probability

    return F_total

# === Bell + Ancilla teleportation ===
def teleport_with_bell_ancilla_hybrid_realistic(alpha, beta, theta, phi, p1, p2, type1, type2):
    # 1. Qubit input
    psi_in = (alpha * basis(2, 0) + beta * basis(2, 1)).unit()
    rho_in = ket2dm(psi_in)

    # 2. Bell pair (qubit 1&2) + ancilla (qubit 3)
    bell = bell_state('00')
    anc  = (np.cos(theta) * basis(2, 0) 
            + np.exp(1j * phi) * np.sin(theta) * basis(2, 1)).unit()
    psi_channel_raw = tensor(bell, anc)  # qubits 1,2,3

    # 3. Entangle ancilla via CNOT(1->3)
    CNOT_3 = Qobj(np.zeros((8, 8)), dims=[[2]*3, [2]*3])
    for x in [0, 1]:
        P = basis(2, x) * basis(2, x).dag()
        X = qeye(2) if x == 0 else sigmax()
        ops = [P if i==0 else X if i==2 else qeye(2) for i in range(3)]
        CNOT_3 += tensor(ops)
    psi_channel = CNOT_3 * psi_channel_raw
    rho_channel = ket2dm(psi_channel)

    # 4. Apply noise to channel qubits 1 & 2
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type1](p1), target_idx=0, N=3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type2](p2), target_idx=0, N=3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type1](p1), target_idx=1, N=3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[type2](p2), target_idx=1, N=3)

    # 5. Combine with input (qubit 0)
    rho_total = tensor(rho_in, rho_channel)  # qubits [0,1,2,3]

    # 6. Bell‐measurement on qubit 0&1
    bell_basis = [bell_state(s) for s in ['00','01','10','11']]
    projs = [tensor(b*b.dag(), qeye(2), qeye(2)) for b in bell_basis]

    F_total = 0
    for i, P in enumerate(projs):
        # Apply Bell projector
        rho_proj1 = P * rho_total * P.dag()
        prob1    = rho_proj1.tr()
        if prob1 == 0:
            continue
        rho_cond1 = rho_proj1 / prob1

        # Now project ancilla (qubit 3) in Z basis and average both outcomes
        for z in [0, 1]:
            proj_z = basis(2, z) * basis(2, z).dag()
            Pz = tensor(qeye(2), qeye(2), qeye(2), proj_z)
            rho_proj2 = Pz * rho_cond1 * Pz.dag()
            prob2 = rho_proj2.tr()
            if prob2 == 0:
                continue
            rho_cond2 = rho_proj2 / prob2

            # Reduced state for Bob (qubit 2)
            rho_b = rho_cond2.ptrace(2)
            # Pauli correction based on Bell‐outcome i
            pauli = pauli_correction_by_bell_and_charlie(i,z)
            rho_corr = pauli * rho_b * pauli.dag()
            # Normalize
            rho_corr = rho_corr / rho_corr.tr()

            # Fidelity for this branch
            F_i = fidelity(rho_corr, rho_in)

            # Weight by joint probability
            F_total += prob1 * prob2 * F_i

    return F_total


# === Grid and Execution ===
# === Main Grid and Execution ===
input_states = {
    '|0>': (1, 0),
    '|1>': (0, 1),
    '|+>': (1/np.sqrt(2), 1/np.sqrt(2)),
    '|->': (1/np.sqrt(2), -1/np.sqrt(2)),
    '|+i>': (1/np.sqrt(2), 1j/np.sqrt(2)),
    '|-i>': (1/np.sqrt(2), -1j/np.sqrt(2))
}

p1_vals = p2_vals = np.linspace(0.0, 0.5, 6)
thetas = [0, np.pi/4, np.pi/2]  # θ for ancilla
phis = [0, np.pi/4]             # φ for ancilla
noise_types = list(noise_kraus_map.keys())
noise_pairs = list(combinations(noise_types, 2))  # 15 combinations

# Writer to Excel
writer = pd.ExcelWriter("Compound_noise_teleportation_results.xlsx", engine="xlsxwriter")

for type1, type2 in noise_pairs:
    # GHZ
    rows_ghz = []
    for label, (a, b) in input_states.items():
        for p1 in p1_vals:
            for p2 in p2_vals:
                F = teleport_with_ghz_hybrid_realistic(a, b, p1, p2, type1, type2)
                rows_ghz.append({'Input': label, 'p1': p1, 'p2': p2, 'Fidelity': F})
    df_ghz = pd.DataFrame(rows_ghz)
    df_ghz.to_excel(writer, sheet_name=f"GHZ_{type1}_{type2}"[:31], index=False)

    # Ancilla
    rows_anc = []
    for label, (a, b) in input_states.items():
        for theta in thetas:
            for phi in phis:
                for p1 in p1_vals:
                    for p2 in p2_vals:
                        F = teleport_with_bell_ancilla_hybrid_realistic(a, b, theta, phi, p1, p2, type1, type2)
                        rows_anc.append({
                            'Input': label, 'theta': theta, 'phi': phi,
                            'p1': p1, 'p2': p2, 'Fidelity': F
                        })
    df_anc = pd.DataFrame(rows_anc)
    df_anc.to_excel(writer, sheet_name=f"Anc_{type1}_{type2}"[:31], index=False)

writer.close()
print("Disimpan ke hasil_teleportasi_REALISTIK.xlsx")

