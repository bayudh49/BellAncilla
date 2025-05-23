import sympy as sp
import os
import platform
import csv
from sympy.physics.quantum import TensorProduct
from itertools import combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json


# === Global Symbol ===
alpha, beta = sp.symbols('alpha beta', complex=True)
p1, p2, theta, phi = sp.symbols('p1 p2 theta phi', positive=True)


# === Utility: Remove conjugates ===
def remove_conjugates(expr):
    
    subs_map = {
        sp.conjugate(p1): p1,
        sp.conjugate(p2): p2,
        sp.conjugate(theta): theta,
        sp.conjugate(phi): phi,
        sp.conjugate(sp.sqrt(p1)): sp.sqrt(p1),
        sp.conjugate(sp.sqrt(1 - p1)): sp.sqrt(1 - p1),
        sp.conjugate(sp.sqrt(p2)): sp.sqrt(p2),
        sp.conjugate(sp.sqrt(1 - p2)): sp.sqrt(1 - p2),
    }
    expr = expr.xreplace(subs_map)

    
    if isinstance(expr, sp.MatrixBase):
        return expr.applyfunc(lambda e: sp.simplify(e))
    else:
        return sp.simplify(expr)



# === Kraus Noise Operator ===

def kraus_bitflip(p):
    I2 = sp.eye(2)
    X = sp.Matrix([[0,1],[1,0]])
    ops = [sp.sqrt(1 - p)*I2, sp.sqrt(p)*X]
    return [remove_conjugates(op) for op in ops]

def kraus_phaseflip(p):
    I2 = sp.eye(2)
    Z = sp.Matrix([[1,0],[0,-1]])
    ops = [sp.sqrt(1 - p)*I2, sp.sqrt(p)*Z]
    return [remove_conjugates(op) for op in ops]

def kraus_bitphaseflip(p):
    I2 = sp.eye(2)
    Y = sp.Matrix([[0,-sp.I],[sp.I,0]])
    ops = [sp.sqrt(1 - p)*I2, sp.sqrt(p)*Y]
    return [remove_conjugates(op) for op in ops]

def kraus_depolarizing(p):
    I2 = sp.eye(2)
    X = sp.Matrix([[0,1],[1,0]])
    Y = sp.Matrix([[0,-sp.I],[sp.I,0]])
    Z = sp.Matrix([[1,0],[0,-1]])
    ops = [
        sp.sqrt(1 - p)*I2,
        sp.sqrt(p/3)*X,
        sp.sqrt(p/3)*Y,
        sp.sqrt(p/3)*Z
    ]
    return [remove_conjugates(op) for op in ops]

def kraus_amplitude_damping(p):
    ops = [
        sp.Matrix([[1,0],[0,sp.sqrt(1 - p)]]),
        sp.Matrix([[0,sp.sqrt(p)],[0,0]])
    ]
    return [remove_conjugates(op) for op in ops]

def kraus_phase_damping(p):
    ops = [
        sp.Matrix([[1,0],[0,sp.sqrt(1 - p)]]),
        sp.Matrix([[0,0],[0,sp.sqrt(p)]])
    ]
    return [remove_conjugates(op) for op in ops]



noise_kraus_map = {
    "BitFlip":          kraus_bitflip,
    "PhaseFlip":        kraus_phaseflip,
    "BitPhaseFlip":     kraus_bitphaseflip,
    "Depolarizing":     kraus_depolarizing,
    "AmplitudeDamping": kraus_amplitude_damping,
    "PhaseDamping":     kraus_phase_damping
}


def is_complex_matrix(mat):
    #Check if a matrix contains imaginary (complex) entries
    return any(entry.has(sp.I) for entry in mat)

def apply_kraus_to_qubit(rho, kraus_ops, target_idx, N):
    out = sp.zeros(2**N, 2**N)
    for idx, K in enumerate(kraus_ops):
        
        
        is_complex = is_complex_matrix(K)
        

        ops = [sp.eye(2)] * N
        ops[target_idx] = K
        E = TensorProduct(*ops)

        if is_complex:
            
            E_conj = E.H
        else:
            
            E_conj = E.T

        term = E * rho * E_conj
        out += term

    return remove_conjugates(out)


def fidelity_sympy(rho1, rho2):
    return sp.sqrt(sp.simplify((rho1 * rho2).trace()))

def partial_trace_013(rho4):
    rb = sp.zeros(2,2)
    for i in range(16):
        for j in range(16):
            bi = format(i,'04b'); bj = format(j,'04b')
            if bi[0]==bj[0] and bi[1]==bj[1] and bi[3]==bj[3]:
                m = int(bi[2]); n = int(bj[2])
                rb[m,n] += rho4[i,j]
    return rb

def pauli_correction_by_bell_and_charlie(bell_idx, charlie_z):
    I2 = sp.eye(2)
    X = sp.Matrix([[0,1],[1,0]])
    Y = sp.Matrix([[0,-sp.I],[sp.I,0]])
    Z = sp.Matrix([[1,0],[0,-1]])
    mapping = {
        (0, 0): I2, (0, 1): Z,
        (1, 0): Z,  (1, 1): I2,
        (2, 0): X,  (2, 1): Y,
        (3, 0): Y,  (3, 1): X,
    }
    return mapping.get((bell_idx, charlie_z), I2)

def ghz_pauli_correction(bell_idx, z):
    I2 = sp.eye(2)
    X = sp.Matrix([[0,1],[1,0]])
    Y = sp.Matrix([[0,-sp.I],[sp.I,0]])
    Z = sp.Matrix([[1,0],[0,-1]])
    table = {
        (0,0): I2, (0,1): Z,
        (1,0): Z, (1,1): I2,
        (2,0): X, (2,1): Y,
        (3,0): Y, (3,1): X
    }
    return table.get((bell_idx,z), I2)

def teleport_with_ghz(a, b, p1, p2, t1, t2):
    ket0 = sp.Matrix([[1],[0]])
    ket1 = sp.Matrix([[0],[1]])
    psi_in = a*ket0 + b*ket1
    rho_in = psi_in * psi_in.H

    ghz = (TensorProduct(ket0,ket0,ket0) + TensorProduct(ket1,ket1,ket1)) / sp.sqrt(2)
    rho_chan = ghz * ghz.H
    rho_chan = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t1](p1), 0, 3)
    rho_chan = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t2](p2), 0, 3)
    rho_chan = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t1](p1), 1, 3)
    rho_chan = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t2](p2), 1, 3)

    rho_total = TensorProduct(rho_in, rho_chan)

    bell = [
        (TensorProduct(ket0,ket0)+TensorProduct(ket1,ket1))/sp.sqrt(2),
        (TensorProduct(ket0,ket0)-TensorProduct(ket1,ket1))/sp.sqrt(2),
        (TensorProduct(ket0,ket1)+TensorProduct(ket1,ket0))/sp.sqrt(2),
        (TensorProduct(ket0,ket1)-TensorProduct(ket1,ket0))/sp.sqrt(2),
    ]
    bell_projs = [TensorProduct(b*b.H, sp.eye(2), sp.eye(2)) for b in bell]

    F_total = 0
    for i,P in enumerate(bell_projs):
        rho_b = P * rho_total * P
        prob_b = rho_b.trace()
        if prob_b == 0:
            continue
        rho_b /= prob_b
        for z in [0,1]:
            proj_z = TensorProduct(sp.eye(2),sp.eye(2),sp.eye(2), sp.diag(1 if z==0 else 0, 1 if z==1 else 0))
            rho_c = proj_z * rho_b * proj_z
            prob_z = rho_c.trace()
            if prob_z == 0:
                continue
            rho_c /= prob_z
            rho_red = sp.zeros(2,2)
            for ii in range(16):
                for jj in range(16):
                    bi = format(ii,'04b'); bj = format(jj,'04b')
                    if bi[0:2]==bj[0:2] and bi[3]==bj[3]:
                        m = int(bi[2]); n = int(bj[2])
                        rho_red[m,n] += rho_c[ii,jj]
            U = ghz_pauli_correction(i,z)
            rho_corr = U * rho_red * U.H
            rho_corr /= rho_corr.trace()
            F_total += prob_b * prob_z * fidelity_sympy(rho_corr, rho_in)
    return F_total

def teleport_with_ancilla(a, b, theta, phi, p1, p2, t1, t2):
    ket0 = sp.Matrix([[1],[0]])
    ket1 = sp.Matrix([[0],[1]])
    psi_in = a*ket0 + b*ket1
    rho_in = psi_in * psi_in.H

    bell = (TensorProduct(ket0, ket0) + TensorProduct(ket1, ket1)) / sp.sqrt(2)
    anc  = sp.cos(theta) * ket0 + sp.exp(sp.I * phi) * sp.sin(theta) * ket1
    psi_channel_raw = TensorProduct(bell, anc)

    CNOT = sp.zeros(8, 8)
    for i in range(8):
        bits = list(map(int, format(i, '03b')))
        if bits[0] == 1:
            bits[2] ^= 1
        j = int(''.join(map(str, bits)), 2)
        CNOT[j, i] = 1
    psi_channel = CNOT * psi_channel_raw
    rho_channel = psi_channel * psi_channel.H
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[t1](p1), 0, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[t2](p2), 0, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[t1](p1), 1, 3)
    rho_channel = apply_kraus_to_qubit(rho_channel, noise_kraus_map[t2](p2), 1, 3)

    rho_total = TensorProduct(rho_in, rho_channel)

    bell_states = [
        (TensorProduct(ket0, ket0) + TensorProduct(ket1, ket1)) / sp.sqrt(2),
        (TensorProduct(ket0, ket0) - TensorProduct(ket1, ket1)) / sp.sqrt(2),
        (TensorProduct(ket0, ket1) + TensorProduct(ket1, ket0)) / sp.sqrt(2),
        (TensorProduct(ket0, ket1) - TensorProduct(ket1, ket0)) / sp.sqrt(2),
    ]
    projs = [TensorProduct(b * b.H, sp.eye(2), sp.eye(2)) for b in bell_states]

    F_total = 0
    for i, P in enumerate(projs):
        rho_proj = P*rho_total*P
        pr1 = rho_proj.trace()
        if pr1 == 0:
            continue
        rho_cond = rho_proj / pr1
        for z in [0, 1]:
            proj_z = sp.diag(1 if z==0 else 0, 1 if z==1 else 0)
            Pz = TensorProduct(sp.eye(2), sp.eye(2), sp.eye(2), proj_z)
            rho_pz = Pz * rho_cond * Pz
            pr2 = rho_pz.trace()
            if pr2 == 0:
                continue
            rho_pzn = rho_pz / pr2
            rho_bob = partial_trace_013(rho_pzn)
            U = pauli_correction_by_bell_and_charlie(i,z)
            rho_corr = U * rho_bob * U.H            
            rho_corr = rho_corr / rho_corr.trace()
            F_total += pr1 * pr2 * fidelity_sympy(rho_corr, rho_in)
    return F_total

# === Parallel task wrappers ===
def compute_ghz_fidelity(task):
    label, a, b, p1_val, p2_val, t1, t2 = task
    F = teleport_with_ghz(a, b, p1_val, p2_val, t1, t2)
    return (t1, t2, [label, p1_val, p2_val, F])

def compute_anc_fidelity(task):
    label, a, b, theta_val, phi_val, p1_val, p2_val, t1, t2 = task
    F = teleport_with_ancilla(a, b, theta_val, phi_val, p1_val, p2_val, t1, t2)
    return (t1, t2, [label, theta_val, phi_val, p1_val, p2_val, F])

def export_latex_teleportation(filename, a, b, theta_val, phi_val, p1_val, p2_val, t1, t2, mode='ghz'):
    if mode == 'ghz':
        F = teleport_with_ghz(a, b, p1_val, p2_val, t1, t2)
    else:
        F = teleport_with_ancilla(a, b, theta_val, phi_val, p1_val, p2_val, t1, t2)

    ket0 = sp.Matrix([[1],[0]])
    ket1 = sp.Matrix([[0],[1]])
    psi_in = a*ket0 + b*ket1
    rho_in = psi_in * psi_in.H

    if mode == 'ghz':
        ghz = (TensorProduct(ket0,ket0,ket0) + TensorProduct(ket1,ket1,ket1)) / sp.sqrt(2)
        rho_chan = ghz * ghz.H
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t1](p1_val), 0, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t2](p2_val), 0, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t1](p1_val), 1, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t2](p2_val), 1, 3)
        rho_chan_noisy = remove_conjugates(rho_chan_noisy)
    else:
        bell = (TensorProduct(ket0, ket0) + TensorProduct(ket1, ket1)) / sp.sqrt(2)
        anc  = sp.cos(theta_val) * ket0 + sp.exp(sp.I * phi_val) * sp.sin(theta_val) * ket1
        psi_channel_raw = TensorProduct(bell, anc)
        CNOT = sp.zeros(8, 8)
        for i in range(8):
            bits = list(map(int, format(i, '03b')))
            if bits[0] == 1:
                bits[2] ^= 1
            j = int(''.join(map(str, bits)), 2)
            CNOT[j, i] = 1
        psi_channel = CNOT * psi_channel_raw
        rho_chan = psi_channel * psi_channel.H
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan, noise_kraus_map[t1](p1_val), 0, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t2](p2_val), 0, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t1](p1_val), 1, 3)
        rho_chan_noisy = apply_kraus_to_qubit(rho_chan_noisy, noise_kraus_map[t2](p2_val), 1, 3)
        rho_chan_noisy = remove_conjugates(rho_chan_noisy)

    rho_join = TensorProduct(rho_in, rho_chan_noisy)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{breqn}
\begin{document}
""")
        def write_section(title_math, matrix):
            f.write(f"\\section*{{{title_math}}}\n")
            latex_str = sp.latex(matrix, mode='plain')
            f.write(f"\\begin{{dmath*}}\n{latex_str}\n\\end{{dmath*}}\n")


        write_section("$\\psi_{\\text{in}}$", psi_in)
        write_section("$\\rho_{\\text{in}}$", rho_in)
        write_section("$\\rho_{\\text{channel}}$", rho_chan)
        write_section("$\\rho_{\\text{noisy}}$", rho_chan_noisy)
        write_section("$\\rho_{\\text{gabungan}}$", rho_join)
        write_section("$\\text{Fidelity}$", F)

        f.write("\\end{document}")

def safe_shutdown():
    if platform.system() == 'Linux':
        print("🔌 Mematikan VM Linux...")
        os.system("sudo shutdown -h now")
    else:
        print("Detected non-Linux system, skip shutdown.")

# === Main function ===
if __name__ == '__main__':
    try:
        output_folder = "outputlatex_subs"
        os.makedirs(output_folder, exist_ok=True)

        progress_file_ghz = "progress_ghz.json"
        progress_file_anc = "progress_anc.json"

        a1, b1 = sp.symbols('a1 b1', complex=True)
        input_states = {'in': (a1, b1)}
        theta_val, phi_val = theta, phi
        p1_val, p2_val = p1, p2
        noise_pairs = list(combinations(noise_kraus_map.keys(), 2))

        # === GHZ Tasks ===
        print("Memproses GHZ secara paralel...")

        all_ghz_tasks = [(label, a, b, p1_val, p2_val, t1, t2)
                         for t1, t2 in noise_pairs
                         for label, (a, b) in input_states.items()]

        # Load finished tasks if available
        finished_ghz = set()
        if os.path.exists(progress_file_ghz):
            with open(progress_file_ghz, 'r') as f:
                finished_ghz = set(json.load(f))

        remaining_ghz_tasks = [t for t in all_ghz_tasks if f"{t[0]}_{t[5]}_{t[6]}" not in finished_ghz]

        with Pool(cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(compute_ghz_fidelity, remaining_ghz_tasks), total=len(remaining_ghz_tasks), desc="GHZ Progress"):
                t1, t2, row = result
                label, p1v, p2v, F = row
                task_key = f"{label}_{t1}_{t2}"
                filename = f"{output_folder}/ghz_{task_key}.tex"
                # export_latex_teleportation(filename, a1, b1, theta_val, phi_val, sp.sympify(p1v), sp.sympify(p2v), t1, t2, mode='ghz')
                export_latex_teleportation(
                    filename,
                    a1, b1,
                    theta_val=theta_val,
                    phi_val=phi_val,
                    p1_val=p1v,
                    p2_val=p2v,
                    t1=t1,
                    t2=t2,
                    mode='ghz'
                )

                finished_ghz.add(task_key)
                with open(progress_file_ghz, 'w') as f:
                    json.dump(list(finished_ghz), f)

        # === ANCILLA Tasks ===
        print("Memproses Ancilla secara paralel...")

        all_anc_tasks = [(label, a, b, theta_val, phi_val, p1_val, p2_val, t1, t2)
                         for t1, t2 in noise_pairs
                         for label, (a, b) in input_states.items()]

        finished_anc = set()
        if os.path.exists(progress_file_anc):
            with open(progress_file_anc, 'r') as f:
                finished_anc = set(json.load(f))

        remaining_anc_tasks = [t for t in all_anc_tasks if f"{t[0]}_{t[7]}_{t[8]}" not in finished_anc]

        with Pool(cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(compute_anc_fidelity, remaining_anc_tasks), total=len(remaining_anc_tasks), desc="Ancilla Progress"):
                t1, t2, row = result
                label, thv, phiv, p1v, p2v, F = row
                task_key = f"{label}_{t1}_{t2}"
                filename = f"{output_folder}/anc_{task_key}.tex"
                export_latex_teleportation(
                    filename,
                    a1, b1,
                    theta_val=thv,
                    phi_val=phiv,
                    p1_val=p1v,
                    p2_val=p2v,
                    t1=t1,
                    t2=t2,
                    mode='ancilla'
                )


                finished_anc.add(task_key)
                with open(progress_file_anc, 'w') as f:
                    json.dump(list(finished_anc), f)

        print("Semua file LaTeX telah disimpan di folder:", output_folder)

    finally:
        safe_shutdown()
