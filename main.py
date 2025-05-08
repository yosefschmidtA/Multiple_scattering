import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import multiprocessing as mp
from tqdm import tqdm
import os  # Import para manipulação de paths

# Parâmetros físicos
k = 10.0
t = 1.0

def load_cluster(filename="clusterxyz.xyz"):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()[1:]
            coords = np.array([list(map(float, line.split()[1:4])) for line in lines if line.strip()])
        return coords
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filename}' não encontrado.")
        return None

def calcular_intensidades_phi(phi_deg, coords):
    if coords is None:
        return None

    emissor_idx = 85
    emissor = coords[emissor_idx]
    espalhadores = np.delete(coords, emissor_idx, axis=0)

    phi_rad = np.radians(phi_deg)
    intensidades_phi = []
    thetas_deg = np.arange(12, 73, 3)

    for theta_deg in thetas_deg:
        theta_rad = np.radians(theta_deg)
        detector_dir = np.array([np.sin(theta_rad) * np.cos(phi_rad),
                                 np.sin(theta_rad) * np.sin(phi_rad),
                                 np.cos(theta_rad)])

        soma_amplitudes = 0.0 + 0.0j
        for i, j in permutations(range(len(espalhadores)), 2):
            a1 = espalhadores[i]
            a2 = espalhadores[j]

            r01 = np.linalg.norm(a1 - emissor)
            r12 = np.linalg.norm(a2 - a1)

            if r01 == 0 or r12 == 0:
                continue

            fase_final = np.dot(a2, detector_dir)
            R_total = r01 + r12
            fase = np.exp(1j * (k * R_total + k * fase_final))
            A = (t**2) / (r01 * r12) * fase
            soma_amplitudes += A

        intensidade = np.abs(soma_amplitudes)**2
        intensidades_phi.append(intensidade)

    # Cria o diretório para salvar as figuras se não existir
    output_dir = "xpd_curvas"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'xpd_curva_phi_{phi_deg}.png')

    plt.figure(figsize=(8, 5))
    plt.plot(thetas_deg, intensidades_phi, color='darkgreen')
    plt.xlabel("Ângulo polar $\\theta$ (graus)")
    plt.ylabel("Intensidade (a.u.)")
    plt.title(f"Curva de difração XPD para $\\phi = {phi_deg}^\circ$ (ordem 2)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return f"Figura salva em: {filename}"

def process_phi(phi_deg, coords):
    return calcular_intensidades_phi(phi_deg, coords)

if __name__ == '__main__':
    phis_deg = np.arange(0, 360, 3)
    num_cores = 10  # Defina o número de núcleos que você quer usar
    pool = mp.Pool(processes=num_cores)
    coords = load_cluster()

    if coords is not None:
        tasks = [(phi, coords) for phi in phis_deg]
        results = []
        for result in tqdm(pool.starmap(process_phi, tasks), total=len(phis_deg), desc="Processando ângulos phi"):
            results.append(result)

        pool.close()
        pool.join()

        print(f"Foram processados {len(phis_deg)} ângulos phi e as figuras foram salvas na pasta 'xpd_curvas'.")
    else:
        print("O programa foi encerrado devido a um erro no carregamento do arquivo do cluster.")
