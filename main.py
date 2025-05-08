import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import multiprocessing as mp
from tqdm import tqdm
import os

# Parâmetros físicos
k = 10.0
t = 1.0
ordem_espalhamento = 4
num_vizinhos_ra = 6

def load_cluster(filename="clusterxyz.xyz"):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()[1:]
            coords = np.array([list(map(float, line.split()[1:4])) for line in lines if line.strip()])
        return coords
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filename}' não encontrado.")
        return None

def calcular_intensidades_theta(theta_deg, coords):
    if coords is None:
        return None

    emissor_idx = 85
    emissor = coords[emissor_idx]
    restantes = np.delete(coords, emissor_idx, axis=0)

    distancias = np.linalg.norm(restantes - emissor, axis=1)
    indices_ra = np.argsort(distancias)[:num_vizinhos_ra]
    espalhadores = restantes[indices_ra]

    phis_deg = np.arange(0, 360, 3)
    theta_rad = np.radians(theta_deg)
    intensidades_theta = []

    for phi_deg in phis_deg:
        phi_rad = np.radians(phi_deg)

        detector_dir = np.array([np.sin(theta_rad) * np.cos(phi_rad),
                                 np.sin(theta_rad) * np.sin(phi_rad),
                                 np.cos(theta_rad)])

        soma_amplitudes = 0.0 + 0.0j

        for caminho in permutations(range(num_vizinhos_ra), ordem_espalhamento):
            pontos = [emissor] + [espalhadores[i] for i in caminho]

            R_total = 0.0
            denominador = 1.0

            for i in range(len(pontos) - 1):
                r = np.linalg.norm(pontos[i+1] - pontos[i])
                if r == 0:
                    break
                R_total += r
                denominador *= r
            else:
                fase_final = np.dot(pontos[-1], detector_dir)
                fase = np.exp(1j * (k * R_total + k * fase_final))
                A = (t ** ordem_espalhamento) / denominador * fase
                soma_amplitudes += A

        intensidade = np.abs(soma_amplitudes) ** 2
        intensidades_theta.append(intensidade)

    output_dir = "xpd_curvas_phi_vs_theta"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'xpd_curva_theta_{theta_deg}.png')

    plt.figure(figsize=(8, 5))
    plt.plot(phis_deg, intensidades_theta, color='darkred')
    plt.xlabel("Ângulo azimutal $\\phi$ (graus)")
    plt.ylabel("Intensidade (a.u.)")
    plt.title(f"Curva XPD para $\\theta = {theta_deg}^\circ$ (ordem {ordem_espalhamento}, R-A {num_vizinhos_ra})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return f"Figura salva: {filename}"

def process_theta(theta_deg, coords):
    return calcular_intensidades_theta(theta_deg, coords)

if __name__ == '__main__':
    thetas_deg = np.arange(12, 73, 3)
    num_cores = 10
    pool = mp.Pool(processes=num_cores)
    coords = load_cluster()

    if coords is not None:
        tasks = [(theta, coords) for theta in thetas_deg]
        results = []
        for result in tqdm(pool.starmap(process_theta, tasks), total=len(thetas_deg), desc="Processando ângulos theta"):
            results.append(result)

        pool.close()
        pool.join()

        print(f"Foram processados {len(thetas_deg)} ângulos theta. Figuras salvas na pasta 'xpd_curvas_phi_vs_theta'.")
    else:
        print("Erro ao carregar o arquivo do cluster.")
