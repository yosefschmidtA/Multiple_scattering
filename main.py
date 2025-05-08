import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Parâmetros físicos
k = 10.0  # número de onda (1/Å)
t = 1.0   # fator de espalhamento (constante)

import numpy as np

# Carregar arquivo do cluster (pulando a primeira linha se for o número de átomos)
with open("clusterxyz.xyz", "r") as f:
    lines = f.readlines()[1:]  # ou [2:] se a segunda linha for um comentário
    coords = np.array([
        list(map(float, line.split()[1:4]))
        for line in lines if line.strip()
    ])

# Escolher um átomo como emissor
# Por exemplo, o átomo no centro da estrutura ou um índice específico
emissor_idx = 85  # escolha apropriada com base na estrutura
emissor = coords[emissor_idx]

# Lista de espalhadores (todos menos o emissor)
espalhadores = np.delete(coords, emissor_idx, axis=0)

# Ângulos de emissão
thetas = np.linspace(0, 180, 360)
intensidades = []

# Loop sobre ângulos
for theta_deg in thetas:
    theta_rad = np.radians(theta_deg)
    detector_dir = np.array([np.sin(theta_rad), 0, np.cos(theta_rad)])  # plano xz

    soma_amplitudes = 0.0 + 0.0j

    # Caminhos ordem 2: emissor -> a1 -> a2 -> detector
    for i, j in permutations(range(len(espalhadores)), 2):
        a1 = espalhadores[i]
        a2 = espalhadores[j]

        r01 = np.linalg.norm(a1 - emissor)
        r12 = np.linalg.norm(a2 - a1)

        if r01 == 0 or r12 == 0:
            continue

        # Vetor final até o detector (a2 para infinito na direção do detector)
        fase_final = np.dot(a2, detector_dir)

        R_total = r01 + r12
        fase = np.exp(1j * (k * R_total + k * fase_final))

        A = (t**2) / (r01 * r12) * fase
        soma_amplitudes += A

    intensidade = np.abs(soma_amplitudes)**2
    intensidades.append(intensidade)

# Plot do padrão XPD
plt.figure(figsize=(8, 5))
plt.plot(thetas, intensidades, color='darkgreen')
plt.xlabel("Ângulo de emissão θ (graus)")
plt.ylabel("Intensidade (a.u.)")
plt.title("XPD simulado (ordem 2, rede cúbica)")
plt.grid(True)
plt.tight_layout()
plt.show()
