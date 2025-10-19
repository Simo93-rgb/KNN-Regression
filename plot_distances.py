import os

import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir, 'Assets')
def minkowski_curve(p, r=1, points=1000):
    """
    Genera i punti di una curva di livello per la distanza di Minkowski
    mantenendo la corretta simmetria in tutti i quadranti.
    """
    # Generiamo punti nel primo quadrante
    t = np.linspace(0, np.pi/2, points)

    # Formula parametrica per la curva di livello della distanza di Minkowski
    x = r * np.cos(t)**(2/p)
    y = r * np.sin(t)**(2/p)

    # Creiamo i punti per tutti i quadranti mantenendo la simmetria
    x_full = np.concatenate([x, -x[::-1], -x, x[::-1]])
    y_full = np.concatenate([y, y[::-1], -y, -y[::-1]])

    return x_full, y_full

# Creiamo il plot
plt.figure(figsize=(10, 10))

# Lista dei valori di p da plottare
p_values = [1, 1.5, 2, 4, 10, 50, 100]
colors = ['purple', 'navy', 'blue', 'cyan', 'green', 'lime', 'yellow']

# Plottiamo le curve per ogni valore di p
for p, color in zip(p_values, colors):
    x, y = minkowski_curve(p, r=4)  # r=4 per match con la griglia
    plt.plot(x, y, '-', label=f'p = {p}', color=color, linewidth=2)

# Aggiungiamo il quadrato di Chebyshev
plt.plot([-4, -4, 4, 4, -4], [-4, 4, 4, -4, -4], 'r--',
         label='Chebyshev (p → ∞)', linewidth=2)

# Configuriamo il grafico
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve distanza di Minkowski al variare di p', fontsize=24)
plt.legend(loc='lower right')

# Impostiamo i limiti degli assi
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)

plt.tight_layout()
plt.savefig(f'{assets_dir}/minkowski as p changes.png', format='png', dpi=600, bbox_inches='tight')
plt.show()