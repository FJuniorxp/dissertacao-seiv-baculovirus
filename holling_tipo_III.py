import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros Biológicos ---
beta = 0.5  # Transmissibilidade
k = 1.0     # Taxa máxima de ingestão
b = 10.0    # Constante de semi-saturação
beta_k = beta * k

# --- Definição da Função de Holling Tipo III ---
def lambda_V(V):
    return beta * k * (V**2) / (b + V**2)

# --- Configurações de estilo ---
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ==============================================================================
# Figura 1: Análise Assintótica e Convergência (Saturação)
# ==============================================================================
V = np.linspace(0, 20, 500)
L = lambda_V(V)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(V, L, label=r'$\lambda(V) = \beta \frac{k V^2}{b + V^2}$', color='#1f77b4', linewidth=2.5)
ax1.axhline(y=beta_k, color='#d62728', linestyle='--', linewidth=2, label=r'Assíntota $\beta k$ (Saturação)')

ax1.set_title('Saturação da Taxa de Infecção', pad=15)
ax1.set_xlabel('Carga Viral Ambiental ($V$)')
ax1.set_ylabel(r'Força de Infecção $\lambda(V)$')
ax1.legend(loc='lower right')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(0, beta_k * 1.1)

plt.tight_layout()
fig1.savefig('figura_saturacao.png', dpi=300) # Salva em alta resolução para o LaTeX

# ==============================================================================
# Figura 2: Comportamento em Baixas Densidades e Efeito Limiar
# ==============================================================================
V_low = np.linspace(0, 4, 500)
L_low = lambda_V(V_low)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(V_low, L_low, label=r'Curva sigmoide em baixas densidades', color='#2ca02c', linewidth=2.5)

# Desenhando a tangente na origem
tangente = np.zeros_like(V_low)
ax2.plot(V_low, tangente, color='black', linestyle='-.', linewidth=2, label=r'Tangente nula na origem ($\lambda\'(0)=0$)')

ax2.set_title('Efeito Limiar (Inércia Inicial à Infecção)', pad=15)
ax2.set_xlabel('Carga Viral Ambiental ($V$)')
ax2.set_ylabel(r'Força de Infecção $\lambda(V)$')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.7)

# Destacando a área de baixa infectividade
ax2.fill_between(V_low, L_low, 0, where=(V_low < 2), color='#2ca02c', alpha=0.1)
ax2.text(1.0, 0.015, 'Barreira\nde Defesa', color='#2ca02c', fontsize=11, fontweight='bold')

plt.tight_layout()
fig2.savefig('figura_limiar.png', dpi=300)

# ==============================================================================
# Figura 3: Ponto de Inflexão e Relação com a CL50
# ==============================================================================
V_inflection = np.sqrt(b / 3)
V_cl50 = np.sqrt(b)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(V, L, label=r'$\lambda(V)$', color='#9467bd', linewidth=2.5)

# Marcando o Ponto de Inflexão
L_inflection = lambda_V(V_inflection)
ax3.plot(V_inflection, L_inflection, 'ko', markersize=8, zorder=5)
ax3.annotate(r'Inflexão ($V = \sqrt{b/3}$)', xy=(V_inflection, L_inflection), xytext=(V_inflection+1, L_inflection - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
ax3.vlines(V_inflection, 0, L_inflection, colors='k', linestyles=':', alpha=0.5)

# Marcando a CL50
L_cl50 = lambda_V(V_cl50)
ax3.plot(V_cl50, L_cl50, 'ro', markersize=8, zorder=5)
ax3.annotate(r'$CL_{50}$ ($V = \sqrt{b}$)', xy=(V_cl50, L_cl50), xytext=(V_cl50 + 1, L_cl50 - 0.05),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5), color='red')
ax3.vlines(V_cl50, 0, L_cl50, colors='red', linestyles=':', alpha=0.5)
ax3.hlines(L_cl50, 0, V_cl50, colors='red', linestyles=':', alpha=0.5)

ax3.axhline(y=beta_k, color='gray', linestyle='--', alpha=0.5)

ax3.set_title(r'Dinâmica do Ponto de Inflexão e $CL_{50}$', pad=15)
ax3.set_xlabel('Carga Viral Ambiental ($V$)')
ax3.set_ylabel(r'Força de Infecção $\lambda(V)$')
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.set_ylim(0, beta_k * 1.1)

plt.tight_layout()
fig3.savefig('figura_inflexao_cl50.png', dpi=300)

plt.show()
