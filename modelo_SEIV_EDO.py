import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Parametros do modelo
# =========================================================
params = {
    # Dinamica do hospedeiro
    "r": 0.15,         # taxa de reproducao
    "K": 1000.0,       # capacidade de suporte
    "mn": 0.05,        # mortalidade natural
    "mi": 0.30,        # mortalidade induzida

    # Latencia
    "sigma": 0.27,     # taxa de transicao E -> I

    # Incidencia dependente de V (Holling tipo III)
    "k": 0.1,
    "beta": 0.8,
    "b": 1e12,

    # Virus ambiental
    "eps": 1e5,        # virus liberados por morte induzida
    "delta": 0.70,     # taxa de decaimento

    # Condicoes iniciais
    "S0": 800.0,
    "E0": 0.0,
    "I0": 0.0,
    "V0": 0.0,
}

# =========================================================
# Funcao de incidencia
# =========================================================
def incidence(V, beta, k, b):
    """
    lambda(V) = beta * k * V^2 / (b + V^2)
    """
    return beta * k * V**2 / (b + V**2)

# =========================================================
# Lado direito do sistema
# =========================================================
def rhs(t, y, p, u_func):
    S, E, I, V = y

    lam = incidence(V, p["beta"], p["k"], p["b"])

    dS = p["r"] * S * (1.0 - S / p["K"]) - lam * S - p["mn"] * S
    dE = lam * S - p["sigma"] * E - p["mn"] * E
    dI = p["sigma"] * E - (p["mn"] + p["mi"]) * I
    dV = u_func(t) + p["eps"] * p["mi"] * I - p["delta"] * V

    return np.array([dS, dE, dI, dV], dtype=float)

# =========================================================
# Metodo de Runge-Kutta de quarta ordem
# =========================================================
def rk4(f, t0, tf, y0, dt, args=()):
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((t.size, len(y0)), dtype=float)
    y[0] = np.array(y0, dtype=float)

    for n in range(t.size - 1):
        tn = t[n]
        yn = y[n]

        k1 = f(tn, yn, *args)
        k2 = f(tn + dt / 2.0, yn + dt * k1 / 2.0, *args)
        k3 = f(tn + dt / 2.0, yn + dt * k2 / 2.0, *args)
        k4 = f(tn + dt, yn + dt * k3, *args)

        y[n + 1] = yn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Seguranca numerica
        y[n + 1] = np.maximum(y[n + 1], 0.0)

    return t, y

# =========================================================
# Funcoes de aplicacao viral
# =========================================================
def u_pulso_inicial_factory(Q0, dt, t_aplic=0.0):
    """
    Aproxima uma aplicacao instantanea por um pulso em um unico passo:
    integral u(t) dt ~= Q0, logo u = Q0 / dt em [t_aplic, t_aplic + dt).
    """
    def u(t):
        if t_aplic <= t < t_aplic + dt:
            return Q0 / dt
        return 0.0
    return u

def u_reaplicacoes_factory(Q0, dt, tempos):
    """
    Reaplicacoes modeladas como pulsos discretos:
    em cada pulso, integral u(t) dt ~= Q0.
    """
    def u(t):
        for ta in tempos:
            if ta <= t < ta + dt:
                return Q0 / dt
        return 0.0
    return u

# =========================================================
# Geracao de figuras separadas
# =========================================================
def plot_scenario_separado(t, S, E, I, V, p, titulo_prefixo, prefixo_arquivo):
    lam = incidence(V, p["beta"], p["k"], p["b"])

    # Evita problemas com escala logaritmica quando V se aproxima de zero
    V_plot = np.maximum(V, 1e-20)

    # Figura 1: S, E, I
    plt.figure(figsize=(10, 4))
    plt.plot(t, S, label="S(t)")
    plt.plot(t, E, label="E(t)")
    plt.plot(t, I, label="I(t)")
    plt.title(f"Dinamica de S, E e I - {titulo_prefixo}")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Populacao")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{prefixo_arquivo}_sei.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figura 2: V(t)
    plt.figure(figsize=(10, 4))
    plt.plot(t, V_plot, label="V(t)")
    plt.yscale("log")
    plt.title(f"Dinamica de V - {titulo_prefixo}")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Carga viral")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{prefixo_arquivo}_v.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figura 3: lambda(V(t))
    plt.figure(figsize=(10, 4))
    plt.plot(t, lam, label=r"$\lambda(V(t))$")
    plt.title(f"Forca de infeccao ao longo do tempo - {titulo_prefixo}")
    plt.xlabel("Tempo (dias)")
    plt.ylabel(r"$\lambda(V)$")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{prefixo_arquivo}_lambda.png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================================================
# Programa principal
# =========================================================
if __name__ == "__main__":
    # Intervalo temporal
    t0 = 0.0
    tf = 200.0
    dt = 0.001

    # Vetor de condicoes iniciais
    y0 = [params["S0"], params["E0"], params["I0"], params["V0"]]

    # Dose por aplicacao
    Q0 = 3e9

    # Cenario A: aplicacao unica inicial
    uA = u_pulso_inicial_factory(Q0=Q0, dt=dt, t_aplic=0.0)

    # Cenario B: reaplicacoes periodicas
    tempos_reaplicacao = [
        0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
        100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0,
        180.0, 190.0, 200.0
    ]
    uB = u_reaplicacoes_factory(Q0=Q0, dt=dt, tempos=tempos_reaplicacao)

    # Simulacoes
    t, YA = rk4(rhs, t0, tf, y0, dt, args=(params, uA))
    _, YB = rk4(rhs, t0, tf, y0, dt, args=(params, uB))

    SA, EA, IA, VA = YA.T
    SB, EB, IB, VB = YB.T

    # Figuras do cenario A
    plot_scenario_separado(
        t, SA, EA, IA, VA, params,
        titulo_prefixo="Aplicacao unica inicial",
        prefixo_arquivo="sim_uni"
    )

    # Figuras do cenario B
    plot_scenario_separado(
        t, SB, EB, IB, VB, params,
        titulo_prefixo="Cenario com reaplicacoes",
        prefixo_arquivo="sim_rea"
    )
