import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACAO GLOBAL DOS GRAFICOS
# =========================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# =========================================================
# PARAMETROS DO MODELO
# =========================================================
params = {
    # Dinamica do hospedeiro
    "r": 0.15,
    "K": 5e5,
    "mn": 0.05,
    "mi": 0.3,

    # Latencia
    "sigma": 0.27,

    # Incidencia dependente de V (Holling tipo III)
    "k": 1,
    "beta": 0.6,
    "b": 1e12,

    # Virus ambiental
    "eps": 1e2,
    "delta": 0.7,

    # Condicoes iniciais
    "S0": 8e4,
    "E0": 0.0,
    "I0": 0.0,
    "V0": 0.0,
}

# =========================================================
# FUNCAO DE INCIDENCIA
# =========================================================
def incidence(V, beta, k, b):
    return beta * k * V**2 / (b + V**2)

# =========================================================
# QUANTIDADE R(V)
# =========================================================
def R_V(S, V, p):
    fator1 = (2.0 * p["beta"] * p["k"] * p["b"] * S * V) / (p["b"] + V**2)**2
    fator2 = (
        p["sigma"] * p["eps"] * p["mi"]
        / ((p["sigma"] + p["mn"]) * (p["mn"] + p["mi"]) * p["delta"])
    )
    return fator1 * fator2

# =========================================================
# LADO DIREITO DO SISTEMA
# =========================================================
def rhs(t, y, p, u_func):
    S, E, I, V = y

    lam = incidence(V, p["beta"], p["k"], p["b"])

    dS = p["r"] * S * (1 - S / p["K"]) - lam * S - p["mn"] * S
    dE = lam * S - p["sigma"] * E - p["mn"] * E
    dI = p["sigma"] * E - (p["mn"] + p["mi"]) * I
    dV = u_func(t) + p["eps"] * p["mi"] * I - p["delta"] * V

    return np.array([dS, dE, dI, dV], dtype=float)

# =========================================================
# METODO DE RUNGE-KUTTA DE QUARTA ORDEM
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
        y[n + 1] = np.maximum(y[n + 1], 0.0)

    return t, y

# =========================================================
# FUNCOES DE APLICACAO VIRAL
# =========================================================
def u_pulso_inicial_factory(Q0, dt, t_aplic=0.0):
    def u(t):
        if t_aplic <= t < t_aplic + dt:
            return Q0 / dt
        return 0.0
    return u

def u_reaplicacoes_factory(Q0, dt, tempos):
    def u(t):
        for ta in tempos:
            if ta <= t < ta + dt:
                return Q0 / dt
        return 0.0
    return u

# =========================================================
# FUNCAO AUXILIAR PARA PADRONIZAR GRAFICOS
# =========================================================
def finalizar_grafico(titulo, xlabel, ylabel, nome_arquivo, usar_log=False):
    plt.title(titulo, pad=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if usar_log:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.show()

# =========================================================
# FIGURAS SEPARADAS POR CENARIO
# =========================================================
def plot_scenario_separado(t, S, E, I, V, p, titulo_prefixo, prefixo_arquivo):
    lam = incidence(V, p["beta"], p["k"], p["b"])
    V_plot = np.maximum(V, 1e-20)

    # Figura 1: S, E, I
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, S, linewidth=2.2, label="Suscetíveis $S(t)$")
    plt.plot(t, E, linewidth=2.2, label="Expostos $E(t)$")
    plt.plot(t, I, linewidth=2.2, label="Infectados $I(t)$")
    finalizar_grafico(
        titulo=f"Dinâmica das populações de lagartas - {titulo_prefixo}",
        xlabel="Tempo (dias)",
        ylabel="População",
        nome_arquivo=f"{prefixo_arquivo}_sei.png"
    )

    # Figura 2: V(t)
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, V_plot, linewidth=2.4, label="Carga viral $V(t)$")
    finalizar_grafico(
        titulo=f"Dinâmica da carga viral ambiental - {titulo_prefixo}",
        xlabel="Tempo (dias)",
        ylabel="Carga viral",
        nome_arquivo=f"{prefixo_arquivo}_v.png",
        usar_log=True
    )

    # Figura 3: lambda(V(t))
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, lam, linewidth=2.4, label=r"Força de infecção $\lambda(V(t))$")
    finalizar_grafico(
        titulo=f"Força de infecção - {titulo_prefixo}",
        xlabel="Tempo (dias)",
        ylabel=r"$\lambda(V)$",
        nome_arquivo=f"{prefixo_arquivo}_lambda.png"
    )

# =========================================================
# GRAFICO SEPARADO: R(V) x lambda(V)
# =========================================================
def plot_RV_vs_lambda_separado(S, V, p, titulo_prefixo, nome_arquivo):
    lam = incidence(V, p["beta"], p["k"], p["b"])
    RV = R_V(S, V, p)

    plt.figure(figsize=(10, 4.8))
    plt.plot(lam, RV, linewidth=2.6, label=rf"$R(V)$ vs. $\lambda(V)$")
    finalizar_grafico(
        titulo=rf"Relação entre a força de infecção e $R(V)$ - {titulo_prefixo}",
        xlabel=rf"Força de infecção $\lambda(V)$",
        ylabel=rf"$R(V)$",
        nome_arquivo=nome_arquivo
    )

# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================
if __name__ == "__main__":
    # Intervalo temporal
    t0 = 0.0
    tf = 40.0
    dt = 0.001

    # Vetor de condicoes iniciais
    y0 = [params["S0"], params["E0"], params["I0"], params["V0"]]

    # Dose por aplicacao
    Q0 = 3e6

    # Cenario A: aplicacao unica inicial
    uA = u_pulso_inicial_factory(Q0=Q0, dt=dt, t_aplic=0.0)

    # Cenario B: reaplicacoes periodicas
    tempos_reaplicacao = [0.0, 7.0, 14.0, 21.0, 28.0, 35.0]
    uB = u_reaplicacoes_factory(Q0=Q0, dt=dt, tempos=tempos_reaplicacao)

    # Simulacoes
    t, YA = rk4(rhs, t0, tf, y0, dt, args=(params, uA))
    _, YB = rk4(rhs, t0, tf, y0, dt, args=(params, uB))

    SA, EA, IA, VA = YA.T
    SB, EB, IB, VB = YB.T

    # Figuras separadas do cenario A
    plot_scenario_separado(
        t, SA, EA, IA, VA, params,
        titulo_prefixo="aplicação única inicial",
        prefixo_arquivo="sim_uni"
    )

    # Figuras separadas do cenario B
    plot_scenario_separado(
        t, SB, EB, IB, VB, params,
        titulo_prefixo="reaplicações periódicas",
        prefixo_arquivo="sim_rea"
    )

    # =========================================================
    # GRAFICO UNICO: COMPARACAO DA POPULACAO TOTAL
    # =========================================================
    NA = SA + EA + IA
    NB = SB + EB + IB

    plt.figure(figsize=(10, 4.8))
    plt.plot(t, NA, linewidth=2.6, label="Aplicação única inicial")
    plt.plot(t, NB, linewidth=2.6, label="Reaplicações periódicas")
    finalizar_grafico(
        titulo="Comparação da população total de lagartas",
        xlabel="Tempo (dias)",
        ylabel="População total",
        nome_arquivo="comparacao_total_lagartas.png"
    )

    # =========================================================
    # GRAFICO ADICIONAL: COMPARACAO ENTRE 1 A 5 APLICACOES
    # =========================================================
    cenarios_aplicacao = {
        "1 aplicação": [0.0],
        "2 aplicações": [0.0, 20.0],
        "3 aplicações": [0.0, 15.0, 30.0],
        "4 aplicações": [0.0, 10.0, 20.0, 30.0],
        "5 aplicações": [0.0, 9.0, 18.0, 27.0, 36.0],
        "6 aplicações": [0.0, 7.0, 14.0, 21.0, 28.0, 35.0],
    }

    plt.figure(figsize=(10, 4.8))

    for rotulo, tempos in cenarios_aplicacao.items():
        u_cenario = u_reaplicacoes_factory(Q0=Q0, dt=dt, tempos=tempos)
        _, Y_cenario = rk4(rhs, t0, tf, y0, dt, args=(params, u_cenario))

        S_c, E_c, I_c, V_c = Y_cenario.T
        N_c = S_c + E_c + I_c

        plt.plot(t, N_c, linewidth=2.2, label=rotulo)

    finalizar_grafico(
        titulo="Comparação da população total para diferentes números de aplicações",
        xlabel="Tempo (dias)",
        ylabel="População total",
        nome_arquivo="comparacao_1_a_5_aplicacoes.png"
    )
