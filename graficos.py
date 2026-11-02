import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

AZUL, AZUL_CLARO, AZUL_ESCURO = "#2a78d6", "#86b6ef", "#104281"
TINTA, GRADE = "#222222", "#dddddd"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.edgecolor": GRADE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "savefig.dpi": 300,
})

avaliacoes = pd.read_csv("ratings_small.csv")

# Formata inteiros com ponto como separador de milhar
def milhar(valor, _=None):
    return f"{int(valor):,}".replace(",", ".")

# Formata decimais com uma casa e vírgula
def virgula(valor):
    return f"{valor:.1f}".replace(".", ",")

# Gera o histograma com a quantidade de avaliações por nota
def gerar_histograma(avaliacoes):
    contagem = avaliacoes["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.bar(contagem.index, contagem.values, width=0.42,
           color=AZUL, zorder=3)
    for nota, qtd in contagem.items():
        ax.text(nota, qtd + 400, milhar(qtd), ha="center",
                va="bottom", fontsize=8.5, color=TINTA)
    ax.set_xticks(contagem.index)
    ax.set_xticklabels([virgula(n) for n in contagem.index])
    ax.yaxis.set_major_formatter(FuncFormatter(milhar))
    ax.set_xlabel("Nota")
    ax.set_ylabel("Quantidade de avaliações")
    ax.set_ylim(0, contagem.max() * 1.12)
    ax.grid(axis="y", color=GRADE, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig("fig_histograma.png")
    plt.close(fig)

gerar_histograma(avaliacoes)

# Gera a pizza com a participação de cada faixa de notas
def gerar_pizza(avaliacoes):
    nota = avaliacoes["rating"]
    faixas = {
        "Notas baixas (0,5 a 2,0)": (nota <= 2.0).sum(),
        "Notas intermediárias (2,5 a 3,5)":
            nota.between(2.5, 3.5).sum(),
        "Notas altas (4,0 a 5,0)": (nota >= 4.0).sum(),
    }
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    fatias, _, rotulos = ax.pie(
        faixas.values(),
        colors=[AZUL_CLARO, AZUL, AZUL_ESCURO],
        startangle=90,
        counterclock=False,
        autopct=lambda p: virgula(p) + "%",
        pctdistance=0.68,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for rotulo, cor in zip(rotulos, [TINTA, "white", "white"]):
        rotulo.set_color(cor)
    ax.legend(fatias, faixas.keys(), loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_aspect("equal")
    fig.savefig("fig_pizza.png", bbox_inches="tight")
    plt.close(fig)

gerar_pizza(avaliacoes)

# Gera o boxplot da quantidade de avaliações por usuário
def gerar_boxplot(avaliacoes):
    por_usuario = avaliacoes.groupby("userId").size()
    linha = {"color": AZUL_ESCURO, "linewidth": 1.2}
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.boxplot(
        por_usuario,
        orientation="horizontal",
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": AZUL_CLARO,
                  "edgecolor": AZUL_ESCURO, "linewidth": 1.2},
        medianprops={"color": AZUL_ESCURO, "linewidth": 2},
        whiskerprops=linha,
        capprops=linha,
        flierprops={"marker": "o", "markersize": 4,
                    "markerfacecolor": AZUL,
                    "markeredgecolor": "white"},
    )
    # A escala logarítmica evita que a caixa fique espremida
    ax.set_xscale("log")
    ax.set_xticks([20, 35, 70, 168, 500, 1000, 2000])
    ax.xaxis.set_major_formatter(FuncFormatter(milhar))
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Quantidade de avaliações por usuário "
                  "(escala logarítmica)")
    ax.grid(axis="x", color=GRADE, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig("fig_boxplot.png")
    plt.close(fig)

gerar_boxplot(avaliacoes)
