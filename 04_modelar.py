import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)


print("=" * 60)
print("CRISP-DM — FASE 4: MODELAGEM")
print("=" * 60)

RANDOM_STATE = 42

MAX_TREINO = 300_000

MAX_TESTE = 100_000

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

le = joblib.load("label_encoder.pkl")


print(
    f"\nDados carregados: "
    f"{len(X_train):,} treino | "
    f"{len(X_test):,} teste"
)

print(
    f"Classes disponíveis: "
    f"{len(le.classes_):,}"
)

print("\n[1] Preparando amostras para modelagem...")


def amostra_estratificada(X, y, max_amostras, random_state=42):
    """
    Seleciona uma amostra estratificada mantendo,
    sempre que possível, todas as classes.
    """

    if len(X) <= max_amostras:
        return X.copy(), y.copy()

    dados = X.copy()
    dados["_target_temp_"] = y.to_numpy()

    n_classes = dados["_target_temp_"].nunique()

    por_classe = max(
        1,
        max_amostras // n_classes
    )

    partes = []

    for _, grupo in dados.groupby(
        "_target_temp_",
        sort=False
    ):
        n = min(len(grupo), por_classe)

        partes.append(
            grupo.sample(
                n=n,
                random_state=random_state
            )
        )

    resultado = pd.concat(
        partes,
        ignore_index=True
    )

    if len(resultado) < max_amostras:

        restantes = dados[
            ~dados.index.isin(resultado.index)
        ]

        quantidade = min(
            max_amostras - len(resultado),
            len(restantes)
        )

        if quantidade > 0:
            complemento = restantes.sample(
                n=quantidade,
                random_state=random_state
            )

            resultado = pd.concat(
                [resultado, complemento],
                ignore_index=True
            )

    y_resultado = resultado["_target_temp_"].copy()

    resultado = resultado.drop(
        columns=["_target_temp_"]
    )

    return resultado, y_resultado


X_train_model, y_train_model = amostra_estratificada(
    X_train,
    y_train,
    MAX_TREINO,
    RANDOM_STATE
)

X_test_model, y_test_model = amostra_estratificada(
    X_test,
    y_test,
    MAX_TESTE,
    RANDOM_STATE
)


print(
    f"    Treino utilizado: "
    f"{len(X_train_model):,}"
)

print(
    f"    Teste utilizado: "
    f"{len(X_test_model):,}"
)

print(
    f"    Classes no treino: "
    f"{y_train_model.nunique():,}"
)

print(
    f"    Classes no teste: "
    f"{y_test_model.nunique():,}"
)

modelos = {

    "Regressão Logística": LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2
    )

}

resultados = {}


for nome, modelo in modelos.items():

    print(
        f"\n[Treinando] {nome}..."
    )

    try:

        modelo.fit(
            X_train_model,
            y_train_model
        )

        y_pred = modelo.predict(
            X_test_model
        )

        acc = accuracy_score(
            y_test_model,
            y_pred
        )

        f1 = f1_score(
            y_test_model,
            y_pred,
            average="macro",
            zero_division=0
        )

        resultados[nome] = {
            "accuracy": acc,
            "f1_macro": f1,
            "modelo": modelo,
            "y_pred": y_pred
        }

        print(
            f"  Accuracy : {acc:.4f}"
        )

        print(
            f"  F1-macro : {f1:.4f}"
        )

    except Exception as e:

        print(
            f"  ⚠ Erro ao treinar {nome}:"
        )

        print(
            f"    {type(e).__name__}: {e}"
        )

if not resultados:

    raise RuntimeError(
        "Nenhum modelo conseguiu ser treinado."
    )

melhor_nome = max(
    resultados,
    key=lambda k: resultados[k]["f1_macro"]
)

melhor = resultados[melhor_nome]


print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

print(
    f"\n★ MELHOR MODELO: {melhor_nome}"
)

print(
    f"  F1-macro : "
    f"{melhor['f1_macro']:.4f}"
)

print(
    f"  Accuracy : "
    f"{melhor['accuracy']:.4f}"
)

print("\n  Relatório de classificação:")

classes_presentes = np.unique(
    np.concatenate(
        [
            y_test_model.to_numpy(),
            melhor["y_pred"]
        ]
    )
)

nomes_classes = le.inverse_transform(
    classes_presentes.astype(int)
)

print(
    classification_report(
        y_test_model,
        melhor["y_pred"],
        labels=classes_presentes,
        target_names=nomes_classes,
        zero_division=0
    )
)

joblib.dump(
    melhor["modelo"],
    "modelo_final.pkl"
)

print(
    "\n✓ Modelo salvo: modelo_final.pkl"
)

os.makedirs(
    "graficos",
    exist_ok=True
)


nomes = list(resultados.keys())

accs = [
    resultados[n]["accuracy"]
    for n in nomes
]

f1s = [
    resultados[n]["f1_macro"]
    for n in nomes
]

x = np.arange(
    len(nomes)
)

BG = "#FAF7EF"
VERDE = "#3D6B4F"
OCRE = "#C9A867"
CINZA = "#5C6B5E"
ESCURO = "#1F2E23"
BORDA = "#DCD2B4"


fig, ax = plt.subplots(
    figsize=(9, 5)
)

fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


for spine in ax.spines.values():
    spine.set_edgecolor(BORDA)


bars1 = ax.bar(
    x - 0.2,
    accs,
    0.35,
    label="Accuracy",
    color=VERDE,
    edgecolor="white"
)


bars2 = ax.bar(
    x + 0.2,
    f1s,
    0.35,
    label="F1-macro",
    color=OCRE,
    edgecolor="white"
)


for b in bars1 + bars2:

    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.005,
        f"{b.get_height():.3f}",
        ha="center",
        fontsize=9,
        color=ESCURO,
        fontweight="bold"
    )


ax.set_xticks(x)

ax.set_xticklabels(
    nomes,
    fontsize=10,
    color=CINZA
)

ax.set_ylim(
    0,
    1.08
)

ax.set_ylabel(
    "Score",
    color=CINZA
)

ax.set_title(
    "Comparação de Modelos — iNaturalist Brasil",
    fontsize=13,
    color=ESCURO,
    fontweight="bold"
)

ax.legend(
    fontsize=10
)

ax.grid(
    axis="y",
    color=BORDA,
    linestyle="--",
    linewidth=0.7
)

ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    "graficos/E_comparacao_modelos.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "✓ Gráfico salvo: "
    "graficos/E_comparacao_modelos.png"
)

print(
    "\n[CONCLUÍDO] Fase 4 — Modelagem"
)

print(
    "Próximo passo: "
    "05_api.py (Fase 6 — Implantação)"
)