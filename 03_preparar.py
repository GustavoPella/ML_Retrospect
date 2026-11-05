import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

SUBPASTA = "inaturalist_dados"
TAXA_CSV = os.path.join(SUBPASTA, "taxa.csv")

ICONIC_IDS = {
    "3":     "Aves",
    "20978": "Amphibia",
    "26036": "Reptilia",
    "40151": "Mammalia",
    "47120": "Actinopterygii",
    "47158": "Insecta",
    "47178": "Arachnida",
    "47115": "Mollusca",
    "47126": "Plantae",
    "47169": "Fungi",
}

print("=" * 60)
print("CORRECA0: derivando grupos taxonomicos via ancestry")
print("=" * 60)

df = pd.read_csv("observacoes_brasil.csv", low_memory=False)
print(f"\n[1] observacoes_brasil.csv: {len(df):,} registros")

df = df[df["quality_grade"] == "research"].copy()
df["taxon_id"] = pd.to_numeric(df["taxon_id"], errors="coerce")
df = df.dropna(subset=["taxon_id", "latitude", "longitude"])
df["taxon_id"] = df["taxon_id"].astype(int).astype(str)
print(f"    Research + lat/lon validos: {len(df):,}")

print(f"\n[2] Carregando taxa.csv...")
taxa = pd.read_csv(TAXA_CSV, sep="\t",
                   usecols=["taxon_id", "ancestry", "name"], dtype=str)

def extrair_grupo(ancestry):
    if pd.isna(ancestry):
        return None
    for parte in str(ancestry).split("/"):
        if parte in ICONIC_IDS:
            return ICONIC_IDS[parte]
    return None

taxa["iconic_taxon_name"] = taxa["ancestry"].apply(extrair_grupo)
cobertura = taxa["iconic_taxon_name"].notna().sum()
print(f"    Taxa com grupo identificado: {cobertura:,} / {len(taxa):,}")

taxa["taxon_id"] = pd.to_numeric(taxa["taxon_id"], errors="coerce")
taxa = taxa.dropna(subset=["taxon_id"])
taxa["taxon_id"] = taxa["taxon_id"].astype(int).astype(str)

df = df.merge(taxa[["taxon_id", "iconic_taxon_name"]], on="taxon_id", how="left")
df = df.dropna(subset=["iconic_taxon_name"])
print(f"\n[3] Apos merge com grupos: {len(df):,}")

df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
df = df.dropna(subset=["observed_on"])
df["mes"]       = df["observed_on"].dt.month
df["trimestre"] = df["observed_on"].dt.quarter
df = df[df["observed_on"].dt.year.between(2010, 2025)]

print(f"\n[4] Distribuicao dos grupos:")
dist = df["iconic_taxon_name"].value_counts()
for cls, cnt in dist.items():
    pct = cnt / len(df) * 100
    print(f"    {cls:<20} {cnt:>8,}  ({pct:5.1f}%)")

FEATURES = ["latitude", "longitude", "mes", "trimestre"]
if "positional_accuracy" in df.columns:
    FEATURES.append("positional_accuracy")

X = df[FEATURES].copy()
for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.fillna(X.median())

le = LabelEncoder()
y = le.fit_transform(df["iconic_taxon_name"])

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[5] Split: {len(X_train):,} treino | {len(X_test):,} teste")
print(f"    Classes: {le.classes_.tolist()}")

N_TREINO = min(200_000, len(X_train))
N_TESTE  = min(50_000,  len(X_test))
rng = np.random.default_rng(42)
Xtr = X_train.iloc[rng.choice(len(X_train), N_TREINO, replace=False)].values
ytr = y_train[rng.choice(len(y_train), N_TREINO, replace=False)]
Xte = X_test.iloc[rng.choice(len(X_test), N_TESTE, replace=False)].values
yte = y_test[rng.choice(len(y_test), N_TESTE, replace=False)]

print(f"\n{'=' * 60}")
print("FASE 4 - MODELAGEM: Random Forest + Gradient Boosting + MLP")
print("=" * 60)

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=50, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50, max_depth=4, random_state=42),
    "MLP (Rede Neural)": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=200,
        random_state=42, early_stopping=True, verbose=False),
}

resultados = {}
for nome, modelo in modelos.items():
    print(f"\n[Treinando] {nome}...")
    try:
        modelo.fit(Xtr, ytr)
        y_pred = modelo.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        f1  = f1_score(yte, y_pred, average="macro", zero_division=0)
        resultados[nome] = {"acc": acc, "f1": f1, "pred": y_pred, "obj": modelo}
        print(f"  Accuracy : {acc:.4f}  |  F1-macro : {f1:.4f}")
    except Exception as e:
        print(f"  Erro: {e}")

if resultados:
    melhor = max(resultados, key=lambda k: resultados[k]["f1"])
    print(f"\n{'=' * 60}")
    print(f"MELHOR MODELO: {melhor}")
    print(f"  Accuracy : {resultados[melhor]['acc']:.4f}")
    print(f"  F1-macro : {resultados[melhor]['f1']:.4f}")
    print(f"\nRelatorio completo:")
    print(classification_report(yte, resultados[melhor]["pred"],
                                 target_names=le.classes_, zero_division=0))
    joblib.dump(resultados[melhor]["obj"], "modelo_final.pkl")
    print("modelo_final.pkl salvo")

os.makedirs("graficos", exist_ok=True)
BG, VERDE, OCRE = '#FAF7EF', '#3D6B4F', '#C9A867'
ESCURO, CINZA, BORDA = '#1F2E23', '#5C6B5E', '#DCD2B4'
nomes = list(resultados.keys())
accs  = [resultados[n]["acc"] for n in nomes]
f1s   = [resultados[n]["f1"]  for n in nomes]
x     = np.arange(len(nomes))
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for s in ax.spines.values(): s.set_edgecolor(BORDA)
b1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color=VERDE, edgecolor="white")
b2 = ax.bar(x + 0.2, f1s,  0.35, label="F1-macro",  color=OCRE,  edgecolor="white")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", fontsize=9,
            color=ESCURO, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=10, color=CINZA)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score", color=CINZA)
ax.set_title("Comparacao de Modelos - iNaturalist Brasil",
             fontsize=12, color=ESCURO, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", color=BORDA, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graficos/E_comparacao_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("graficos/E_comparacao_modelos.png salvo")
print("\n[CONCLUIDO]")