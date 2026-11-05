import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

PASTA       = "inaturalist_dados"
OBS_CSV     = os.path.join(PASTA, "observations.csv")
TAXA_CSV    = os.path.join(PASTA, "taxa.csv")
SAIDA_BR    = "observacoes_brasil.csv"
CHUNK_SIZE  = 500_000

LAT_MIN, LAT_MAX = -33.8, 5.3
LON_MIN, LON_MAX = -73.9, -34.7

BG     = '#FAF7EF'
VERDE  = '#3D6B4F'
OCRE   = '#C9A867'
CINZA  = '#5C6B5E'
ESCURO = '#1F2E23'
BORDA  = '#DCD2B4'

print("=" * 60)
print("CRISP-DM — FASE 2: ENTENDIMENTO DOS DADOS")
print("iNaturalist Open Data — Filtrado para o Brasil")
print("=" * 60)

print("\n[1] Lendo cabeçalho de observations.csv...")
amostra = pd.read_csv(OBS_CSV, sep="\t", nrows=5, compression="infer",
                      low_memory=False)
print(f"Colunas disponíveis ({len(amostra.columns)}):")
for col in amostra.columns:
    print(f"  - {col}")

COLUNAS = [
    "observation_uuid", "observed_on", "latitude", "longitude",
    "positional_accuracy", "taxon_id", "quality_grade",
    "num_identification_agreements", "num_identification_disagreements",
    "captive_cultivated",
]
COLUNAS_EXISTENTES = [c for c in COLUNAS if c in amostra.columns]

print(f"\n[2] Filtrando observações do Brasil em chunks de {CHUNK_SIZE:,}...")
chunks_brasil = []
total_lidas   = 0
total_brasil  = 0

for chunk in pd.read_csv(OBS_CSV, sep="\t", chunksize=CHUNK_SIZE,
                          usecols=COLUNAS_EXISTENTES,
                          dtype={"taxon_id": "str",
                                 "observation_uuid": "str",
                                 "quality_grade": "str",
                                 "captive_cultivated": "str"},
                          compression="infer", low_memory=False):

    chunk["latitude"]  = pd.to_numeric(chunk["latitude"],  errors="coerce")
    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")

    filtro = (
        chunk["latitude"].between(LAT_MIN, LAT_MAX) &
        chunk["longitude"].between(LON_MIN, LON_MAX)
    )
    br = chunk[filtro]
    chunks_brasil.append(br)
    total_lidas  += len(chunk)
    total_brasil += len(br)
    print(f"  Lidas: {total_lidas:>10,}  |  Brasil acumulado: {total_brasil:>8,}")

df = pd.concat(chunks_brasil, ignore_index=True)
df.to_csv(SAIDA_BR, index=False)
print(f"\n✓ Salvo: {SAIDA_BR}  ({len(df):,} registros, "
      f"{os.path.getsize(SAIDA_BR)/1e6:.1f} MB)")

print("\n[3] Carregando taxa.csv...")
taxa = pd.read_csv(
    TAXA_CSV,
    sep="\t",
    compression="infer",
    usecols=[
        "taxon_id",
        "ancestry",
        "rank_level",
        "rank",
        "name",
        "active"
    ],
    dtype=str,
    low_memory=False
)
taxa["taxon_id"] = taxa["taxon_id"].astype(str)
df["taxon_id"]   = df["taxon_id"].astype(str)

df = df.merge(taxa, on="taxon_id", how="left")
print(f"  Merge concluído. {df['name'].notna().sum():,} obs com espécie identificada.")

print("\n[4] ESTATÍSTICAS GERAIS")
print(f"  Total de observações no Brasil : {len(df):,}")
print(f"  Período                        : {df['observed_on'].min()} → {df['observed_on'].max()}")
print(f"  Espécies únicas                : {df['taxon_id'].nunique():,}")
print(f"  Qualidade 'research'     : {(df['quality_grade']=='research').sum():,}")
print(f"\n  Valores ausentes por coluna:")
print(df.isnull().sum().to_string())
print(f"\n  Distribuição por qualidade:")
print(df["quality_grade"].value_counts().to_string())
print(f"\n  Distribuição por grupo taxonômico:")

os.makedirs("graficos", exist_ok=True)

def fig_base(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDA)
    ax.tick_params(colors=CINZA, labelsize=10)
    return fig, ax

fig, ax = fig_base(8, 5)
qg = df["quality_grade"].value_counts()
ax.bar(qg.index, qg.values, color=[VERDE, OCRE, CINZA][:len(qg)],
       edgecolor="white", width=0.55)
for i, (idx, val) in enumerate(qg.items()):
    ax.text(i, val + qg.max()*0.01, f"{val:,}",
            ha="center", fontsize=10, color=ESCURO, fontweight="bold")
ax.set_title("Qualidade das Observações — Brasil", fontsize=13,
             color=ESCURO, fontweight="bold")
ax.set_ylabel("Número de observações", color=CINZA)
ax.grid(axis="y", color=BORDA, linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graficos/A_qualidade.png", dpi=150, bbox_inches="tight")
plt.close()



df["ano"] = pd.to_datetime(df["observed_on"], errors="coerce").dt.year
serie = df.groupby("ano").size().reset_index(name="obs")
serie = serie[serie["ano"].between(2010, 2025)]
fig, ax = fig_base(10, 5)
ax.fill_between(serie["ano"], serie["obs"], alpha=0.25, color=VERDE)
ax.plot(serie["ano"], serie["obs"], color=VERDE, linewidth=2.5, marker="o", ms=5)
ax.set_title("Crescimento das Observações ao Longo do Tempo — Brasil",
             fontsize=13, color=ESCURO, fontweight="bold")
ax.set_xlabel("Ano", color=CINZA)
ax.set_ylabel("Observações", color=CINZA)
ax.grid(color=BORDA, linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graficos/C_serie_temporal.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = fig_base(9, 7)
amostra_mapa = df[df["quality_grade"] == "research"].sample(
    min(50_000, len(df)), random_state=42)
ax.scatter(amostra_mapa["longitude"], amostra_mapa["latitude"],
           alpha=0.05, s=0.8, color=VERDE)
ax.set_title("Distribuição Geográfica das Observações — Brasil",
             fontsize=13, color=ESCURO, fontweight="bold")
ax.set_xlabel("Longitude", color=CINZA)
ax.set_ylabel("Latitude", color=CINZA)
ax.grid(color=BORDA, linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.savefig("graficos/D_mapa.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ Gráficos salvos em ./graficos/")
print("\n[CONCLUÍDO] Fase 2 — Entendimento dos Dados")
