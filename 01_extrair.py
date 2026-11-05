import tarfile
import os

ARQUIVO = "inaturalist-open-data-latest.tar.gz"
DESTINO = "inaturalist_dados"

os.makedirs(DESTINO, exist_ok=True)

print(f"Extraindo {ARQUIVO}...")
with tarfile.open(ARQUIVO, "r:gz") as tar:
    membros = tar.getmembers()
    print(f"Arquivos encontrados: {[m.name for m in membros]}")
    tar.extractall(path=DESTINO)
    print("Extração concluída.")
    for m in membros:
        caminho = os.path.join(DESTINO, m.name)
        if os.path.exists(caminho):
            tamanho_gb = os.path.getsize(caminho) / 1e9
            print(f"  {m.name}: {tamanho_gb:.2f} GB")
