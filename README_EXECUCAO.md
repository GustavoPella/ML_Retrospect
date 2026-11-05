# Retrospect — iNaturalist CRISP-DM

## Instalar dependências
```bash
pip install -r requirements.txt
https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz   
```

## Sequência de execução

```bash
# 1. Baixar o arquivo inaturalist-open-data-latest.tar.gz
# 1.1 Extrair
python 01_extrair.py

# 2. Explorar e filtrar para o Brasil (Fase 2 CRISP-DM)
python 02_explorar.py

# 3. Limpar e preparar os dados (Fase 3 CRISP-DM)
python 03_preparar.py

# 4. Treinar e comparar modelos (Fase 4 CRISP-DM)
python 04_modelar.py

# 5. Subir a API de predição (Fase 6 CRISP-DM)
python 05_api.py
# Acesse: http://localhost:8000/docs
```

## Exemplo de chamada à API
```bash
curl -X POST http://localhost:8000/api/prever-especie \
  -H "Content-Type: application/json" \
  -d '{"latitude":-25.4,"longitude":-49.2,"mes":9,"trimestre":3}'
```