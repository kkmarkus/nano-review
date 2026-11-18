# nano-review

Transformer pequeno, **treinado do zero** (sem modelo pré-treinado), que gera resenhas de
produto em português a partir de uma nota de 1 a 5. Projeto integrador individual.

- **Dados:** B2W-Reviews01 (licença CC BY-NC-SA 4.0; origem e esquema ficam no manifesto em `docs/`).
- **Usuário:** desenvolvedor que precisa de textos sintéticos para testar telas, moderação ou
  classificadores de sentimento. A IA não serve para publicar avaliações falsas.

## Status

- Etapa 0 concluída: esqueleto do repositório, ambiente e porta de qualidade (`make check`).
- Etapa 1 em andamento: charter (`docs/charter.md`), dados brutos com manifesto
  (`docs/data_manifest.md`), EDA (`notebooks/01_eda.ipynb`) e pipeline de limpeza e split por
  produto (`dvc.yaml`). Falta o baseline.

## Como preparar o ambiente (WSL/Ubuntu, Python 3.12)

Mantenha o projeto dentro do sistema de arquivos do WSL (por exemplo `~/workspace/nano-review`),
fora de `/mnt/c` e de pastas sincronizadas pelo OneDrive.

```bash
# Se o comando abaixo reclamar de ensurepip/venv: sudo apt install python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# O torch (build CPU) vem do índice próprio da PyTorch; instalar antes do resto.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt
make check
```

`make check` roda `ruff` (estilo), `mypy --strict` (tipos) e `pytest` (testes). Ele precisa
passar antes de qualquer commit ou entrega.

## Estrutura

```
src/nano_review/   código do pacote (cada subpasta nasce na etapa que a usa)
  data/            limpeza (clean.py) e split por produto (split.py)
tests/             testes automatizados
docs/              charter, manifesto de dados e relatório
notebooks/         exploração de dados (não é o pipeline: as regras de verdade ficam em src/)
reports/figures/   figuras geradas pelos notebooks
reports/metrics/   métricas geradas pelo pipeline (funil da limpeza, tamanhos do split)
configs/           hiperparâmetros do modelo em YAML
params.yaml        parâmetros do pipeline de dados (lidos pelo DVC)
dvc.yaml           estágios do pipeline de dados
```

## Dados e modelos (DVC)

`data/` e `models/` são versionados pelo **DVC**, não pelo Git. O Git guarda só os arquivos `.dvc`
(um ponteiro com o hash do conteúdo); os arquivos grandes ficam no cache do DVC e no remote.

```
data/raw/        download original (GitHub da B2W, via `dvc import-url`); nunca editar à mão
data/processed/  limpo, deduplicado, sem dados pessoais
data/curated/    splits prontos (train, validation, test), feitos por produto
models/          checkpoints treinados
```

As pastas nascem quando o primeiro arquivo chega (sem `.gitkeep`: se o Git rastreasse um arquivo
dentro de uma pasta, o `dvc add` dessa pasta falharia com "already tracked by SCM").

O remote padrão é uma pasta local fora do repositório (`~/dvc-storage/nano-review`):

```bash
mkdir -p ~/dvc-storage/nano-review   # só na primeira vez em uma máquina nova
dvc pull                             # baixa do remote os dados que o Git aponta
dvc push                             # envia para o remote os dados novos
```

### Pipeline de dados

`data/processed` e `data/curated` não são baixados nem editados: são **gerados** pelo pipeline do
`dvc.yaml`, com os parâmetros do `params.yaml`.

```bash
source .venv/bin/activate   # os comandos do pipeline usam o python do .venv
dvc repro                   # refaz só o que mudou (código, dado de entrada ou parâmetro)
dvc metrics show            # funil da limpeza e tamanhos do split
```

| Estágio | Entrada | Saída |
|---|---|---|
| `clean` | `data/raw/B2W-Reviews01.csv` | `data/processed/reviews.parquet`, `reports/metrics/cleaning.json` |
| `split` | `data/processed/reviews.parquet` | `data/curated/{train,validation,test}.parquet`, `reports/metrics/split.json` |

O split é **por produto** (nenhum produto aparece em dois conjuntos), com 80/10/10 das resenhas.
**`test.parquet` não é lido por nada até a Etapa 4**: o teste é aberto uma única vez, no fim.
