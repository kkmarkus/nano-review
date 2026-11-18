# nano-review

Transformer pequeno, **treinado do zero** (sem modelo pré-treinado), que gera resenhas de
produto em português a partir de uma nota de 1 a 5. Projeto integrador individual.

- **Dados:** B2W-Reviews01 (licença CC BY-NC-SA 4.0; origem e esquema ficam no manifesto em `docs/`).
- **Usuário:** desenvolvedor que precisa de textos sintéticos para testar telas, moderação ou
  classificadores de sentimento. A IA não serve para publicar avaliações falsas.

## Status

- Etapa 0 concluída: esqueleto do repositório, ambiente e porta de qualidade (`make check`).
- Etapa 1 concluída: charter com a hipótese e os limiares (`docs/charter.md`), dados brutos com
  manifesto (`docs/data_manifest.md`), EDA (`notebooks/01_eda.ipynb`), limpeza e split por produto,
  rubrica humana (`docs/rubric.md`), juiz de sentimento e baseline de n-gramas (`dvc.yaml`).
- Etapa 2 concluída: tokenizador BPE treinado só no treino (`notebooks/02_tokenization.ipynb`),
  DataLoader com lotes agrupados por comprimento, bigrama neural com forward e backward conferidos à
  mão (`notebooks/03_forward_backward.ipynb`), Transformer com a atenção escrita à mão, primeiro
  treino e curva (`reports/figures/train_curve.png`) e análise da atenção
  (`notebooks/04_attention.ipynb`).

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
  data/            limpeza (clean.py), split por produto (split.py), Dataset e DataLoader (dataset.py)
  tokenization/    tokenizador BPE (bpe.py) e BPE didático escrito à mão (bpe_demo.py)
  models/          loss, bigrama neural e Transformer com atenção escrita à mão
  training/        laço de treino, avaliação, checkpoint e curva (train.py)
  baseline/        juiz de sentimento (judge.py) e geradores de n-gramas (ngram.py, evaluate.py)
  evaluation/      métricas do charter, as mesmas para o baseline e o Transformer
tests/             testes automatizados
docs/              charter, manifesto de dados e relatório
notebooks/         exploração de dados (não é o pipeline: as regras de verdade ficam em src/)
reports/figures/   figuras geradas pelos notebooks
reports/metrics/   métricas geradas pelo pipeline (limpeza, split, juiz, baseline)
reports/samples/   amostras geradas, para a rubrica e a análise de erros
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
| `judge` | `train`, `validation` | `models/judge.joblib`, `reports/metrics/judge.json` |
| `baseline` | `train`, `validation`, juiz | `reports/metrics/baseline.json`, `reports/samples/baseline.jsonl` |
| `tokenizer` | `train` (a validação só é medida) | `models/tokenizer.json`, `reports/metrics/tokenizer.json` |
| `train` | `train`, `validation`, tokenizador | `models/transformer.pt`, `reports/metrics/train.json`, `train_log.csv`, `reports/figures/train_curve.png` |

O treino do Transformer leva ~13 minutos em CPU (8 threads). `dvc plots show` desenha a curva de
treino e validação a partir de `reports/metrics/train_log.csv`.

O split é **por produto** (nenhum produto aparece em dois conjuntos), com 80/10/10 das resenhas.
**`test.parquet` não é lido por nada até a Etapa 4**: o teste é aberto uma única vez, no fim.
