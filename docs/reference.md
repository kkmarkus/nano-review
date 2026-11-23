# Referência de comandos

Detalhes de uso que não cabem no README: pipeline do DVC, experimentos, rubrica, API e Docker.
Todos os comandos rodam na raiz do projeto, com o `.venv` ativo (`source .venv/bin/activate`).

## 1. Dados e modelos (DVC)

[`data/`](../data/) e [`models/`](../models/) são versionados pelo **DVC**, não pelo Git. O Git guarda só os arquivos `.dvc`
e o [`dvc.lock`](../dvc.lock) (ponteiros com o hash do conteúdo). Os arquivos grandes ficam no cache do DVC e no
remote.

```text
data/raw/        download original (GitHub da B2W, via `dvc import-url`); nunca editar à mão
data/processed/  limpo, deduplicado, sem dados pessoais
data/curated/    splits prontos (train, validation, test), feitos por produto
models/          tokenizador, juiz de sentimento e Transformer
```

As pastas nascem quando o primeiro arquivo chega, sem `.gitkeep`. Se o Git rastreasse um arquivo
dentro de uma pasta, o `dvc add` dessa pasta falharia com "already tracked by SCM".

O remote padrão é uma pasta local fora do repositório (`~/dvc-storage/nano-review`):

```bash
dvc pull   # baixa do remote os dados que o Git aponta
dvc push   # envia para o remote os dados novos
```

Esse remote só existe nesta máquina. Num clone novo, o `dvc pull` falha com "missing cache files";
o caminho é baixar o CSV da fonte com `dvc update` e gerar o resto com `dvc repro` (README, seção
"Reproduzir o pipeline").

### Estágios do pipeline

`data/processed` e `data/curated` não são baixados nem editados: o pipeline do [`dvc.yaml`](../dvc.yaml) os
**gera**, com os parâmetros do [`params.yaml`](../params.yaml). O DVC só refaz um estágio quando a entrada, o código
ou um parâmetro dele muda.

```bash
dvc repro          # refaz só o que mudou
dvc metrics show   # funil da limpeza e tamanhos do split
dvc plots show     # curva de treino e validação (reports/metrics/train_log.csv)
```

| Estágio | Entrada | Saída |
|---|---|---|
| `clean` | `data/raw/B2W-Reviews01.csv` | `data/processed/reviews.parquet`, [`reports/metrics/cleaning.json`](../reports/metrics/cleaning.json) |
| `split` | `data/processed/reviews.parquet` | `data/curated/{train,validation,test}.parquet`, [`reports/metrics/split.json`](../reports/metrics/split.json) |
| `judge` | `train`, `validation` | `models/judge.joblib`, [`reports/metrics/judge.json`](../reports/metrics/judge.json) |
| `baseline` | `train`, `validation`, juiz | [`reports/metrics/baseline.json`](../reports/metrics/baseline.json), [`reports/samples/baseline.jsonl`](../reports/samples/baseline.jsonl) |
| `tokenizer` | `train` (a validação só é medida) | `models/tokenizer.json`, [`reports/metrics/tokenizer.json`](../reports/metrics/tokenizer.json) |
| `train` | `train`, `validation`, tokenizador | `models/transformer.pt`, [`reports/metrics/train.json`](../reports/metrics/train.json), `train_log.csv`, [`reports/figures/train_curve.png`](../reports/figures/train_curve.png) |
| `evaluate` | checkpoint, tokenizador, juiz, `train`, `validation` | [`reports/metrics/transformer_eval.json`](../reports/metrics/transformer_eval.json) (H1 a H6), [`reports/samples/transformer_eval.jsonl`](../reports/samples/transformer_eval.jsonl) |
| `decoding` | checkpoint, tokenizador, juiz, `train` | [`reports/metrics/decoding.json`](../reports/metrics/decoding.json), [`reports/samples/decoding.jsonl`](../reports/samples/decoding.jsonl) (Etapa 4.1) |
| `inference_check` | checkpoint, tokenizador, juiz, `train` | [`reports/metrics/inference.json`](../reports/metrics/inference.json), [`reports/samples/inference.jsonl`](../reports/samples/inference.jsonl) (Etapa 4.2) |
| `final_eval` | tudo acima, mais `test.parquet` | [`reports/metrics/final_eval.json`](../reports/metrics/final_eval.json), [`reports/samples/final_eval.jsonl`](../reports/samples/final_eval.jsonl), [`reports/samples/test_real_control.jsonl`](../reports/samples/test_real_control.jsonl) (Etapa 4.3; rodou uma vez e está congelado, `frozen: true`) |

O treino do Transformer, na configuração final, leva ~66 minutos em CPU (16 núcleos). A avaliação
leva ~4 minutos.

## 2. Experimentos e MLflow (Etapa 3)

Cada experimento muda **uma** variável, e a hipótese dele é escrita em [`docs/experiments.md`](../docs/experiments.md)
**antes** de rodar. O DVC guarda a configuração de cada experimento, e o MLflow mostra a comparação:

```bash
dvc exp run -n e2-lr-1e-3 --set-param train.learning_rate=0.001   # treina e avalia com um parâmetro trocado
dvc exp show                                                      # parâmetros e métricas de cada experimento
mlflow ui --host 127.0.0.1 --port 5000                            # painel em http://127.0.0.1:5000
```

- O nome dado com `-n` vira o nome do run no MLflow, e assim os dois se cruzam.
- O `train` grava no run os parâmetros, a curva passo a passo e os artefatos. O `evaluate`
  acrescenta H1 a H6 ao **mesmo** run.
- O treino para sozinho (early stopping) quando a validação passa `train.patience` avaliações sem
  melhorar. O checkpoint é sempre o da melhor validação.
- Rode um experimento por vez, na pasta do projeto: o `mlflow.db` e a pasta `mlruns/` ficam aqui,
  fora do Git. Os números que contam continuam versionados em [`reports/metrics/`](../reports/metrics/).
- Para promover um experimento ao workspace: `dvc exp apply <nome>`.
- Como o MLflow fica fora do Git, as runs são exportadas para [`reports/experiments/`](../reports/experiments/): `runs.csv`
  (parâmetros e métricas finais) e `curves.csv` (curvas passo a passo). Depois de rodar
  experimentos novos:

```bash
python -m nano_review.training.export_runs --out-dir reports/experiments
```

## 3. Rubrica humana e análise de erros

A rubrica não é estágio do DVC. A planilha pontuada é trabalho humano, e um `dvc repro` não pode
regerá-la. As regras de pontuação estão em [`docs/rubric.md`](../docs/rubric.md). A chave de cada lote só é aberta
depois que a planilha inteira foi pontuada.

### Lote v1 (Etapa 3, rubrica 1.0)

```bash
# 1. Monta o lote cego: a planilha sem a origem e, separada, a chave. Recusa sobrescrever um lote.
python -m nano_review.evaluation.rubric batch --trigram reports/samples/baseline.jsonl \
    --transformer reports/samples/transformer_eval.jsonl \
    --validation data/curated/validation.parquet --tokenizer models/tokenizer.json \
    --block-size 128 --out-dir reports/rubric --per-rating 20 --seed 2026
# 2. Depois de pontuar reports/rubric/sheet_v1.csv: abre a chave e calcula o H3.
python -m nano_review.evaluation.rubric summarize --sheet reports/rubric/sheet_v1.csv \
    --key reports/rubric/key_v1.csv --metrics reports/metrics/rubric.json
# 3. Mede as causas das amostras ruins, a partir da classificação feita à mão.
python -m nano_review.evaluation.error_analysis --sheet reports/rubric/sheet_v1.csv \
    --key reports/rubric/key_v1.csv --types reports/rubric/error_types_v1.csv \
    --train data/curated/train.parquet --metrics reports/metrics/error_analysis.json
```

### Lote v2 (Etapa 4.3, rubrica 1.1)

O lote v2 lê as fontes de arquivos `.jsonl` e exclui os textos já lidos no v1
([`docs/final_test.md`](../docs/final_test.md)). O `--version` do `summarize` precisa ser o da rubrica usada (o padrão é
1.0).

```bash
python -m nano_review.evaluation.rubric batch-jsonl \
    --source transformer=reports/samples/inference.jsonl@t1.0-k0 \
    --source transformer_k50=reports/samples/inference.jsonl@t1.0-k50 \
    --source real=reports/samples/test_real_control.jsonl \
    --exclude reports/rubric/sheet_v1.csv --per-rating 10 --seed 2029 \
    --out-dir reports/rubric --suffix _v2 --version 1.1
python -m nano_review.evaluation.rubric summarize --sheet reports/rubric/sheet_v2.csv \
    --key reports/rubric/key_v2.csv --metrics reports/metrics/rubric_v2.json --version 1.1
```

## 4. API

A API serve o gerador do produto (`nano_review.inference.generator`): a decodificação escolhida na
Etapa 4.1 e os filtros da 4.2 ([`docs/inference.md`](../docs/inference.md)). O modelo carrega uma vez, quando a API sobe.

```bash
uvicorn nano_review.api.app:app --host 127.0.0.1 --port 8000
```

O `127.0.0.1` deixa a API visível só nesta máquina. Ela não tem autenticação nem limite de pedidos,
porque foi feita para a demonstração e para testes locais. No Docker, ela escuta em `0.0.0.0`
**dentro** do contêiner, e quem decide o que fica exposto é a porta publicada.

| Rota | O que faz |
|---|---|
| `GET /` | Página de demonstração: escolher a nota e a quantidade, gerar e copiar as resenhas (o estilo e o script dela ficam em `/static`) |
| `GET /docs` | Documentação automática, onde dá para testar cada rota |
| `GET /health` | Se o modelo está carregado, o número de parâmetros, o hash do checkpoint e os padrões |
| `POST /generate` | As resenhas para uma nota |

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "n": 3}'
```

| Campo do pedido | Aceito | Padrão |
|---|---|---|
| `rating` | inteiro de 1 a 5 | obrigatório |
| `n` | inteiro de 1 a 20 | 1 |
| `temperature` | de 0,1 a 2 | 1,0 |
| `top_k` | inteiro de 0 ao tamanho do vocabulário (8.000); 0 pede "sem top-k" | sem top-k |
| `seed` | inteiro de 0 a 2⁶³ − 1; o mesmo pedido com a mesma semente repete o texto | sorteada (abaixo de 2⁵³, para o JavaScript da página não arredondar), e devolvida na resposta |

- **422, pedido inválido:** tipo errado, fora do intervalo ou campo que não existe. O `detail` vem
  em dois formatos: uma lista (`type`, `loc`, `msg`), quando o formato do pedido está errado, ou um
  texto, quando o pedido é bem formado mas o gerador o recusa (um top-k maior que o vocabulário).
- **503, tentativas esgotadas:** quando as 5 tentativas de um texto caem nos filtros, o pedido
  **inteiro** falha, com os motivos, e as resenhas já geradas são descartadas. Com o padrão, isso
  não aconteceu em 2.000 resenhas (1.000 na Etapa 4.2 e 1.000 no teste final), nem nas 1.000 com
  top-k 50. Longe do padrão, acontece: em 20 pedidos de uma resenha, 2 falharam com temperatura
  0,1 e 1 com temperatura 2,0. Num pedido de várias resenhas, a chance de falha cresce com o número
  delas.
- **Toda resposta traz `"synthetic": true` e o aviso de que o texto é sintético.** Ele serve para
  testar software e não pode ser publicado como avaliação de um cliente real ([`docs/charter.md`](../docs/charter.md),
  seção 9).
- **Os caminhos do modelo** vêm das variáveis de ambiente `NANO_REVIEW_CHECKPOINT` e
  `NANO_REVIEW_TOKENIZER`, com o padrão `models/transformer.pt` e `models/tokenizer.json`.

## 5. Docker

A imagem traz a API, a página e o modelo final, e roda em qualquer máquina com Docker, sem `.venv`,
DVC nem dados. Os comandos de build e execução estão no README. Detalhes:

- **Porta ocupada:** se a 8000 já estiver em uso (por exemplo, pela API rodando no `.venv`), pare o
  outro processo ou use `make docker-run PORT=8001`. Para parar o contêiner: `Ctrl+C` ou
  `docker stop nano-review`.
- **Sem internet:** a página e a API funcionam offline. O `/docs` não: a documentação automática do
  FastAPI carrega o estilo e o script de uma CDN, e fica em branco sem internet.
- **O que entra na imagem:** o [`.dockerignore`](../.dockerignore) é uma lista do que **pode** entrar (o [`src/`](../src/), o
  [`pyproject.toml`](../pyproject.toml), o [`requirements-api.txt`](../requirements-api.txt) e os dois arquivos do modelo). Dados, `.git`, `.venv`,
  MLflow e relatórios ficam de fora.
- **As dependências:** o torch CPU 2.15.0 e as bibliotecas do [`requirements-api.txt`](../requirements-api.txt), todas nas
  versões exatas do `.venv`, inclusive as indiretas. Elas incluem o joblib e o scikit-learn, porque
  o gerador importa uma função de `evaluation/metrics.py`, e esse módulo importa o juiz de
  sentimento.
- **O tamanho:** ~2 GB. O torch sozinho ocupa ~970 MB, as outras bibliotecas ~420 MB e o modelo
  22 MB.
- **A segurança:** a API roda como o usuário `app`, sem privilégios de root.
- **A saúde:** um `HEALTHCHECK` chama o `/health` a cada 30 s, e o `docker ps` mostra o contêiner
  como `healthy`. A API sobe em uns 5 s.

## 6. Estrutura completa do repositório

```text
nano-review/
├── src/nano_review/        # código do pacote Python
│   ├── data/               # limpeza, split por produto, Dataset e DataLoader
│   ├── tokenization/       # tokenizador BPE e um BPE didático escrito à mão
│   ├── models/             # bigrama neural e Transformer (atenção à mão)
│   ├── training/           # laço de treino, checkpoint e export do MLflow
│   ├── baseline/           # juiz de sentimento e geradores de n-gramas
│   ├── evaluation/         # métricas, avaliação, rubrica, erros, decodificação, teste
│   ├── inference/          # sorteio (temperatura, top-k) e gerador com filtros
│   ├── api/                # API FastAPI e a página de demonstração
│   └── reproducibility.py  # set_seed: mesma semente, mesmo resultado
├── tests/                  # testes automatizados (make check)
├── notebooks/              # EDA, tokenização, forward/backward e atenção
├── docs/                   # charter, manifesto, rubrica, experimentos, erros, inferência, teste
├── data/                   # raw, processed e curated (DVC)
├── models/                 # tokenizador, juiz e Transformer (DVC)
├── reports/
│   ├── metrics/            # métricas do pipeline e da rubrica (JSON)
│   ├── figures/            # figuras dos notebooks e do treino
│   ├── samples/            # amostras geradas e o controle real do lote v2
│   ├── experiments/        # runs do MLflow exportadas (CSV)
│   └── rubric/             # planilha cega, chave e tipos de erro
├── configs/                # reservada, vazia por enquanto
├── params.yaml             # parâmetros de todos os estágios
├── dvc.yaml                # estágios do pipeline
├── dvc.lock                # hashes da última execução de cada estágio
├── requirements.txt        # dependências do projeto
├── requirements-dev.txt    # ferramentas: ruff, mypy, pytest
├── requirements-api.txt    # só o que a API importa, para o Docker
├── Dockerfile              # imagem da API com o modelo final
├── LICENSE                 # MIT, para o código
├── LICENSE-DATA.md         # CC BY-NC-SA 4.0, para dados, modelos e relatórios
├── .dockerignore           # o que pode entrar na imagem
└── Makefile                # make check, docker-build, docker-run
```
