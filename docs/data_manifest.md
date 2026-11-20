# Manifesto de dados

> Registra **de onde** veio cada arquivo de dados, **sob qual licença** e **como conferir** que é o mesmo
> arquivo. Este manifesto é escrito antes da EDA, para que a procedência fique documentada antes de qualquer
> decisão de limpeza.

## B2W-Reviews01 (bruto)

| Campo | Valor |
|---|---|
| Nome | B2W-Reviews01 |
| Fonte primária | <https://github.com/americanas-tech/b2w-reviews01> (repositório arquivado, só leitura desde 2022) |
| Commit fixado | `4639429ec698d7821fc99a0bc665fa213d9fcd5a` (14/09/2021, o último do repositório) |
| URL do download | `https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv` |
| Como foi baixado | `dvc import-url <URL acima> data/raw/B2W-Reviews01.csv` (a origem e o hash ficam em `data/raw/B2W-Reviews01.csv.dvc`) |
| Data do download | 2026-11-17 (18:06 UTC) |
| Arquivo local | `data/raw/B2W-Reviews01.csv` (nunca editar à mão) |
| Tamanho | 49.453.175 bytes |
| SHA-256 | `821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38` |
| MD5 (o hash que o DVC usa) | `3d6bb9c7546fe32bb4dff53eaeedafda` |
| Formato | CSV, UTF-8 sem BOM, separador **vírgula**, aspas `"` como delimitador de texto, 1 linha de cabeçalho |
| Registros | 132.373 resenhas (132.374 linhas físicas; nenhum texto tem quebra de linha) |
| Período | 2018-01-01 a 2018-05-31 (Americanas.com) |

Para conferir o arquivo em qualquer máquina:

```bash
sha256sum data/raw/B2W-Reviews01.csv
```

### Divergência entre a documentação e o arquivo

O README do repositório diz que o separador é ponto e vírgula. O arquivo do commit fixado usa **vírgula**:
lido com `sep=";"`, o pandas encontra uma única coluna. Vale o que o arquivo mostra.

## Licença e atribuição

- **Licença: CC BY-NC-SA 4.0** (<https://creativecommons.org/licenses/by-nc-sa/4.0/>), conforme o
  repositório original da B2W Digital.
  - **BY:** dar crédito à B2W Digital e citar o artigo abaixo.
  - **NC:** só uso não comercial. O nano-review é um projeto acadêmico.
  - **SA:** qualquer dado derivado (por exemplo `data/processed` e `data/curated`) segue a mesma licença.
- **Espelho no Hugging Face** (<https://huggingface.co/datasets/ruanchaves/b2w-reviews01>): o card de lá
  declara CC BY 4.0. Um espelho não pode afrouxar a licença do original, então **vale a CC BY-NC-SA 4.0**.
  O espelho serve só como referência para o esquema das colunas.
- **Citação:** Real, L.; Oshiro, M.; Mafra, A. (2019). *B2W-Reviews01: an open product reviews corpus*.
  Symposium in Information and Human Language Technology (STIL).

## Colunas

Contagem de nulos medida no arquivo bruto (132.373 linhas).

| Coluna | Tipo | Nulos | Papel no projeto |
|---|---|---:|---|
| `submission_date` | texto (`%Y-%m-%d %H:%M:%S`) | 0 | Contexto para a EDA. Não é entrada do modelo. |
| `reviewer_id` | texto | 0 | **Descartar (dado pessoal).** Identifica o avaliador. |
| `product_id` | texto | 0 | **Chave do split por produto.** Não é entrada do modelo. |
| `product_name` | texto | 84 | Contexto para a EDA. Pode conter nome próprio de loja ou marca. |
| `product_brand` | texto | 91.391 | Contexto para a EDA (69% nulo). |
| `site_category_lv1` | texto | 6 | Extensão futura: condicionamento por categoria. |
| `site_category_lv2` | texto | 4.013 | Extensão futura. |
| `review_title` | texto | 302 | Descartado na limpeza (decisão da EDA): só o `review_text` é alvo da geração. |
| `overall_rating` | inteiro 1 a 5 | 0 | **Condicionamento** (`<nota_k>`) e rótulo do classificador. |
| `recommend_to_a_friend` | `Yes`/`No` | 18 | Checagem de consistência com a nota na EDA. |
| `review_text` | texto | 3.275 | **Alvo da geração.** As linhas sem texto saem na limpeza. |
| `reviewer_birth_year` | número | 5.984 | **Descartar (dado pessoal).** |
| `reviewer_gender` | `F`/`M` | 4.136 | **Descartar (dado pessoal).** |
| `reviewer_state` | sigla da UF | 3.991 | **Descartar (dado pessoal).** |

Além das colunas descartadas, o `review_text` passa por um filtro de e-mails, telefones e CPFs na
limpeza, porque o cliente pode ter escrito esses dados na própria resenha. O filtro não encontrou
nenhum caso. Há indício de anonimização na origem (um número aparece como "xxx1111111111xxx"), mas
não há documentação dela, por isso o filtro continua necessário, inclusive para a entrada da API.
Fora do filtro ficam códigos de rastreio dos Correios (por exemplo "PP075161001BR", em 4 resenhas):
não identificam a pessoa sozinhos, e o charter não os exige.

## Primeiros números

| Nota | Resenhas | Fração |
|---:|---:|---:|
| 1 | 27.369 | 20,7% |
| 2 | 8.389 | 6,3% |
| 3 | 16.315 | 12,3% |
| 4 | 32.345 | 24,4% |
| 5 | 47.955 | 36,2% |

- Há 48.001 produtos únicos e 112.993 avaliadores únicos.
- `recommend_to_a_friend`: 96.368 `Yes`, 35.987 `No`, 18 nulos.
- Estes números são do arquivo **bruto**. Eles mudam depois da limpeza e da deduplicação, e a EDA
  (`notebooks/`) é onde eles são analisados.

## Os 5 Vs deste dataset

Os 5 Vs descrevem um conjunto de dados por cinco eixos: volume, velocidade, variedade, veracidade e
valor. Aqui eles servem para justificar as escolhas de engenharia, e não para inflar o problema.

| V | No B2W-Reviews01 | Consequência no projeto |
|---|---|---|
| **Volume** | 132.373 resenhas brutas (49 MB de CSV). Depois da limpeza, 126.055 resenhas e ~2,9 milhões de palavras (mediana de 16 por resenha), de 48.001 produtos. | Cabe na memória de um notebook. O processamento é em pandas, numa máquina só. Spark ou um cluster não teriam justificativa técnica. |
| **Velocidade** | Base estática: resenhas de 01/01/2018 a 31/05/2018 (~880 por dia, em média), publicadas uma vez e fixadas num commit. Não chega dado novo. | Processamento em **batch** (abaixo). |
| **Variedade** | Texto livre e informal (gírias, falta de acento, erros de digitação), mais campos estruturados (nota, data, produto, categoria) e o perfil do avaliador. | Só `review_text` e `overall_rating` entram no modelo. O perfil sai por ser dado pessoal, e o tokenizador BPE lida com o texto informal sem dicionário fixo. |
| **Veracidade** | Funil medido em `reports/metrics/cleaning.json`: 3.275 linhas sem texto, 105 vazias depois da normalização, 521 textos idênticos com notas diferentes e 2.417 duplicatas. A documentação diverge do arquivo (o separador é vírgula, e não ponto e vírgula). A nota nem sempre combina com o texto: o juiz de sentimento acerta só 0,728 em resenhas reais. | Limpeza e dedupe **antes** do split. O teto de 0,728 é a referência da coerência (H2), e não 1,0. |
| **Valor** | Resenhas reais, em português, com nota de 1 a 5 dada pelo próprio cliente, sob licença que permite uso acadêmico. | A nota vira o condicionamento (`<nota_k>`): é o que permite gerar resenha *com o tom pedido*, o objetivo do projeto. |

## Batch, e não streaming

O pipeline é **batch**: cada estágio do `dvc.yaml` lê um arquivo inteiro, processa e grava outro, e o
`dvc repro` só refaz o que mudou. É a escolha certa aqui por três motivos:

1. **A fonte não muda.** O arquivo está fixado num commit de um repositório arquivado. Streaming serve
   para dados que chegam continuamente, e aqui não chega nada novo.
2. **O treino é offline.** O modelo é treinado uma vez, em CPU, e avaliado numa validação fixa. Um
   fluxo contínuo de dados mudaria a validação a cada leitura, e os experimentos deixariam de ser
   comparáveis.
3. **Reprodutibilidade.** Batch com hash de entrada (DVC) garante que a mesma entrada gere a mesma
   saída. Isso é o que permite comparar runs e reproduzir o projeto em outra máquina.

Streaming passaria a fazer sentido se o produto coletasse resenhas novas o tempo todo e o modelo
precisasse se atualizar com elas. Isso fica fora do escopo. Na Etapa 4, a API atende pedidos **um a
um** (inferência online), mas o modelo que ela carrega continua sendo treinado em batch.

## Formatos

| Camada | Formato | Por quê |
|---|---|---|
| `data/raw` | CSV (49 MB) | É o formato publicado pela fonte. Fica intocado, para conferência pelo hash. |
| `data/processed` | Parquet (11,5 MB, só as 4 colunas que o projeto usa) | Colunar, tipado e comprimido. Lê só as colunas pedidas e preserva os tipos (a nota continua inteira, e o texto, texto), sem as ambiguidades de separador e de aspas do CSV. |
| `data/curated` | Parquet (train, validation, test) | O mesmo motivo, e cada split é um arquivo separado: o `test.parquet` fica fechado, sem ser lido junto com os outros. |
| `models/` | `.pt` (PyTorch), `.json` (tokenizador), `.joblib` (juiz) | O formato nativo de cada biblioteca, carregado pela mesma biblioteca que salvou. |
