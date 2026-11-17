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
| `review_title` | texto | 302 | Avaliar na EDA se entra junto com o texto. |
| `overall_rating` | inteiro 1 a 5 | 0 | **Condicionamento** (`<nota_k>`) e rótulo do classificador. |
| `recommend_to_a_friend` | `Yes`/`No` | 18 | Checagem de consistência com a nota na EDA. |
| `review_text` | texto | 3.275 | **Alvo da geração.** As linhas sem texto saem na limpeza. |
| `reviewer_birth_year` | número | 5.984 | **Descartar (dado pessoal).** |
| `reviewer_gender` | `F`/`M` | 4.136 | **Descartar (dado pessoal).** |
| `reviewer_state` | sigla da UF | 3.991 | **Descartar (dado pessoal).** |

Além das colunas descartadas, o texto livre (`review_text` e `review_title`) passa por um filtro de
e-mails, telefones e CPFs na limpeza, porque o cliente pode ter escrito esses dados na própria resenha.

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
