# Licença dos dados, do modelo e dos textos gerados

O **código** deste repositório está sob a licença MIT (arquivo `LICENSE`). Ela **não** cobre o que
deriva do dataset B2W-Reviews01. Esse conteúdo segue a licença do dataset original:

**Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional (CC BY-NC-SA 4.0)**
<https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt_BR>

## O que esta licença cobre

| Conteúdo | Onde está | Por que deriva dos dados |
|---|---|---|
| Dados brutos, limpos e divididos | `data/` (versionados pelo DVC) | são o próprio dataset, ou ele limpo e dividido |
| Tokenizador, juiz de sentimento e Transformer treinados | `models/` e os arquivos do Release `v0.1.0` | foram treinados nas resenhas do dataset |
| Resenhas geradas pelos modelos | `reports/samples/` e as respostas da API | foram produzidas por modelos treinados no dataset |
| Resenhas reais e planilhas da rubrica | `reports/rubric/` e `reports/samples/test_real_control.jsonl` | contêm resenhas reais do dataset |

## O que a licença exige

- **Atribuição (BY):** dar crédito ao dataset e indicar as mudanças feitas. Para isso:
  - **o crédito:** B2W-Reviews01, da B2W Digital, publicado em
    <https://github.com/americanas-tech/b2w-reviews01>;
  - **a citação:** Real, L.; Oshiro, M.; Mafra, A. (2019). *B2W-Reviews01: an open product reviews
    corpus*. Symposium in Information and Human Language Technology (STIL);
  - **as mudanças** feitas neste projeto: limpeza e normalização do texto, remoção das colunas de
    perfil do avaliador e de dados pessoais, deduplicação e divisão por produto em treino, validação
    e teste. Os detalhes estão em `docs/data_manifest.md`.
- **Não comercial (NC):** nenhum desses conteúdos pode ser usado para fins comerciais.
- **Compartilhar igual (SA):** quem redistribuir esses conteúdos, ou algo feito a partir deles,
  precisa usar a mesma licença.

## Além da licença: o uso pretendido

As resenhas geradas são **texto sintético para testar software** (telas, moderação, classificadores
de sentimento). Elas não podem ser publicadas como avaliação de um cliente real: isso engana
consumidores e, no Brasil, pode configurar publicidade enganosa (`docs/charter.md`, seção 9).

Os conteúdos são fornecidos "no estado em que se encontram", sem garantia de qualquer tipo.
