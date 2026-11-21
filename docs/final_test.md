# Teste final (Etapa 4.3)

> **Pré-registro**, escrito em 20/11/2026, **antes** de o teste ser aberto. O código e este documento
> vão para o Git (commit e push) antes da primeira e única execução do estágio `final_eval`. Os
> resultados entram depois, numa seção própria, sem mudar nada do que está acima dela.

## O que já está fixado

Nada do que segue muda depois de o teste ser aberto:

- **O modelo:** o checkpoint final da Etapa 3 (`models/transformer.pt`, run `final-combinada`).
- **A decodificação:** temperatura 1, sem top-k (`docs/inference.md`, 4.1).
- **Os filtros do produto:** até 83 tokens de texto, sem palavra repetida 3 vezes seguidas, sem dado
  pessoal nem URL, e até 5 tentativas (4.2).
- **Os limiares:** os do charter (seção 7), os mesmos da validação.

## O que o teste mede, e o que não mede

Num modelo gerador, o teste não entra do mesmo jeito em todas as hipóteses:

| Hipótese | Como é medida no teste | Limiar |
|---|---|---|
| H1, perplexity por palavra | sobre as **resenhas do teste**: é a única que pergunta "o modelo prevê bem texto real que nunca viu?". O trigrama é medido nas mesmas resenhas, para a comparação continuar justa | ≤ 72 |
| H2, coerência macro | sobre **1.000 resenhas novas**, geradas pelo produto com uma semente nunca usada (2027) | ≥ 0,57 |
| H4, cópia exata | idem | ≤ 1% |
| H5, cópia de 8 palavras | idem | ≤ 7% |
| H6, distinct-2 | idem | ≥ 0,59 |
| H3, rubrica humana | **não é medida no teste**: ela compara Transformer e trigrama, e foi medida na validação. O lote v2 (abaixo) é outra pergunta | — |

**Por que uma semente nova:** a decodificação foi escolhida olhando as métricas das amostras da
semente 42. Medir o resultado final nessas mesmas amostras daria um número otimista, porque a escolha
favoreceu justamente a combinação que se saiu melhor nelas. É o mesmo motivo pelo qual o teste existe.

**Referências humanas, do teste:** para ler H2, H5 e H6, o estágio também mede as resenhas reais do
teste com a mesma régua. São elas o teto do juiz de sentimento, o distinct-2 e a cópia do treino
(depois e antes do dedupe). Na validação, essas referências foram 0,728, 0,65 e 3,3% / 5,5%.

## Conferências de sanidade, antes de ler o teste

O estágio primeiro recalcula, na **validação**, dois números já conhecidos:
- a perplexity por palavra do Transformer (69,31, `reports/metrics/transformer_eval.json`);
- a do trigrama (90,2, `reports/metrics/baseline.json`).

Se algum não bater (tolerância de 0,01), o estágio **para com erro antes de abrir o teste**. Um
número de teste só vale se o código que o mede reproduz o que já se sabe.

## Uma única vez

- `final_eval` é o **único** código do projeto que lê `data/curated/test.parquet`.
- Depois da execução, o estágio é **congelado** (`dvc freeze final_eval`). Um `dvc repro` não o roda
  de novo, nem se uma dependência mudar.
- **Se alguma hipótese reprovar no teste, o resultado é reportado como saiu.** Não há segunda
  tentativa, nem ajuste de modelo, decodificação, filtro ou limiar. O que pode mudar depois é só a
  explicação, numa seção nova, separada dos números.

## Lote humano v2 (rubrica v1.1)

O lote v2 não testa uma hipótese do charter. Ele responde a duas perguntas sobre o **produto**:
1. Uma pessoa lê o texto padrão da API como resenha?
2. Cortar a cauda da distribuição (top-k 50), que reduziu as palavras inventadas em seis vezes na
   4.1, melhora a fluência que uma pessoa percebe?

**Montagem:**

| Fonte | De onde vem | Por nota | Total |
|---|---|---:|---:|
| Padrão (T 1, sem top-k) | `reports/samples/inference.jsonl`, gerado na 4.2, antes de o teste ser aberto | 10 | 50 |
| Top-k 50 | idem | 10 | 50 |
| Resenhas reais do teste (controle) | sorteadas pelo `final_eval` entre as que têm até 83 tokens, o mesmo limite do produto | 10 | 50 |

- **O controle tem até 83 tokens** para ter tamanho comparável ao texto gerado: uma resenha real muito
  longa denunciaria a origem e não teria um par no produto.
- **Exclusão:** sai do sorteio todo texto idêntico, depois de normalizar caixa, pontuação e espaços,
  a um dos 300 textos do lote v1 (`docs/inference.md`, 4.1).
- **Às cegas, como no v1:** os 150 textos são embaralhados, com ids anônimos. A chave fica num
  arquivo separado, aberto só depois da pontuação inteira. A pontuação usa a rubrica **v1.1**.

**Como ler (fixado agora):**
- O número principal é a fração de amostras ruins (algum critério com nota 1) de cada fonte.
- A comparação entre o padrão e o top-k 50 é **exploratória**. A direção esperada, pela 4.1, é menos
  amostras ruins com top-k 50. Com 50 textos por fonte e frações perto de 30%, o intervalo de 95% da
  diferença é de uns ±18 pontos: diferenças menores que isso não serão tratadas como achado.
- Uma fração de ruins no padrão perto da do lote v1 (37%) quer dizer que os filtros da 4.2 não mudaram
  a qualidade percebida. Bem abaixo, que ajudaram, com as diferenças de rubrica (v1.0 contra v1.1)
  entrando na leitura.

## Ensaio, antes de abrir o teste (20/11/2026)

O estágio foi ensaiado de ponta a ponta com a **validação no lugar do teste**, com as saídas e o
MLflow numa pasta temporária, fora do repositório. O objetivo era garantir que a execução de verdade
não quebre no meio: uma quebra depois de ler o teste obrigaria a lê-lo de novo.

- A sanidade reproduziu exatamente os números conhecidos: 69,31 (Transformer) e 90,2 (trigrama).
- Todas as medidas foram calculadas e todas as saídas gravadas (1.000 resenhas, 50 de controle).
- O registro no MLflow foi conferido à parte, com o mesmo formato (SQLite) do projeto.

**Transparência:** as métricas das resenhas geradas (H2, H4, H5 e H6) não dependem do teste: as
resenhas vêm do modelo, e a cópia e o vocabulário, do treino. Por isso o ensaio já as calculou com a
semente 2027, e a execução de verdade vai reproduzi-las exatamente. Isso não muda nada, porque tudo
acima desta seção (modelo, decodificação, filtros, limiares e a regra "reprovou é reprovou") foi
fixado e commitado antes. O que o teste traz de novo é o H1 e as referências humanas.

## Resultado (20/11/2026, 19:48–19:53 UTC, commit 0ea7148)

O estágio `final_eval` rodou **uma vez**, em 4 min 43 s, e foi congelado em seguida (`frozen: true`
no `dvc.yaml`). Os números estão em `reports/metrics/final_eval.json` e no run `final-test` do
MLflow.

**A sanidade passou antes de o teste ser lido:** 69,31 (Transformer) e 90,2 (trigrama), os mesmos
números já conhecidos.

### As hipóteses no teste

| # | Medida | Limiar | Validação (Etapa 3) | **Teste** | Passou? |
|---|---|---|---:|---:|---|
| H1 | Perplexity por palavra (Transformer) | ≤ 72 | 69,31 | **68,94** | sim |
| — | Perplexity por palavra (trigrama), as mesmas resenhas | — | 90,2 | 88,87 | — |
| H2 | Coerência macro | ≥ 0,57 | 0,721 | **0,704** | sim |
| H4 | Cópia exata | ≤ 1% | 0,0% | **0,0%** | sim |
| H5 | Cópia de 8 palavras | ≤ 7% | 3,1% | **1,8%** | sim |
| H6 | Distinct-2 | ≥ 0,59 | 0,681 | **0,679** | sim |
| H3 | Rubrica humana | < trigrama | 37% contra 85% | não se mede no teste | sim (validação) |

O `reports/metrics/final_eval.json` mostra o H3 como "pendente": o estágio só calcula o que é
automático, e o H3 é trabalho humano, medido na validação (`reports/metrics/rubric.json`). O arquivo
não foi editado à mão, porque o estágio está congelado.

As colunas têm fontes diferentes:
- **H1, teste:** as 12.607 resenhas do teste, 280 delas pontuadas por janela deslizante por serem
  maiores que o contexto.
- **H2, H4, H5 e H6, teste:** 1.000 resenhas novas do produto, geradas com a decodificação padrão, os filtros
  e a semente 2027.
- **H2, H4, H5 e H6, validação:** o texto cru da Etapa 3, com a semente 42.

### As referências humanas do teste

| Medida | Resenhas reais da validação | Resenhas reais do teste | Produto (teste) |
|---|---:|---:|---:|
| Coerência pelo juiz (o teto) | 0,728 | 0,755 | 0,704 |
| Distinct-2 | 0,655 | 0,652 | 0,679 |
| Cópia de 8 palavras, depois do dedupe | 3,3% | 3,5% | 1,8% |
| Cópia de 8 palavras, antes do dedupe | 5,5% | 5,1% | — |
| Palavras por resenha (média) | — | 26,2 | 17,4 |

Cada referência usa 200 resenhas reais por nota, como as amostras geradas.

### Leitura

1. **O modelo generaliza para resenhas que nunca viu.** A perplexity no teste (68,94) é até um
   pouco menor que na validação (69,31). A vantagem sobre o trigrama se mantém: 22,4% menor no teste,
   contra 23,2% na validação. A ligeira melhora não é mérito do modelo, e sim a variação natural entre
   dois conjuntos de produtos diferentes: o trigrama também melhorou no teste (de 90,2 para 88,87).
2. **As cinco hipóteses medidas passam no teste**, todas com folga. A menor folga é a do H2: 0,704
   contra 0,57.
3. **A semente nova custou um pouco de coerência**, como o pré-registro previa. Com o mesmo pipeline,
   as resenhas da semente 42 tinham coerência 0,728 (`reports/metrics/inference.json`), e as da
   semente 2027, 0,704: 2,4 pontos a menos. Isso está dentro do erro padrão de uns 2 pontos, mas na
   direção esperada. É o motivo de não medir o resultado final nas amostras usadas para escolher.
4. **O produto fica perto do teto do juiz:** 0,704 contra 0,755 nas resenhas reais do teste, ou 93%
   do teto. Por nota, a mais difícil continua sendo a 3 (0,52 no produto, contra 0,615 nas reais).
5. **Mais variado e menos copiado que as pessoas:** o distinct-2 do produto (0,679) passa o das
   resenhas reais (0,652), e a cópia de 8 palavras (1,8%) fica abaixo da humana (3,5%). Os textos
   também são mais curtos (17,4 contra 26,2 palavras), em parte pelo limite de 83 tokens, e texto
   curto repete menos trechos longos. Por isso a cópia baixa não prova sozinha que o modelo não
   decora.
6. **O produto se comportou como na 4.2:** nenhuma resenha esgotou as tentativas, 2,7% precisaram de
   mais de uma (no máximo 3), e o tempo médio foi de 172 ms por resenha nesta máquina.

## Resultado do lote v2 (rubrica v1.1)

Pontuado às cegas pelo autor, com a chave aberta só depois da planilha inteira. Os números estão em
`reports/metrics/rubric_v2.json`.

| Fonte | Amostras ruins | Fluência | Plausibilidade | Coerência | Repetição |
|---|---:|---:|---:|---:|---:|
| Resenhas reais do teste (controle) | **0%** (0 de 50) | 2,94 | 2,96 | 2,84 | 3,00 |
| Padrão (T 1, sem top-k) | **10%** (5 de 50) | 2,22 | 2,54 | 2,82 | 2,96 |
| Top-k 50 | **0%** (0 de 50) | 2,70 | 2,78 | 2,88 | 2,98 |

Amostra ruim é a que tem algum critério com nota 1. As colunas de critério são médias de 1 a 3.

### Leitura, pela regra escrita antes

1. **O autor, às cegas, leu 45 dos 50 textos padrão como resenha.** Os 5 ruins foram todos por
   fluência 1, e nenhum por plausibilidade, coerência ou repetição. Eles ficam nas notas 1, 2 e 5
   (2, 2 e 1 textos). Mas "não ruim" não quer dizer "perfeito": 29 dos 50 tiveram fluência 2 (dá para
   entender, com trechos estranhos), e só 13 tiveram nota 3 nos quatro critérios. Nas resenhas reais
   do teste, foram 39 de 50, e no top-k 50, 27.
2. **Top-k 50 contra o padrão: na direção esperada, mas não é um achado.** A diferença é de 10 pontos
   (0% contra 10%). O pré-registro fixou que diferenças menores que uns 18 pontos não seriam
   tratadas como achado, e 10 está abaixo disso. Aquela margem foi calculada para frações perto de
   30%; com as frações observadas, o teste exato de Fisher dá p = 0,056, e a conclusão é a mesma. A
   fluência média (2,70 contra 2,22) aponta para o mesmo lado que a queda de seis vezes nas palavras
   inventadas da 4.1, mas a comparação das médias não foi pré-registrada e fica como descrição. O
   padrão do produto continua o da regra da 4.1 (temperatura 1, sem top-k), escolhido pela
   diversidade (H6), que o top-k 50 reprova.
3. **O padrão caiu de 37% (lote v1) para 10%, e aqui a leitura fica mais cautelosa que o
   pré-registro.** A regra escrita antes dizia que um número bem abaixo de 37% indicaria que os
   filtros ajudaram, "com as diferenças de rubrica entrando na leitura". Olhando os dados, essas
   diferenças pesam demais para atribuir a queda aos filtros. Mudaram várias coisas ao mesmo tempo:
   - **a rubrica:** a v1.1 esclareceu três fronteiras (fluência 1 ou 2, coerência na nota 3 e
     plausibilidade de textos curtos). Das 37 amostras ruins do v1, 33 tinham fluência 1 e 18
     tinham plausibilidade 1, e só 16 eram ruins apenas pela fluência;
   - **os filtros da 4.2:** o limite de 83 tokens corta os textos longos, que a análise de erros
     associou às amostras ruins;
   - **as amostras:** são outras, sorteadas de outro conjunto;
   - **o avaliador:** é o mesmo, mas pontuou o v2 depois de fazer a análise de erros do v1, o que
     pode mudar o olhar;
   - **a composição do lote:** no v1, um terço era a salada do trigrama, que pode ter feito o texto
     do Transformer parecer melhor ou pior por contraste; no v2, dois terços são texto do modelo.

   O desenho do lote não separa essas causas. O que dá para afirmar é o número de agora: com a
   rubrica v1.1, o texto padrão do produto teve 10% de amostras ruins.
4. **O controle seguiu funcionando neste lote:** 0% de resenhas reais ruins (3% no v1). Com a
   rubrica v1.1, este avaliador não puniu o jeito informal de escrever do B2W em nenhuma das 50.

**Limite:** são 50 textos por fonte e um avaliador só, que é o autor. A margem de 95% da fração de
ruins do padrão vai de 4% a 21%. Os números servem para descrever o produto, e não para comparações
finas.
