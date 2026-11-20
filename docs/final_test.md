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
