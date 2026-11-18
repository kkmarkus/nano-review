# Charter do projeto nano-review

> Versão 0.5 (fim da Etapa 2: definição da cópia exata corrigida e cautelas de medida acrescentadas
> pela revisão, antes de medir qualquer amostra do Transformer; os limiares não mudaram). Versão 0.4:
> fim da Etapa 1, baseline medido e limiares da hipótese fixados antes do Transformer. Este documento fixa o problema e a forma de medir o sucesso
> **antes** de existir modelo, para que a avaliação não seja ajustada depois para caber no resultado.

## 1. Problema e nicho

Quem desenvolve telas de avaliação, filtros de moderação ou classificadores de sentimento precisa de
muitos textos de resenha para testar o sistema. Dados reais têm problemas para esse uso: dados pessoais,
licença, poucas resenhas em algumas notas (no B2W, a nota 2 é só 6,3% das resenhas e a nota 3, 12,3%,
contra 36,2% da nota 5) e o fato de que não dá para pedir "mais 500 resenhas de nota 2".

O nano-review é um gerador de resenhas curtas de produto em português, condicionado à nota, treinado
**do zero** (Transformer só-decoder pequeno, sem modelo pré-treinado) em CPU.

## 2. Usuário e decisão

- **Usuário:** desenvolvedor ou analista de QA que precisa de textos sintéticos.
- **Decisão que o sistema apoia:** "este conjunto de textos serve para testar a minha tela, o meu filtro
  ou o meu classificador?". Para isso, o texto precisa soar como resenha, ser coerente com a nota pedida
  e variar entre amostras.

## 2.1 Erro caro

1. **Texto que contradiz a nota** (por exemplo, um elogio entregue como nota 1). É o pior erro porque
   não aparece: o usuário roda o próprio classificador sobre o texto, vê um "erro" e conclui que o
   classificador dele falhou, quando quem errou foi o gerador. Por isso a coerência é a métrica principal.
2. **Cópia de uma resenha real.** O texto deixa de ser sintético e pode carregar informação de um
   cliente real. Por isso a taxa de cópia é medida em toda avaliação.
3. Texto pouco fluente ou repetitivo é um erro mais barato, porque o usuário percebe na hora e descarta.

## 3. Entrada e saída

| | Descrição | Exemplo |
|---|---|---|
| Entrada | Uma nota inteira de 1 a 5 (mais parâmetros de amostragem: temperatura, top-k, tamanho máximo) | `5` |
| Formato interno | Token de prefixo da nota, depois o texto, depois fim de sequência | `<nota_5> chegou rápido e funciona bem <eos>` |
| Saída | Uma resenha curta em português | "chegou rápido e funciona bem, recomendo" |

Condicionamento da v1: **só a nota**. A categoria do produto fica como extensão.

## 4. Dados

B2W-Reviews01 (GitHub da americanas-tech, commit fixado), resenhas reais da Americanas/Submarino, licença CC BY-NC-SA 4.0 (exige
atribuição, uso não comercial e compartilhamento pela mesma licença). Detalhes no
`docs/data_manifest.md`. Split por produto em treino, validação e teste; o teste é aberto uma única vez,
na Etapa 4.

## 5. Métricas

Todas são medidas na **validação** durante o desenvolvimento e no **teste** só no fim.

| Métrica | O que mede | Como |
|---|---|---|
| **Coerência com a nota** (principal) | A resenha gerada combina com a nota pedida? | Gerar N amostras por nota, classificar com o classificador de sentimento de 3 classes (1-2 negativo, 3 neutro, 4-5 positivo) e calcular a fração que cai na classe certa. Média **macro** entre as classes, para a nota 5 (a mais comum) não inflar o número. |
| Perplexity | Quão bem o modelo prevê texto real que ele não viu | exp da perda média (entropia cruzada) por token na validação. |
| Diversidade | O modelo repete sempre a mesma frase? | distinct-1 e distinct-2: n-gramas únicos divididos pelo total de n-gramas nas amostras. |
| Cópia do treino | O modelo está decorando? | Fração de amostras com pelo menos 5 palavras idênticas a uma resenha de treino (cópia exata) e fração com algum trecho de 8 palavras presente no treino. Texto curto e genérico ("ótimo", "boa qualidade") coincide com o treino sem ter sido decorado, por isso não conta como cópia exata. |
| Rubrica humana | O que número nenhum pega: fluência, plausibilidade | Rubrica escrita **antes** de lermos amostras (até o início da Etapa 3), aplicada a uma amostra fixa. |
| Análise de erros | Por que o modelo erra, não só quanto | Pelo menos 10 amostras ruins analisadas uma a uma: nota pedida, texto gerado, o que está errado, hipótese da causa e ação possível. |

Três cautelas:

1. **O juiz erra.** A coerência depende de um classificador treinado nos dados reais. A coerência das
   **resenhas reais** da validação, medida por ele, é o teto: 0,728 (seção 6.1), e não 1,0. Um gerador
   não pode ser julgado mais coerente do que o juiz consegue enxergar. Por isso a referência real é
   sempre reportada junto.
2. **Perplexity só se compara na mesma unidade.** O baseline trabalha com palavras e o Transformer vai
   trabalhar com tokens BPE. A comparação usa a **perplexity por palavra** para os dois:
   exp(−log P(texto) / número de palavras), com as palavras contadas pela mesma tokenização do baseline
   (palavras e pontuação, mais o `<eos>`). Para o Transformer, log P(texto) é a soma dos log-probs dos
   tokens BPE. A probabilidade de um texto não depende de como ele foi quebrado, então essa conta é
   justa. Há uma ressalva, e ela **favorece o baseline**: o trigrama junta as palavras raras em `<unk>`
   e não paga para soletrá-las, e o Transformer paga. Se o Transformer vencer assim, vence com folga.
   Os dois precisam ser medidos nas **mesmas resenhas**. O treino do Transformer descarta as 2,4% mais
   longas que o `block_size`, e a `val_loss` da Etapa 2 também (`reports/metrics/train.json`). A
   avaliação do H1 não pode descartá-las, porque são justamente as mais difíceis: na Etapa 3, antes de
   medir o H1, elas passam a ser pontuadas por janela deslizante. **Ainda não implementado.**
3. **O juiz é um saco de palavras.** O TF-IDF ignora a ordem: uma salada de palavras com "ótimo" e
   "recomendo" parece positiva. O baseline mostrou isso na prática: o unigrama por nota, que não forma
   frase nenhuma, teve coerência **maior** que o trigrama (0,678 contra 0,386). Por isso a coerência
   **nunca é lida sozinha**: ela só vale para um modelo que também tem perplexity baixa e passa na
   rubrica de fluência.
4. **A referência humana de cópia é medida depois da deduplicação.** Como o dedupe apagou os textos
   repetidos, nenhuma resenha real da validação pode ser cópia exata do treino, e a cópia de 8 palavras
   delas também sai menor. Medida no CSV bruto (só produtos do treino e da validação), antes do dedupe
   e da remoção dos textos com notas conflitantes, a referência humana é de **0,72%** de cópia exata
   (com pelo menos 5 palavras) e **5,5%** de cópia de 8 palavras (`real_validation_copy_before_dedupe`
   em `reports/metrics/baseline.json`). Esses são os números de comparação para H4 e H5. Duas
   ressalvas: sem os textos conflitantes, a cópia exata cai para uns 0,63%; e essa referência usa toda
   a validação, na distribuição natural das notas, enquanto as amostras dos geradores são 200 por nota.
   O relatório também mostra `any` (cópia exata **ou** de 8 palavras), porque uma cópia exata de 5 a 7
   palavras não tem trecho de 8.
5. **A EDA olhou o CSV bruto inteiro**, inclusive linhas que depois foram para o teste, para decidir as
   regras de limpeza. É prática comum e nenhuma regra usa a nota ou o texto de um exemplo específico,
   mas é uma limitação: o teste não está 100% fora de toda decisão.

## 6. Baseline

Cada lado tem um **piso trivial** e um **baseline de verdade**. O piso mostra quanto se ganha "de
graça"; sem ele, um número alto pode ser só efeito do desbalanceamento.

| | Piso trivial | Baseline |
|---|---|---|
| Gerador | Unigrama por nota: sorteia palavras pela frequência na nota pedida, sem olhar o contexto | Trigrama de palavras com suavização, treinado no formato `<nota_k> texto <eos>` |
| Juiz (classificador) | Classe majoritária: responde sempre a classe mais comum (positivo) | TF-IDF + regressão logística, treinada no conjunto de treino real |

- O trigrama é o "modelo de linguagem mais simples que funciona": ele só memoriza sequências locais de
  três palavras. Se o Transformer não o superar, o Transformer não se justifica.
- A regressão logística é a ponte para a rede neural: uma camada linear seguida de softmax, treinada
  com cross-entropy por **descida do gradiente**, o mesmo ciclo forward → loss → backward → atualização
  dos pesos que treina o Transformer. O relatório explica esse treino passo a passo. O juiz foi
  implementado com o scikit-learn (`LogisticRegression`, que usa L-BFGS, um método de gradiente mais
  sofisticado que a descida simples; o modelo é o mesmo). O ciclo forward → backward escrito à mão
  aparece na Etapa 2, no bigrama neural em PyTorch.

### 6.1 Resultados do baseline (validação, 18/11/2026)

Juiz (3 classes), contra o piso da classe majoritária:

| | Acurácia | F1 macro | Recall neg. | Recall neutro | Recall pos. |
|---|---:|---:|---:|---:|---:|
| Classe majoritária | 0,592 | 0,248 | 0,00 | 0,00 | 1,00 |
| TF-IDF + regressão logística | 0,801 | 0,729 | 0,89 | 0,54 | 0,82 |

Geradores (200 amostras por nota, semente 42), contra 200 resenhas reais da validação por nota:

| | Perplexity por palavra ↓ | Coerência macro | Amostras vazias | Distinct-2 | Cópia exata | Cópia de 8 palavras |
|---|---:|---:|---:|---:|---:|---:|
| Unigrama por nota (piso) | 516 | 0,678 | 3,8% | 0,84 | 0,0% | 0,0% |
| **Trigrama (baseline)** | **90,2** | **0,386** | 1,3% | 0,71 | 0,0% | 0,2% |
| Resenhas reais (referência) | — | 0,728 | 0,0% | 0,65 | 0,0% | 3,3% |

Na versão 0.4 esta tabela mostrava um sorteio que não se repetia: a ordem do vocabulário no sorteio
dependia do *hash* do Python, que muda a cada execução, e a mesma semente dava amostras diferentes
(coerência do trigrama 0,407 naquela execução). Corrigido na revisão do fim da Etapa 2; os números
acima se repetem byte a byte a cada `dvc repro`. A perplexity não dependia do sorteio e não mudou. As
"cópias exatas" do trigrama eram textos de 1 e 2 palavras e deixaram de contar (seção 5).

Amostra vazia (sem nenhuma palavra, como "" ou "!!!") conta como **incoerente**: sem essa regra, o juiz
a chamaria de positiva, o que daria acerto grátis nas notas 4 e 5.

Leitura: o trigrama escreve trechos locais fluentes, mas perde o tom, porque o token `<nota_k>` sai
da janela de duas palavras logo no começo. Nas notas 1 e 2, a coerência dele (0,36) fica perto do
acaso entre 3 classes. Fonte: `reports/metrics/judge.json` e `reports/metrics/baseline.json`.

Os pesos da interpolação (λ = 0,2 unigrama, 0,4 bigrama, 0,4 trigrama) foram escolhidos numa fatia
de 10% do **treino**, separada por produto e com vocabulário próprio, e não na validação. O modelo
final usa o vocabulário do treino inteiro; a diferença na taxa de `<unk>` entre os dois é pequena.

## 7. Hipótese

> Em comparação com o trigrama, o Transformer treinado do zero tem **perplexity por palavra menor** e
> **coerência com a nota maior** na validação, **sem copiar o treino mais do que resenhas humanas**
> copiam e **sem perder diversidade em relação às resenhas reais**.

A hipótese é relativa ao baseline de propósito. Os limiares abaixo foram fixados em 18/11/2026,
**depois** de medir o baseline e **antes** de treinar o Transformer. Eles não mudam depois de ver os
resultados dele.

Única alteração, na versão 0.5, antes de qualquer amostra do Transformer ser medida: a revisão do fim
da Etapa 2 mostrou que as 3 "cópias exatas" do trigrama eram "otimo", "Boa qualidade." e "boa
qualidade". Isso é texto genérico, e não memorização. Por isso a cópia exata passou a contar só
amostras com pelo menos 5 palavras. O número dos limiares não mudou; a justificativa de H4 e H5 passou
a usar a referência humana medida antes do dedupe (cautela 4).

| # | Critério | Limiar | Por quê |
|---|---|---|---|
| H1 | Perplexity por palavra | **≤ 72** (20% abaixo dos 90,2 do trigrama) | Uma melhora pequena poderia ser ruído da tokenização. A ressalva do `<unk>` (cautela 2) favorece o trigrama, então 20% é exigente. |
| H2 | Coerência macro | **≥ 0,57** | Fecha pelo menos metade da distância entre o trigrama e o teto do juiz (0,728). O limiar foi calculado com 0,407, o sorteio da versão 0.4; na medida reproduzível o trigrama tem 0,386, e 0,57 continua acima da metade (0,557). Com 200 amostras por nota, o erro padrão da coerência macro é de uns 2 pontos, então 16 pontos de ganho não são ruído. |
| H3 | Fluência (rubrica, às cegas) | Fração de amostras ruins **menor** que a do trigrama | Impede que a coerência venha de salada de palavras (cautela 3). |
| H4 | Cópia exata (amostras com pelo menos 5 palavras) | **≤ 1%** | Resenha idêntica a uma real não é sintética e pode carregar informação de um cliente (erro caro nº 2). As pessoas repetem uma resenha inteira em 0,72% dos casos (cautela 4); 1% é pouco acima disso. |
| H5 | Cópia de 8 palavras | **≤ 7%** | Humanos também repetem trechos ("chegou antes do prazo previsto"): 3,3% depois do dedupe e 5,5% antes (cautela 4). Exigir 0% seria exigir mais do que das pessoas. |
| H6 | Distinct-2 | **≥ 0,59** (90% dos 0,65 das resenhas reais) | A comparação é com o texto real, e não com o trigrama: o distinct alto do trigrama (0,71) e do piso (0,84) vem de texto incoerente, que é "diverso" à toa. |

**Leitura do resultado:** a hipótese é confirmada se H1 a H6 passam. Se H1 passa e H2 não, o modelo
aprendeu a escrever mas não a usar a nota, e o experimento "com/sem condicionamento" da Etapa 3 é o
lugar de investigar. Qualquer resultado é reportado como saiu, inclusive a reprovação.

## 8. Riscos

| Risco | Efeito | Mitigação |
|---|---|---|
| Desbalanceamento (nota 5 tem 36,2%, nota 2 só 6,3%; no classificador de 3 classes, o neutro fica com 12,3%) | O modelo aprende a gerar elogio para qualquer nota e vê poucos exemplos de nota 2 | Coerência macro por classe; avaliar amostragem balanceada ou peso por classe na Etapa 3 |
| Memorização | O "gerador" vira um copiador de resenhas reais | Métrica de cópia; dedupe antes do split; split por produto |
| Dados pessoais no texto | Vazar e-mail, telefone ou CPF de clientes reais | Descartar colunas de perfil do avaliador; filtrar o texto por regex na limpeza |
| CPU limita o tamanho do modelo | Texto pouco fluente | Resenhas curtas, vocabulário BPE pequeno, modelo pequeno; treinar cedo, não só no fim |
| Juiz fraco | Coerência medida com ruído | Reportar a acurácia do juiz ao lado do piso da classe majoritária; rubrica humana como segunda opinião |
| Alucinação | O texto inventa fatos sobre o produto (voltagem, tamanho, marca) | Deixar claro que a saída é texto de teste, não informação; condicionamento só pela nota, sem nome de produto |
| Uso indevido | Publicar avaliações falsas em loja real | Ver seção 9 |

## 9. Ética e fora do escopo

- **Uso pretendido:** gerar texto de teste para software (telas, moderação, classificadores).
- **Quando o modelo NÃO deve ser usado:**
  - publicar avaliações em loja, marketplace ou rede social. Resenha falsa engana consumidores e, no
    Brasil, pode configurar publicidade enganosa (CDC);
  - servir de fonte de informação sobre um produto, porque o texto pode inventar fatos;
  - substituir dados reais no treino de um classificador que vá para produção sem validação em dados
    reais, porque o texto herda os vieses do B2W e os do próprio gerador.
- Fora do escopo da v1: condicionamento por categoria ou produto, resenhas longas, uso comercial
  (a licença CC BY-NC-SA não permite), qualquer modelo pré-treinado como núcleo.

## 10. Decisões pendentes

- [x] Limiares numéricos da hipótese: seção 7 (18/11/2026).
- [x] Rubrica humana: `docs/rubric.md` (17/11/2026, antes de qualquer amostra gerada).
- [x] Corte de comprimento das resenhas: feito na Etapa 2, em tokens BPE, com referência de ~128
  tokens (decisão 13 da EDA, `notebooks/01_eda.ipynb`).
