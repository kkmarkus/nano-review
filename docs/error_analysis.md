# Análise de erros do Transformer (sub-etapa 3.5)

> O que dá errado nas amostras ruins do checkpoint final da Etapa 3, por que dá errado e o que
> fazer. As amostras foram geradas pelo modelo, e o vocabulário de referência vem do treino. **Nada
> vem do teste**, que continua fechado até a Etapa 4.

## De onde vêm as amostras

- O ponto de partida é o lote cego da rubrica v1.0 (`reports/rubric/`): 100 amostras do Transformer,
  20 por nota, sorteadas com temperatura 1 e sem top-k.
- Pelas notas do autor, **37 são ruins**, isto é, têm algum critério com nota 1 (ver o H3 em
  `docs/experiments.md`).
- As 37 foram classificadas à mão em `reports/rubric/error_types_v1.csv`. Cada amostra recebe um só
  tipo, o que mais pesa nela.
- As medidas vêm de `nano_review.evaluation.error_analysis` e ficam em
  `reports/metrics/error_analysis.json`:

```bash
python -m nano_review.evaluation.error_analysis --sheet reports/rubric/sheet_v1.csv \
    --key reports/rubric/key_v1.csv --types reports/rubric/error_types_v1.csv \
    --train data/curated/train.parquet --metrics reports/metrics/error_analysis.json
```

## Tipos de erro

| Tipo | No código | O que é | Amostras | Por nota (1/2/3/4/5) |
|---|---|---|---:|---|
| Deriva | `drift` | A primeira frase faz sentido, e o texto se perde depois | 13 | 1/5/2/3/2 |
| Salada | `garbled` | Não há frase com sentido desde o começo | 12 | 4/3/3/0/2 |
| Mistura | `mixed_topics` | As frases até se leem, mas falam de produtos incompatíveis ou se contradizem | 8 | 2/3/2/0/1 |
| Tom trocado | `wrong_tone` | O texto se entende, mas o tom não combina com a nota | 3 | 0/2/1/0/0 |
| Loop | `loop` | A mesma palavra se repete em sequência | 1 | 0/0/1/0/0 |

**Quase todo erro é de forma, e não de tom:** 33 das 37 amostras são deriva, salada ou mistura. Só 3
erram o tom, e uma delas (s059) só conta como ruim pelo rigor do lote v1: pela rubrica v1.1, ela teria
coerência 2 e não seria ruim.

## As hipóteses de causa, medidas

Uma palavra é **inventada** quando não aparece em nenhuma das 100.831 resenhas do treino. Números,
datas e medidas não contam. Resenha real também tem palavra inventada (erro de digitação, nome raro),
e por isso as resenhas reais do lote são a referência.

| Grupo | Amostras | Mediana de palavras | Com palavra inventada | Inventadas por 100 palavras | 1ª inventada: posição relativa | 1ª inventada: nº da palavra |
|---|---:|---:|---:|---:|---:|---:|
| Resenhas reais | 100 | 17 | 15% | 0,8 | 0,47 | 7ª |
| Transformer, ok | 63 | 12 | 16% | 1,5 | 0,45 | 9ª |
| **Transformer, ruim** | **37** | **22** | **78%** | **6,0** | **0,32** | **9ª** |
| Trigrama (todas) | 100 | 18,5 | 0% | 0,0 | — | — |

As duas últimas colunas são medianas, calculadas só entre as amostras que têm palavra inventada. A
posição relativa vai de 0 (primeira palavra) até perto de 1 (última). O número da palavra conta a partir do
início (1ª = primeira).

### 1. Palavras inventadas: associação forte, mecanismo ainda não testado

- **O que foi medido:** 78% das amostras ruins têm pelo menos uma palavra que não existe no treino.
  Nas amostras ok do Transformer, são 16%, quase a mesma fração das resenhas reais (15%). Mas a taxa
  por palavra das ok é quase o dobro da real (1,5 contra 0,8 por 100 palavras). As boas amostras do modelo
  inventam palavras em tão poucos textos quanto uma pessoa, mas inventam mais dentro deles.
- **A associação é, em parte, construída.** Quem pontuou viu as palavras inventadas ao dar a nota de
  fluência, e uma palavra que não existe puxa a fluência para baixo. Por isso, os 78% mostram que
  palavras inventadas e amostras ruins andam juntas, mas não provam que uma causa a outra.
- **O mecanismo proposto:** o Transformer escreve com **pedaços** de palavra (tokens BPE). Ao
  sortear com temperatura 1, às vezes ele junta pedaços que não formam palavra nenhuma: "refresias",
  "derrumar", "contortado". Isso é uma hipótese. O teste dela é gerar de novo com top-k ou com
  temperatura menor e ver se as palavras inventadas e as amostras ruins caem juntas, na Etapa 4.
- **O trigrama não serve de controle aqui.** Ele só sorteia palavras que viu no treino, então não
  pode inventar nenhuma: os 0% são uma consequência da construção dele, e não um resultado. O que ele
  mostra é o limite da medida: quase todas as 85 amostras ruins do trigrama (73 têm fluência 1) são salada de palavras reais, e a
  medida não pega nenhuma delas.
- **O que a medida não pega:** 8 das 37 ruins não têm palavra inventada. Só parte delas é palavra
  real no lugar errado, como "urina" num texto sobre entrega (s042) e o fim sem ligação da s118. As
  outras são tom trocado (s059, s062, s131), loop (s297), contradição (s056) e mistura (s165).

### 2. Comprimento: associado, sem direção definida

- As amostras ruins têm mediana de **22 palavras**, quase o dobro das ok (12).
- Há duas leituras, e a medida não separa uma da outra:
  1. **O texto longo quebra mais.** Cada token sorteado é uma nova chance de erro, e um erro estraga o
     contexto dos tokens seguintes.
  2. **O texto que quebrou fica longo.** O modelo, perdido, demora a chegar ao `<eos>`.
- **Onde o texto quebra:** em número de palavras, a primeira inventada aparece na **9ª palavra**,
  tanto nas ruins quanto nas ok. Os grupos são pequenos: são 29 ruins, 10 ok e 15 reais com palavra
  inventada. A posição relativa parece diferente (0,32 contra 0,45), mas isso é consistente com um
  efeito do comprimento: a mesma 9ª palavra fica mais cedo, em proporção, num texto de 22 palavras do
  que num de 12. O que separa as ruins não é *onde* aparece a primeira palavra inventada, e sim
  *quantas* aparecem (6,0 contra 1,5 por 100 palavras).

### 3. Falta de contexto (`block_size`): descartada

- O modelo vê até **128 tokens**. As amostras ruins têm, na mediana, **29 tokens**, e só 2 das 37
  passam de 100: s197 (126) e s146 (113).
- A primeira palavra inventada aparece por volta da 9ª palavra, muito antes do limite. Mesmo nas mais
  longas, ela está a 40% do texto na s197, a 1% na s146 e a 8% na s154 (94 tokens).
- A s197 é o caso extremo da leitura 2 da hipótese anterior. Com o prefixo, ela ocupa todo o
  contexto, e o texto termina no meio de uma frase ("Deveria :"). Ela foi, muito provavelmente,
  cortada pelo limite de geração, sem chegar ao `<eos>`. Só 6 das 1.000 amostras da avaliação chegam
  a 125 tokens.
- Por isso, aumentar o `block_size` não resolveria nada. O que quebra o texto é a qualidade de cada
  passo da geração, e não a memória do modelo.

### 4. Nota rara: apoiada, com cautela

| Nota | Fração no treino | Amostras ruins do Transformer |
|---:|---:|---:|
| 2 | 6,2% | **65%** |
| 3 | 12,3% | 45% |
| 1 | 19,4% | 35% |
| 5 | 37,0% | 25% |
| 4 | 25,0% | 15% |

- Quanto mais rara a nota no treino, mais amostras ruins ela tem. A única inversão é entre as notas 4
  e 5. A correlação de postos (Spearman) é −0,9.
- O efeito aparece sobretudo na **forma**: a nota 2 concentra 5 das 13 derivas. O tom raramente é o
  oposto do pedido (2 das 20 amostras de nota 2 têm coerência 1), mas também raramente é certo: só 9
  das 20 têm coerência 3, e as outras 9 ficam no tom ambíguo (coerência 2).
- **Cautela:** são 5 pontos com 20 amostras cada. O padrão é consistente, mas não é prova. A nota 3,
  além de rara, é ambígua por natureza.

## As 12 amostras, uma a uma

**Regra de escolha**, mecânica, para evitar escolher só os casos que confirmam a explicação:
1. Para cada nota, entram as duas amostras ruins de menor `sample_id` com tipos diferentes. Se a nota
   só tem um tipo, entram as duas de menor `sample_id`.
2. Depois, para cada tipo que ficou de fora, entra a amostra de menor `sample_id` desse tipo.

O resultado é 10 + 2 = 12 amostras. As notas F, P, C e R são fluência, plausibilidade, coerência e
repetição.

### s008: nota 1, salada (F1 P1 C2 R3)

> ter Esse jogo de futuramente, quero saber as refresias.

- **Problema:** não há frase válida. Só "quero saber" faz sentido, e o tom é neutro para uma nota 1.
- **Medida:** 9 palavras, 1 inventada ("refresias").
- **Causa provável:** o texto já começa fora do padrão ("ter" em minúscula, logo depois da nota), e o
  modelo não se recupera de um começo improvável (causa A).
- **Ação:** top-k na inferência, para cortar os tokens improváveis desde o primeiro passo.

### s109: nota 1, mistura (F1 P2 C3 R3)

> Comprei de forma guarda roupa carro fina e veio preto e branco , veio o mini bs6, mas a cor não
> veio. [...] Mto frágil ásperamente o gavetas, quero devolver e ter o meu dinheiro [...]

- **Problema:** guarda-roupa, carro, pote, bolsa e gavetas no mesmo texto. O tom (quero devolver) está
  certo.
- **Medida:** 48 palavras, 2 inventadas, as duas na segunda metade.
- **Causa provável:** o modelo só recebe a nota, e nada sobre o produto. Ele mistura o vocabulário de
  categorias diferentes, que no treino aparecem com a mesma nota (causa C). O texto longo agrava
  (causa B).
- **Ação:** condicionar também pela categoria, a extensão já prevista no charter.

### s003: nota 2, deriva (F2 P1 C2 R3)

> o produto não é de uma grande potência, onde parece uma penas de outras marcas...tive que derrumar
> bem o tecido, na cor dourado, [...] quero no Reclamena num 100,00 a 20W [...] conectaná-la

- **Problema:** a primeira oração é uma reclamação plausível. Depois vêm tecido, travesseiro e 20 W.
- **Medida:** 62 palavras, a quarta mais longa entre as ruins, com 4 inventadas. A primeira aparece a
  27% do texto.
- **Causa provável:** é o caso típico das causas A e B juntas. Cada palavra inventada piora o contexto
  da seguinte.
- **Ação:** top-k. Um limite de tamanho da geração na API reduziria o estrago, mas é mitigação, e não
  cura.

### s032: nota 2, mistura (F1 P2 C2 R3)

> O ôtra legal seus brinquedos;é difícil acender, pintura,estou esperando o técnico 3 para o carro
> [...] tá faltando um número para o motor para reposição!

- **Problema:** brinquedos, pintura, técnico, carro e motor. Algumas frases se leem ("estou esperando
  o técnico"), mas o conjunto não fala de um produto só.
- **Medida:** 52 palavras. A primeira inventada ("ôtra") é a **segunda palavra** do texto.
- **Causa provável:** começo fora do padrão (causa A) e falta do produto no condicionamento
  (causa C).
- **Ação:** top-k e condicionamento por categoria.

### s042: nota 3, deriva (F1 P2 C3 R3)

> O produto é bom, mas a entrega pela Americanas deveria ter uma urina na loja, ocorreu muito porq a
> mesma tinha problemas.

- **Problema:** a estrutura da nota 3 está certa ("é bom, mas" mais uma crítica à entrega). O meio da
  frase não faz sentido.
- **Medida:** 22 palavras e **nenhuma inventada**. "Urina" existe no treino (4 resenhas). O erro é uma
  palavra real no lugar errado, e a medida de palavras inventadas não o pega.
- **Causa provável:** o modelo aprendeu a **estrutura** de uma resenha de nota 3, mas não o sentido de
  cada frase. É um limite de capacidade: 5,2 M de parâmetros e ~2,3 milhões de palavras de treino
  (causa B).
- **Ação:** nenhuma barata. Mais dados ou mais parâmetros com regularização, como o dropout 0,1, que
  é trabalho futuro em `docs/experiments.md`.

### s056: nota 3, mistura (F1 P2 C2 R3)

> comprei esse smartphone pelo nao saiu nada não funciona funciona mas como o celular é perfeito pra
> meu namorado

- **Problema:** é uma contradição interna, "não funciona" e "é perfeito" sobre o mesmo celular.
- **Medida:** 19 palavras, nenhuma inventada.
- **Causa provável:** o modelo aprendeu que a nota 3 junta elogio e crítica, a satisfação parcial
  que a rubrica descreve para essa nota. Mas ele junta as duas coisas sem ligação lógica. A nota 3
  também é a mais difícil para o juiz de sentimento (coerência 0,525).
- **Ação:** nenhuma no modelo atual. Fica registrada como limite.

### s059: nota 3, tom trocado (F3 P3 C1 R3)

> Não veio todas as pilhas que achei fraco e parece ser muito frágil

- **Problema:** o texto é fluente e plausível, mas só reclama, numa nota 3.
- **Medida:** 13 palavras, nenhuma inventada.
- **Causa provável:** não é defeito do modelo. É a fronteira da rubrica que a v1.1 esclareceu: na nota
  3, um texto só negativo vale coerência 2. Pela v1.1, esta amostra não seria ruim.
- **Ação:** nenhuma no modelo. Mostra quanto o rigor do lote v1 pesa nos números (ver
  `docs/rubric.md`).

### s297: nota 3, loop (F1 P1 C2 R1)

> Bom s bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla suficiente bla
> bla bla

- **Problema:** é a única repetição mecânica do lote.
- **Medida:** **22 resenhas do treino** têm "bla bla bla". O modelo não inventou o padrão: ele o
  aprendeu dos dados. Provavelmente são clientes que enchem o texto para atingir um mínimo de
  caracteres.
- **Causa provável:** os dados (causa E), e não a arquitetura. A limpeza limita letras repetidas
  ("muuuuito"), mas não palavras repetidas.
- **Ação:** na Etapa 4, recusar ou regenerar na inferência uma saída com a mesma palavra repetida
  várias vezes seguidas. Filtrar essas resenhas na limpeza também resolveria, mas mudaria os dados e
  exigiria retreinar.

### s118: nota 4, deriva (F1 P2 C3 R3)

> Indico ela! Prático e eficaz Recomendo ou comprar não vai vim com ele

- **Problema:** o começo é uma recomendação correta, e o fim é um pedaço de frase sem ligação.
- **Medida:** 13 palavras, nenhuma inventada.
- **Causa provável:** o modelo emenda um pedaço de frase que não cabe no contexto (causa B). A
  expressão em si não existe no treino; a mais próxima é "não vem com", presente em 181 resenhas (com ou sem acento).
- **Ação:** nenhuma barata.

### s166: nota 4, deriva (F1 P2 C3 R3)

> Produto muito bom. Para um alarme estavam todas as muitas formas que eu preciso "nole" e parece
> eficiente.

- **Problema:** começo e fim coerentes com a nota 4, e o meio sem sentido.
- **Medida:** 18 palavras. A inventada ("nole") fica a 78% do texto.
- **Causa provável:** um token improvável no meio da geração (causa A).
- **Ação:** top-k.

### s072: nota 5, mistura (F1 P2 C3 R3)

> Acredito que o produto seja bom, cumpre o que promete! Recomendo! Por isso não ter esse tipo de
> barbeador, pq ela ajuda a segurar desde o raro e protege. Contortado, bem legal no carro.

- **Problema:** as duas primeiras frases são uma resenha de nota 5 perfeita. Depois vêm barbeador e
  carro juntos.
- **Medida:** 34 palavras, 1 inventada ("contortado", a 85% do texto).
- **Causa provável:** a falta do produto no condicionamento (causa C) somada ao texto longo
  (causa B).
- **Ação:** condicionamento por categoria.

### s198: nota 5, deriva (F1 P1 C3 R3)

> Aparelho lindo e desempenho Ótimo funcionamento, água feita em plástico entretia de uma
> compartimentos de cinema em casa!!!

- **Problema:** o elogio do começo está certo para a nota 5. Depois vêm água, plástico e cinema em
  casa.
- **Medida:** 18 palavras. A inventada ("entretia") fica a 56% do texto.
- **Causa provável:** as causas A e C juntas.
- **Ação:** top-k e condicionamento por categoria.

## Causa → ação

| # | Causa | Evidência | Ação | Quando |
|---|---|---|---|---|
| A | O sorteio com temperatura 1 e sem top-k escolhe tokens BPE improváveis, que formam palavras inventadas e tiram o texto do padrão | Associação: 78% das ruins têm palavra inventada, contra 16% das ok (6,0 contra 1,5 por 100 palavras). O mecanismo não foi testado | Inferência com top-k e temperatura menor, medindo de novo a coerência, a cópia (H5) e a diversidade (H6), que podem piorar | Etapa 4 |
| B | Capacidade e dados limitados: o modelo aprende a estrutura da resenha, mas não o sentido de cada frase | Erros com palavra real (s042, s118); as ruins têm quase o dobro de palavras (22 contra 12), sem direção causal definida | Limite de tamanho da geração na API, como mitigação; dropout 0,1 com mais treino como trabalho futuro | Etapa 4 (limite); futuro (modelo) |
| C | O condicionamento é só a nota, e o modelo mistura o vocabulário de produtos diferentes | 7 amostras de mistura de produtos (a oitava do tipo, s056, é contradição) | Condicionar também pela categoria (extensão do charter) | Futuro |
| D | A nota 2 é rara no treino | 6,2% do treino e 65% de ruins; Spearman −0,9 em 5 notas | Peso por classe ou sobreamostragem da nota 2 no treino | Futuro |
| E | Padrão dos próprios dados | "bla bla bla" em 22 resenhas do treino | Checar repetição na inferência; filtrar na limpeza se houver novo treino | Etapa 4 |
| — | Falta de contexto (`block_size`) | **Descartada:** mediana de 29 dos 128 tokens; a 1ª palavra inventada aparece por volta da 9ª palavra; mesmo a mais longa (s197) quebra a 40% do texto | Nenhuma | — |

## Limitações

- **Uma pessoa só classificou.** Os tipos foram atribuídos à mão, um tipo por amostra.
  Deriva e mistura às vezes se sobrepõem, como em s003 e s198.
- **A medida de palavras inventadas é parcial e, em parte, circular.** Ela pega o defeito típico do
  BPE, mas não a palavra real no lugar errado nem a salada de palavras reais. E quem pontuou a
  fluência viu as mesmas palavras inventadas, então a associação com as amostras ruins é, em parte,
  construída.
- **100 amostras e 20 por nota.** Os números por nota são indicativos, e as causas, hipóteses
  apoiadas por evidência, e não provas. Os testes de verdade das ações A e E vêm na Etapa 4, com as
  mesmas métricas.
- **O rigor do lote v1** aumenta a contagem de amostras ruins (ver `docs/rubric.md`). Pelo menos s059
  sairia da lista pela v1.1.
- **A posição relativa da primeira palavra inventada depende do comprimento do texto.** Por isso, ela
  vem acompanhada da posição absoluta (o número da palavra).
