# Relatório final do nano-review

> Projeto integrador individual. Este relatório resume o projeto e aponta, em cada seção, o documento
> que tem o detalhe. Todos os números vêm de arquivos versionados em [`reports/metrics/`](../reports/metrics/).

## 0. Resumo

**Problema.** Quem desenvolve telas de avaliação, filtros de moderação ou classificadores de
sentimento precisa de muitas resenhas para testar o sistema. Resenhas reais trazem dados pessoais,
restrições de licença e poucos exemplos nas notas raras.

**Solução.** O nano-review gera resenhas curtas de produto em português a partir de uma nota de 1 a
5. É um Transformer só-decoder de ~5,2 milhões de parâmetros, **treinado do zero** em CPU, sem nenhum
modelo pré-treinado. A atenção é escrita à mão. O modelo é servido por uma API com página de
demonstração e roda num contêiner Docker.

**Resultado.** No conjunto de teste, aberto uma única vez, o modelo prevê texto real melhor que o
baseline de trigramas: perplexity por palavra de **68,94**, contra **88,87**, o que é 22% menor. Pelo juiz
de sentimento, as resenhas geradas combinam com a nota pedida com coerência macro de **0,704**. As
resenhas reais do mesmo teste dão 0,755, então o gerador chega a 93% disso. Elas não copiam o treino
(0% de cópia exata) e variam um pouco mais que as reais (distinct-2 de 0,679, contra 0,652). As seis
hipóteses fixadas antes do treino passaram: cinco no teste, e a rubrica humana (H3) na validação. Um clone novo do repositório reproduz o
modelo bit a bit.

**Limite.** O texto é sintético e serve para testar software. Ele não pode ser publicado como
avaliação real (seção 5).

## 1. Problema, usuário e decisão

| | |
|---|---|
| **Usuário** | Desenvolvedor ou analista de QA que precisa de textos sintéticos de resenha |
| **Decisão apoiada** | "Este conjunto de textos serve para testar a minha tela, o meu filtro ou o meu classificador?" |
| **Entrada** | Uma nota de 1 a 5, mais parâmetros de amostragem opcionais (temperatura, top-k, semente) |
| **Saída** | Uma ou mais resenhas curtas em português, marcadas como sintéticas |
| **Erro mais caro** | Texto que contradiz a nota: o usuário culparia o próprio classificador por um erro do gerador |

O segundo erro mais caro é **copiar uma resenha real**. Nesse caso o texto deixa de ser sintético e
pode carregar informação de um cliente. Texto pouco fluente é um erro mais barato, porque o usuário
percebe na hora e descarta.

Por isso a métrica principal é a **coerência com a nota**, e a taxa de cópia é medida em toda
avaliação. O problema, as métricas e os limiares de sucesso foram fixados no
[charter](charter.md) **antes** de existir modelo, para que a avaliação não fosse ajustada ao
resultado.

## 2. Dados e engenharia de dados

### 2.1 Origem e os 5 Vs

A fonte é o **B2W-Reviews01**: 132.373 resenhas reais da Americanas.com, de janeiro a maio de 2018,
publicadas pela B2W Digital sob a licença CC BY-NC-SA 4.0. O arquivo vem de uma URL fixada num commit
do GitHub da americanas-tech. O SHA-256 e o MD5 estão registrados no
[manifesto de dados](data_manifest.md), que foi escrito antes da EDA.

| V | No B2W-Reviews01 | Consequência no projeto |
|---|---|---|
| **Volume** | 132.373 resenhas (49 MB); depois da limpeza, 126.055, com mediana de 16 palavras | Cabe na memória. O processamento usa pandas, numa máquina só, e um cluster não teria justificativa |
| **Velocidade** | Base estática, fixada num commit. Não chega dado novo | Processamento em lote (batch), refeito pelo DVC só quando algo muda |
| **Variedade** | Texto livre e informal, mais campos estruturados e o perfil do avaliador | Só o texto e a nota entram no modelo. O BPE lida com gírias e erros de digitação sem dicionário fixo |
| **Veracidade** | Textos vazios, duplicados e com notas conflitantes. A documentação da fonte diverge do arquivo (o separador é vírgula). A nota nem sempre combina com o texto: nas resenhas reais da validação, a coerência medida pelo juiz é só 0,728 | Limpeza e dedupe antes do split. O teto da coerência é 0,728, não 1,0 |
| **Valor** | Resenhas reais em português, com a nota dada pelo próprio cliente | A nota vira o condicionamento: é ela que permite pedir o tom da resenha |

A EDA está em [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). O achado que mais pesou no
projeto foi o **desbalanceamento das notas**: a nota 5 tem 36,2% das resenhas, e a nota 2, só 6,3%
([`eda_ratings.png`](../reports/figures/eda_ratings.png)). Esse desequilíbrio reaparece no fim: a nota
2 é a pior na rubrica humana (seção 3).

### 2.2 Pipeline e camadas

O pipeline é um conjunto de estágios do DVC ([`dvc.yaml`](../dvc.yaml)), com todos os parâmetros num
único [`params.yaml`](../params.yaml). Cada estágio declara entradas, código e saídas. O
`dvc repro` só refaz o que mudou, e o [`dvc.lock`](../dvc.lock) guarda o hash de cada saída.

| Camada | Conteúdo | Formato | Regra |
|---|---|---|---|
| `data/raw` | O CSV original | CSV (49 MB) | Nunca é editado; é conferido pelo hash |
| `data/silver` | Texto limpo, sem dados pessoais, sem duplicatas, só com as 4 colunas usadas | Parquet (11,5 MB) | Gerado pelo estágio `clean` |
| `data/gold` | `train`, `validation` e `test`, separados por produto | Parquet, um arquivo por conjunto | Gerado pelo estágio `split` |

As camadas seguem a arquitetura medalhão: `raw` é a cópia fiel da fonte, `silver` é o dado limpo e
confiável, e `gold` é o dado pronto para o uso final, que aqui é o treino e a avaliação do modelo.

O Parquet substitui o CSV a partir da segunda camada por três motivos. Ele é colunar, então lê só as
colunas pedidas. Ele preserva os tipos: a nota continua inteira. E ele não tem as ambiguidades de
separador e de aspas que o CSV da fonte tinha.

O processamento é em **lote, e não em streaming**, porque a fonte não muda e o treino é offline. Um
fluxo contínuo de dados mudaria a validação a cada leitura, e os experimentos deixariam de ser
comparáveis.

### 2.3 Qualidade: limpeza, dados pessoais e duplicatas

Cada regra de limpeza é uma função pequena e testada em
[`src/nano_review/data/clean.py`](../src/nano_review/data/clean.py), e vem de uma decisão registrada
na EDA. O funil está em [`cleaning.json`](../reports/metrics/cleaning.json):

| Etapa | Linhas removidas | Por quê |
|---|---:|---|
| Bruto | 132.373 | — |
| Sem texto | 3.275 | Não há o que aprender |
| Com dado pessoal no texto (e-mail, telefone, CPF) | 0 | O filtro existe e é testado, mas não encontrou casos |
| Vazio depois da normalização | 105 | Sobrava só HTML, URL ou símbolos |
| Mesmo texto com notas diferentes | 521 | Manter um dos dois seria escolher o rótulo ao acaso |
| Texto duplicado | 2.417 | Uma cópia no treino e outra no teste seria vazamento |
| **Processado** | **126.055** | |

**Dados pessoais.** As colunas de perfil do avaliador (id, ano de nascimento, gênero e UF) nunca saem
de `data/raw`. O texto passa por um filtro de e-mail, telefone e CPF. Cada padrão exige uma
"assinatura" do dado, como uma arroba com domínio, um DDD entre parênteses ou a palavra "cpf". Sem
isso, preço, ano e número de pedido seriam confundidos com telefone. O mesmo filtro protege a saída da
API.

**Normalização.** Ela padroniza o Unicode (NFC), tira HTML e URLs e limita letras repetidas a 3
("muuuuuito" vira "muuuito"). Isso mantém a ênfase, que é estilo real de resenha, e corta variações
que o tokenizador teria de aprender uma a uma. **Caixa e acentos ficam**, porque a saída precisa
parecer resenha real.

**Duplicatas.** São comparadas por uma chave que ignora caixa, pontuação e espaços. A comparação é
feita **no corpus inteiro**, e não por produto: o mesmo texto-modelo em dois produtos diferentes
vazaria entre treino e teste.

### 2.4 Split por produto e prevenção de vazamento

O split separa **produtos**, e não linhas: nenhum produto aparece em dois conjuntos. Resenhas do mesmo
produto repetem nome, defeito e vocabulário. Com o mesmo produto no treino e no teste, o teste mediria
a memória do produto, e não a capacidade de generalizar.

Os produtos são embaralhados com semente fixa e vão enchendo os conjuntos até a **contagem de
resenhas** chegar a 80/10/10. Sortear 10% dos *produtos* não bastaria, por causa da cauda longa: dos
46.206 produtos que sobram depois da limpeza, um tem 779 resenhas, e 72% têm uma só
([`eda_reviews_per_product.png`](../reports/figures/eda_reviews_per_product.png)). Resultado
([`split.json`](../reports/metrics/split.json)):

| Conjunto | Resenhas | Fração | Produtos | Nota 2 |
|---|---:|---:|---:|---:|
| `train` | 100.831 | 80,0% | 37.068 | 6,2% |
| `validation` | 12.617 | 10,0% | 4.786 | 6,9% |
| `test` | 12.607 | 10,0% | 4.352 | 6,5% |

A distribuição das notas ficou parecida nos três conjuntos sem precisar de estratificação.

**As quatro regras contra vazamento:**

1. **Dedupe antes do split.** Uma resenha repetida não pode cair em dois conjuntos.
2. **Split por produto**, como descrito acima.
3. **O tokenizador BPE é treinado só no `train`.** A validação é apenas medida por ele.
4. **O `test.parquet` foi lido uma única vez**, pelo estágio `final_eval`, a partir de um commit que
   já tinha o pré-registro publicado ([`final_test.md`](final_test.md)). Depois disso, o estágio foi
   congelado (`frozen: true`) para que nenhum `dvc repro` o rode de novo.

## 3. A IA própria

### 3.1 Baseline e juiz de sentimento

Cada lado da avaliação tem um piso trivial e um baseline de verdade ([charter](charter.md), seção 6).
O piso mostra quanto se ganha de graça. Sem ele, um número alto poderia ser só efeito do
desbalanceamento.

| | Piso trivial | Baseline | Resultado na validação |
|---|---|---|---|
| **Gerador** | Unigrama por nota (sorteia palavras pela frequência) | **Trigrama** de palavras, interpolado, treinado no formato `<nota_k> texto <eos>` | Perplexity por palavra de 90,2; coerência macro de 0,386 |
| **Juiz** | Classe majoritária (responde sempre "positivo") | TF-IDF + regressão logística em 3 classes (notas 1 e 2 negativo, 3 neutro, 4 e 5 positivo) | Acurácia de 0,801 contra 0,592; F1 macro de 0,729 contra 0,248 |

O trigrama é o modelo de linguagem mais simples que funciona: ele só lembra as duas palavras
anteriores. Se o Transformer não o superasse, não se justificaria. A limitação do trigrama é
justamente o que o projeto quer resolver: o token `<nota_k>` sai da janela logo nas primeiras
palavras, e o trigrama perde o tom. Por isso a coerência dele (0,386) fica perto do acaso entre
3 classes. Os pesos da interpolação foram escolhidos numa fatia do treino, e não na validação.

O juiz é o instrumento que mede a coerência das resenhas geradas. Ele não é perfeito: na classe
neutra, acerta só 54%. Por isso a referência da coerência é a das **resenhas reais** medidas pelo
mesmo juiz (0,728 na validação), e não 1,0. Fontes: [`judge.json`](../reports/metrics/judge.json) e
[`baseline.json`](../reports/metrics/baseline.json).

### 3.2 Tokens: o BPE

O texto vira números por um tokenizador **BPE** (byte-pair encoding) treinado **só no `train`**. O BPE
começa com caracteres e junta, repetidamente, o par vizinho mais frequente, até chegar a 8.000
tokens. Palavras comuns viram um token só, e palavras raras ou com erro de digitação viram pedaços.
Por isso a taxa de `<unk>` na validação é praticamente zero (0,0002%): o modelo nunca encontra uma
palavra que não consiga escrever ([`tokenizer.json`](../reports/metrics/tokenizer.json)).

Cada exemplo de treino vira a sequência `<bos> <nota_k> texto <eos>`. O token da nota é o que permite
pedir o tom. O `<eos>` ensina o modelo a parar. O passo a passo, com um BPE didático escrito à mão, está
em [`notebooks/02_tokenization.ipynb`](../notebooks/02_tokenization.ipynb).

**Contexto (`block_size`) de 128 tokens.** Com o vocabulário final, 1,9% das resenhas de treino passam
de 128 tokens, e só 0,24% passam de 256. O custo da atenção cresce com o quadrado do contexto, e o
produto é um gerador de resenhas **curtas**. Por isso as 1.829 resenhas mais longas ficaram fora do
treino, em vez de serem cortadas: um exemplo cortado não termina em `<eos>`, e o modelo não aprenderia
a parar.

### 3.3 Do bigrama ao Transformer: forward e backward

Antes do Transformer, um **bigrama neural** em PyTorch serviu para mostrar o ciclo de treino em
pequena escala ([`notebooks/03_forward_backward.ipynb`](../notebooks/03_forward_backward.ipynb)). O
ciclo é o mesmo nos dois modelos:

1. **Forward:** os tokens entram, e o modelo devolve, para cada posição, um *logit* por token do
   vocabulário, uma pontuação para "qual vem a seguir".
2. **Loss:** a entropia cruzada, a média de `-log` da probabilidade que o modelo deu ao token
   certo. É zero se o modelo dá probabilidade 1 ao token certo, e cresce quanto mais ele erra. A
   perplexity é `exp(loss)`.
3. **Backward:** o `loss.backward()` aplica a regra da cadeia de trás para frente e calcula o
   gradiente da loss em relação a **cada peso**. Na primeira etapa da volta, o gradiente em relação aos
   logits tem uma forma fechada: `softmax(logits) − one_hot(alvo)`. Ele empurra para cima o logit do
   token certo e para baixo os outros, cada um na proporção da probabilidade que recebeu. O notebook
   calcula esse gradiente à mão e confere que ele é igual ao do `backward()`.
4. **Atualização:** o otimizador AdamW move cada peso um pouco na direção que diminui a loss.

O bigrama mostra também o limite que motivou o Transformer: a nota só influencia a primeira palavra,
porque o modelo só enxerga o token anterior.

### 3.4 Arquitetura: Transformer só-decoder com a atenção à mão

O modelo está em [`src/nano_review/models/transformer.py`](../src/nano_review/models/transformer.py). A
atenção foi escrita à mão, sem `nn.MultiheadAttention`, para que cada passo fique visível. Os shapes
do forward são estes (B = lote, T = posições, C = 256, H = 4 cabeças, D = C/H = 64, V = 8.000):

```text
tokens (B, T)
  -> embedding de token + embedding de posição     (B, T, C)
  -> 4 blocos, cada um:
       x = x + Atenção(LayerNorm(x))              (B, T, C)
       x = x + MLP(LayerNorm(x))                  (B, T, C)
  -> LayerNorm final -> camada de saída            (B, T, V)  = logits
```

**A atenção.** Em cada posição, três projeções do mesmo vetor:
- a **query** (Q), "o que eu procuro";
- a **key** (K), "o que eu ofereço";
- o **value** (V), "o que eu passo adiante".

O peso que a posição `i` dá à posição `j` é `softmax(Q_i · K_j / √D)`, e a saída de `i` é a média
dos values ponderada por esses pesos.

- **Divisão por √D:** sem ela, com D grande, os scores ficam grandes e o softmax satura, com um peso
  perto de 1 e o resto perto de 0. Nessa região o gradiente quase some.
- **Máscara causal:** a posição `i` só vê as posições `j ≤ i`. Sem ela, o modelo poderia olhar para
  o próprio token que deve prever, e o treino não ensinaria nada.
- **Várias cabeças:** cada uma faz a atenção num pedaço de 64 dimensões e pode aprender a olhar para
  coisas diferentes.

Os outros componentes:
- **Embedding de posição:** a atenção sozinha compara todos com todos e não sabe a ordem das palavras.
- **MLP:** a atenção mistura informação entre posições, e o MLP processa o que cada posição juntou.
- **Conexões residuais e LayerNorm antes de cada sub-camada:** dão ao gradiente um caminho direto até
  as primeiras camadas e deixam o treino estável.
- **Pesos amarrados:** a camada de saída reutiliza a tabela de embedding.

O modelo final tem **5.240.320 parâmetros**.

**O que a atenção aprendeu** ([`notebooks/04_attention.ipynb`](../notebooks/04_attention.ipynb),
[`attention_maps.png`](../reports/figures/attention_maps.png)). Sem nada programado para isso, as
cabeças se especializaram:
- uma olha para o próprio token;
- outra olha para o token anterior, que é o papel do bigrama, agora como uma peça entre várias;
- outras ligam as palavras que decidem o tom: "recomendo" volta a um "Não" anterior, e "troca"
  volta a "quebrado".

A cabeça mais atenta à nota dá a `<nota_k>` 0,28 do peso, quase 5 vezes a atenção uniforme. É assim
que a nota chega ao fim da resenha, e é o que o trigrama não consegue fazer.

### 3.5 Treino

O laço está em [`src/nano_review/training/train.py`](../src/nano_review/training/train.py). A cada
passo: forward, loss, `zero_grad`, `backward`, corte da norma do gradiente em 1,0 e `optimizer.step`.

| Escolha | Valor | Por quê |
|---|---|---|
| Otimizador | AdamW, weight decay 0,1 só nas matrizes | O Adam ajusta o passo de cada peso pela história do gradiente. O weight decay puxa os pesos para perto de zero e reduz o sobreajuste |
| Learning rate | 0,001, com warmup de 200 passos e decaimento em cosseno até 10% disso | No começo, os gradientes são instáveis, e um passo grande logo de saída pode estragar o modelo |
| Corte do gradiente | norma máxima de 1,0 | Um lote atípico não dá um salto grande nos pesos |
| Lote e duração | 32 resenhas, 15.000 passos (~70 min em CPU) | O orçamento do projeto em CPU |
| Seleção do checkpoint | Melhor validação; early stopping com paciência de 8 avaliações | Nunca fica o modelo do último passo só por ser o último |
| Semente | 42, fixada em `random`, `numpy` e `torch` | Mesma configuração, mesmo resultado (seção 4) |

A curva de treino e validação ([`train_curve.png`](../reports/figures/train_curve.png)) ainda caía no
último passo, então o early stopping não chegou a disparar. A loss de validação final é 3,71 por token
([`train.json`](../reports/metrics/train.json)).

### 3.6 Experimentos com hipóteses

Os experimentos ([`docs/experiments.md`](experiments.md)) seguiram regras escritas antes de qualquer run:
- **uma variável por vez**, contra a run 0;
- a hipótese de cada experimento escrita antes de rodar;
- a perplexity por palavra na validação como métrica de decisão, e a coerência como guarda;
- o ruído entre sementes medido antes de tudo (E0): uma melhora só conta se for maior que o dobro da
  diferença entre duas sementes.

Cada run foi registrada no MLflow.

| Run | Mudança | Perplexity por palavra | Coerência | Leitura |
|---|---|---:|---:|---|
| run 0 | configuração da Etapa 2 | 218,5 | 0,617 | Referência; o diagnóstico foi subajuste |
| E0 | outra semente | 219,3 | 0,642 | Mede o ruído |
| E2 | learning rate 3e-4 → 1e-3 | 141,7 | 0,673 | Adotado |
| E3 | `d_model` 128 → 256 | 140,1 | 0,670 | Adotado, com custo 2× |
| E4 | sem dropout | 193,1 | 0,608 | Adotado: o modelo subajustava, e não sobreajustava |
| E5 | vocabulário 4.000 → 8.000 | 174,5 | 0,627 | Adotado; a previsão (±5%) errou: ganho de 20% |
| E6 | **sem o token da nota** | 224,6 | **0,312** | Ablação: sem a nota, a coerência cai para o acaso |
| E1 | 30.000 passos | 97,9 | 0,676 | Mais treino ajuda muito |
| **final** | E2 + E3 + E4 + E5, 15.000 passos | **69,3** | **0,721** | Checkpoint final |

O E6 é o experimento mais importante para a defesa. Ele prova que a coerência **vem da nota**: o
mesmo modelo, retreinado sem `<nota_k>`, gera texto igualmente fluente, mas sem relação com a nota
pedida. Uma das previsões errou (E5), e o erro está registrado como resultado.

**A perplexity por palavra** é a métrica que permite comparar modelos com vocabulários diferentes, e
o Transformer com o trigrama. A probabilidade de um texto não depende de como ele foi quebrado em
tokens, então a soma dos `log` das probabilidades é dividida pelo número de **palavras**, e não de
tokens.

### 3.7 Validação, teste final e rubrica humana

As hipóteses H1 a H6 e os seus limiares foram fixados no charter **depois** de medir o baseline e
**antes** de treinar o Transformer. Passaram na validação, e depois no teste, aberto uma única vez a
partir de um commit que já tinha o pré-registro publicado ([`final_test.md`](final_test.md)):

| # | Critério | Limiar | Trigrama | Validação | **Teste** |
|---|---|---|---:|---:|---:|
| H1 | Perplexity por palavra | ≤ 72 | 90,2 (88,87 no teste) | 69,31 | **68,94** |
| H2 | Coerência macro | ≥ 0,57 | 0,386 | 0,721 | **0,704** |
| H3 | Amostras ruins na rubrica humana | < trigrama | 85% | 37% | — (só validação) |
| H4 | Cópia exata do treino | ≤ 1% | 0,0% | 0,0% | **0,0%** |
| H5 | Cópia de 8 palavras seguidas | ≤ 7% | 0,2% | 3,1% | **1,8%** |
| H6 | Distinct-2 (diversidade) | ≥ 0,59 | 0,71 | 0,681 | **0,679** |

No teste, a perplexity até melhorou um pouco em relação à validação. O trigrama também melhorou
(90,2 → 88,87), o que indica que as resenhas do teste são um pouco mais fáceis, e não que houve
vazamento. H2, H4, H5 e H6 foram medidas em 1.000 resenhas novas, geradas com uma semente nunca usada.

**A rubrica humana (H3)** ([`docs/rubric.md`](rubric.md)) foi definida **antes** de ler qualquer
amostra. Ela tem quatro critérios de 1 a 3: fluência, plausibilidade, coerência com a nota e repetição.
Uma amostra é ruim se tem nota 1 em qualquer critério. O lote foi pontuado **às cegas**: 100 amostras
do Transformer, 100 do trigrama e 100 resenhas reais como controle, embaralhadas e sem a origem. A
chave só foi aberta no fim.

| Fonte | Amostras ruins | Na nota 2 |
|---|---:|---:|
| Resenhas reais (controle) | 3% | 0% |
| **Transformer** | **37%** | 65% |
| Trigrama | 85% | 90% |

Fonte: [`rubric.json`](../reports/metrics/rubric.json).

### 3.8 Análise de erros

As 37 amostras ruins do Transformer foram classificadas uma a uma
([`docs/error_analysis.md`](error_analysis.md)). **Quase todo erro é de forma, e não de tom:**
- 33 das 37 são deriva (o texto começa bem e se perde), salada (sem sentido desde o começo) ou mistura
  de produtos;
- só 3 erram o tom.

As causas foram medidas, e não só descritas:

| Causa | Evidência | Ação |
|---|---|---|
| O sorteio escolhe tokens improváveis, que formam palavras inventadas | 78% das amostras ruins têm palavra inventada, contra 16% das boas | Testar temperatura e top-k (3.9) |
| Capacidade limitada do modelo | As ruins têm quase o dobro de palavras (22 contra 12) | Limite de tamanho na geração |
| O condicionamento é só a nota | 7 amostras misturam produtos | Trabalho futuro: condicionar pela categoria |
| A nota 2 é rara no treino (6,2%) | 65% de amostras ruins na nota 2 | Trabalho futuro: peso por classe |
| Falta de contexto | **Descartada:** a mediana é de 29 dos 128 tokens | Nenhuma |

### 3.9 Inferência: temperatura, top-k e filtros

O modelo gera um token por vez: calcula a distribuição do próximo token, sorteia um e repete até o
`<eos>`.
- **A temperatura** divide os logits antes do softmax. Abaixo de 1, concentra a probabilidade nos
  tokens mais prováveis.
- **O top-k** sorteia só entre os k tokens mais prováveis.

A análise de erros sugeria que os dois reduziriam as palavras inventadas, e reduziram. Mas a escolha
foi feita por uma regra escrita **antes** da varredura: a combinação precisa passar em H2, H4, H5 e H6
na validação ([`docs/inference.md`](inference.md), seção 4.1). Das 12 combinações, **só a temperatura
1 sem top-k passou nas quatro**:

- Com top-k 50, a coerência sobe para 0,763, mas a diversidade cai para 0,511 e reprova no H6.
- Com temperatura mais baixa, a diversidade cai primeiro: 0,9 já reprova no H6. A partir de 0,8, a
  cópia do treino também passa do limite (H5, e depois H4).

É o preço de concentrar o sorteio: o texto fica mais seguro, mas repetitivo e mais perto do que o
modelo decorou.

O produto acrescenta filtros a cada texto gerado:
- até 83 tokens;
- nenhuma palavra repetida 3 vezes seguidas;
- nenhum dado pessoal ou URL;
- até 5 tentativas.

No teste, das 1.000 resenhas pedidas, as 1.000 foram entregues, e 97,3% saíram na primeira tentativa.

## 4. MLOps e produto

### 4.1 Repositório, ambiente e qualidade

O código fica no Git, com um commit por entrega e mensagens que dizem o que mudou e por quê. O
[README](../README.md) leva de um clone novo até a demonstração. O detalhe dos comandos fica na
[referência](reference.md).

- **Ambiente:** Python 3.12, torch CPU e um `requirements-lock.txt` com as versões exatas de todas as
  dependências, inclusive as indiretas. O `requirements.txt` diz do que o projeto depende, e o lock
  diz com quais versões os números foram obtidos.
- **Porta de qualidade:** o `make check` roda `ruff` (estilo), `mypy --strict` (tipos) e `pytest`, e
  precisa passar antes de qualquer commit. São **313 testes**:

| Área | Testes | Exemplos do que é testado |
|---|---:|---|
| `data` | 91 | Cada regra de limpeza, os padrões de dado pessoal (inclusive os falsos positivos), o split sem produto repetido, o DataLoader |
| `evaluation` | 74 | Métricas de coerência, cópia e distinct-n, a rubrica cega, a regra de escolha da decodificação |
| `inference` | 41 | Temperatura, top-k, os filtros e as novas tentativas do gerador |
| `api` | 36 | Rotas, validação do pedido, 422 e 503, nenhum caso que devolva 500, e o link certo no log do Docker |
| `baseline` | 30 | Trigrama, unigrama e o juiz de sentimento |
| `models` | 14 | Shapes do forward, a máscara causal (o futuro não muda o passado) e o gradiente calculado à mão igual ao do `backward()` |
| `tokenization` | 14 | O BPE e o BPE didático |
| `training` | 11 | Agenda do learning rate, early stopping e o export do MLflow |

### 4.2 DVC e MLflow

- **DVC:** versiona os dados e os modelos, que são grandes demais para o Git, e define o pipeline
  ([`dvc.yaml`](../dvc.yaml)). São dez estágios, de `clean` a `final_eval`, com todos os parâmetros
  em [`params.yaml`](../params.yaml). O `dvc.lock` registra o hash de cada entrada e saída, e o
  `dvc repro` refaz só o que mudou.
- **MLflow:** cada run de treino, avaliação e decodificação grava parâmetros, curvas e métricas H1 a
  H6. O mesmo nome aparece no `dvc exp show` e no MLflow, para os dois se cruzarem. Como o banco do
  MLflow fica fora do Git, as runs são exportadas para
  [`reports/experiments/`](../reports/experiments/).

### 4.3 API e página de demonstração

A API, em FastAPI ([`src/nano_review/api/`](../src/nano_review/api/)), serve o gerador com a
decodificação e os filtros escolhidos na seção 3.9.

| Rota | Função |
|---|---|
| `GET /` | Página de demonstração: escolher a nota e a quantidade, gerar e copiar as resenhas |
| `GET /health` | Se o modelo está carregado, o número de parâmetros e o hash do checkpoint |
| `POST /generate` | As resenhas para uma nota, com temperatura, top-k e semente opcionais |
| `GET /docs` | Documentação automática, onde cada rota pode ser testada |

As decisões de desenho estão em [`docs/inference.md`](inference.md), seção 4.4. As principais:
- **O modelo carrega uma vez**, quando a API sobe, e não a cada pedido.
- **A validação nunca devolve 500.** Um pedido inválido dá 422 com o motivo. Um texto que cai nos
  filtros 5 vezes dá 503 com os motivos.
- **A semente usada volta na resposta**, para que um bom texto possa ser repetido.
- **Toda resposta traz `"synthetic": true` e o aviso** de que o texto é sintético.
- **O texto do modelo entra na página como texto, e nunca como HTML.** Nada do que o modelo escreve
  vira código no navegador.
- **A página funciona sem internet**, sem bibliotecas externas.

No teste final, uma resenha levou 111 ms na mediana e 510 ms no percentil 95.

### 4.4 Docker e publicação do modelo

- **Docker:** a imagem ([`Dockerfile`](../Dockerfile)) traz a API, a página e o modelo final, e roda
  sem `.venv`, DVC nem dados.
  - A base e as bibliotecas têm versão exata.
  - O `.dockerignore` lista só o que **pode** entrar.
  - A API roda como um usuário sem privilégios de root.
  - Um `HEALTHCHECK` chama o `/health` a cada 30 s.
- **Release:** o remote do DVC é uma pasta local. Por isso, o checkpoint e o tokenizador estão
  publicados no Release `v0.1.0` do GitHub, e o README traz o SHA-256 dos dois para conferência.

### 4.5 Reprodução limpa do zero

Para provar que tudo isso funciona fora da máquina de desenvolvimento, o projeto foi reproduzido a
partir de um clone novo, seguindo só o README ([`docs/reproduction.md`](reproduction.md)):
- o ambiente foi instalado do zero, sem cache;
- o CSV foi baixado da fonte original;
- o pipeline inteiro foi refeito em ~2 h;
- a API e o Docker foram testados com o modelo do Release.

**Todos os artefatos saíram bit a bit iguais, inclusive o checkpoint**, e o modelo do Release é
idêntico ao retreinado. A reprodução também encontrou dois defeitos, que foram corrigidos: o
`dvc pull` do README só funcionava na máquina original, e as versões das dependências mudavam a cada
instalação.

## 5. Ética, limitações e trabalho futuro

### 5.1 Uso pretendido e uso indevido

O nano-review gera **texto de teste para software**: telas de avaliação, filtros de moderação e
classificadores de sentimento ([charter](charter.md), seção 9).

**Ele não deve ser usado para:**
- **Publicar avaliações** em loja, marketplace ou rede social. Uma resenha falsa engana o consumidor
  e, no Brasil, pode configurar publicidade enganosa (Código de Defesa do Consumidor).
- **Servir de informação sobre um produto.** O modelo inventa fatos, como voltagem, tamanho e
  defeitos, porque é condicionado só pela nota, e não pelo produto.
- **Substituir dados reais no treino de um classificador que vá para produção** sem validação em
  dados reais. O texto herda os vieses do B2W e os do próprio gerador.

**O que o projeto faz para reduzir o risco:**
- Toda resposta da API vem marcada como sintética, com um aviso, e a página mostra o mesmo aviso.
- O gerador filtra dados pessoais e URLs na saída.
- A taxa de cópia do treino é medida em toda avaliação: 0% de cópia exata no teste.

Nada disso impede um uso de má-fé: um texto copiado da página perde o aviso. É por isso que o uso
pretendido está escrito no README, na licença e aqui.

### 5.2 Dados, privacidade e licença

- **Dados pessoais:**
  - as colunas de perfil do avaliador (id, ano de nascimento, gênero e UF) nunca saem de `data/raw`;
  - o texto passa por um filtro de e-mail, telefone e CPF, na limpeza e na saída da API.
  - O filtro não encontrou casos no dataset, mas a origem não documenta como foi anonimizada, então
    ele continua necessário.
- **Licença:** os dados são CC BY-NC-SA 4.0. Tudo o que deriva deles segue a mesma licença e não pode
  ter uso comercial ([`LICENSE-DATA.md`](../LICENSE-DATA.md)). Isso inclui o tokenizador, os modelos,
  as resenhas geradas e os relatórios. O código é MIT ([`LICENSE`](../LICENSE)).
- **Viés da fonte:** as resenhas são de uma única loja (Americanas.com), de cinco meses de 2018. O
  modelo aprende o vocabulário, os produtos e o tom desse recorte. Ele não representa o português de
  todas as plataformas nem de todas as regiões.

### 5.3 Limitações do modelo e da avaliação

- **Fluência:** pela rubrica, 37% das amostras ainda são ruins, quase todas por forma: o texto começa
  bem e se perde, ou mistura produtos. O modelo gera 3,8 palavras inventadas a cada 100, e as resenhas
  reais do lote da rubrica têm 0,8.
- **Resenhas mais curtas que as reais:** 17,4 palavras em média no teste, contra 26,2 das reais.
- **A nota 2 é a pior** (65% de amostras ruins), porque é só 6,2% do treino. Na coerência, a classe
  neutra (nota 3) fica em 0,52, contra ~0,80 das outras duas.
- **O juiz de sentimento é um instrumento imperfeito:** acerta só 54% na classe neutra. A coerência é
  sempre comparada com a das resenhas reais medidas pelo mesmo juiz, mas o ruído dele entra nos
  números.
- **A rubrica humana teve um avaliador só, o autor**, com 100 amostras por fonte. A avaliação foi cega
  e teve um controle real, o que reduz o viés, mas não substitui vários avaliadores com medida de
  concordância.
- **A reprodução bit a bit foi na mesma máquina.** Em outra CPU, as somas em ponto flutuante podem sair
  em outra ordem, e o checkpoint pode não ser idêntico.
- **A API não tem autenticação nem limite de pedidos.** Ela foi feita para a demonstração e para
  testes locais, e por isso só escuta em `127.0.0.1`.

### 5.4 Trabalho futuro

| Ideia | Ataca qual limitação | Evidência |
|---|---|---|
| Condicionar também pela categoria do produto | Mistura de produtos | 7 das 37 amostras ruins (seção 3.8) |
| Peso por classe ou sobreamostragem da nota 2 | Nota rara | 65% de ruins na nota 2 |
| Modelo maior com dropout 0,1 e mais passos | Fluência | A curva ainda caía no último passo; o E1 (30.000 passos) ganhou 55% de perplexity |
| Mais avaliadores na rubrica, com medida de concordância | Avaliação humana | Um avaliador só |

## 6. Referências

- **Dataset:** Real, L.; Oshiro, M.; Mafra, A. (2019). *B2W-Reviews01: an open product reviews
  corpus*. Symposium in Information and Human Language Technology (STIL). Publicado pela B2W Digital
  em <https://github.com/americanas-tech/b2w-reviews01>, sob a licença CC BY-NC-SA 4.0.
- **Licença dos dados:** Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional,
  <https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt_BR>.
- **Transformer:** Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural
  Information Processing Systems (NeurIPS).
- **BPE para texto:** Sennrich, R.; Haddow, B.; Birch, A. (2016). *Neural Machine Translation of Rare
  Words with Subword Units*. Annual Meeting of the Association for Computational Linguistics (ACL).
- **AdamW:** Loshchilov, I.; Hutter, F. (2019). *Decoupled Weight Decay Regularization*.
  International Conference on Learning Representations (ICLR).
