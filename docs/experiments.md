# Experimentos da Etapa 3

> As hipóteses abaixo foram escritas **antes** de rodar qualquer experimento (19/11/2026). A seção
> "Resultados" é preenchida depois, run por run, sem mudar o que foi previsto aqui. Uma previsão
> errada também é resultado: ela mostra o que o modelo ensinou que a gente não sabia.

## Regras

1. **Uma variável por vez**, sempre contra a run 0. Se duas coisas mudam juntas, não dá para saber
   qual delas causou o efeito.
2. **Métrica de decisão: perplexity por palavra na validação inteira** (H1). É a única métrica que
   compara modelos com vocabulários diferentes (E5) e que usa as 12.617 resenhas. A loss por token
   não serve para isso: com outro vocabulário, "token" muda de tamanho.
3. **Guarda: a coerência macro** (H2). Uma mudança que melhore a perplexity e derrube a coerência em
   mais de 0,03 não é adotada. Esse valor é cerca de 2 erros padrão: com 200 amostras por nota, o erro
   padrão da coerência macro é de uns 1,6 ponto (charter, seção 7).
4. **Ruído antes de efeito.** O E0 repete a run 0 com outra semente. Uma mudança só conta como
   melhora se a diferença for **maior que o dobro** da diferença entre as duas sementes.
5. **Orçamento:** ~1 h por run e ~6 h no total, em CPU. Antes de passar disso, conversamos.
6. Tudo é escolhido pela **validação**. O teste continua fechado até a Etapa 4.

Cada run é um `dvc exp run -n <nome> --set-param ...`. O nome aparece igual no MLflow e no `dvc exp show`.

## Referência: run 0 (configuração da Etapa 2)

`d_model` 128, 4 camadas, 4 cabeças, dropout 0,1, lote 32, LR 3e-4, 6.000 passos (~2 épocas),
vocabulário 4.000. O treino leva ~13 min, a avaliação ~2 min.

| Perplexity por palavra | Coerência macro | Distinct-2 | Cópia exata | Cópia de 8 palavras | Palavras por amostra |
|---:|---:|---:|---:|---:|---:|
| 218,5 | 0,617 | 0,68 | 0,3% | 1,5% | 16,9 |

A validação ainda caía no último passo, e as losses de treino e de validação, medidas por token,
são iguais (4,35 contra 4,34). O diagnóstico é **subajuste**: o modelo ainda não aprendeu o que
poderia, e não está decorando.

## Fase 1: triagem no orçamento curto (6.000 passos)

### E0: ruído entre sementes

- **Muda:** `train.seed` 42 → 43, com o resto igual à run 0.
- **Por quê:** sem saber quanto o resultado varia por acaso (outra inicialização dos pesos, outra
  ordem dos lotes), uma diferença de 3% pode ser efeito ou sorte.
- **Previsão:** perplexity a menos de 3% da run 0 e coerência a menos de 0,03.
- **Custo:** ~15 min.

### E2: learning rate maior

- **Muda:** `train.learning_rate` 3e-4 → 1e-3.
- **Por quê:** em subajuste, o problema é chegar longe em poucos passos. Um LR maior dá passos
  maiores. O corte do gradiente em 1,0 e o warmup protegem contra a instabilidade, que é o risco
  de um LR alto. Comparar pelo menos duas learning rates mostra se o LR é o gargalo do treino.
- **Previsão:** **melhora** a perplexity. Se a curva oscilar ou a loss subir no começo, o LR passou
  do ponto.
- **Custo:** ~15 min.

### E3: modelo maior

- **Muda:** `train.d_model` 128 → 256 (de ~1,3 M para ~4,2 M parâmetros).
- **Por quê:** mais capacidade por camada. Modelos maiores costumam aprender mais **por passo**.
- **Previsão:** **melhora** a perplexity com o mesmo número de passos, mas o treino custa ~3× mais
  CPU. A pergunta prática é se o ganho compensa o tempo, porque no orçamento de 1 h ele treina menos
  passos.
- **Custo:** ~40 min.

### E4: sem dropout

- **Muda:** `train.dropout` 0,1 → 0,0.
- **Por quê:** o dropout é uma regularização: ele atrapalha de propósito para o modelo não decorar. Em
  subajuste, não há decoreba para evitar, e o dropout só torna o aprendizado mais lento.
- **Previsão:** **pequena melhora** (1 a 5%). Se o E1 mostrar a validação se afastando do treino, o
  dropout volta, e um valor maior (0,2) passa a ser testado.
- **Custo:** ~15 min.

### E5: vocabulário maior

- **Muda:** `tokenizer.vocab_size` 4.000 → 8.000. Isso retreina o tokenizador, que continua treinado
  só no treino.
- **Por quê:** com mais tokens, as palavras são quebradas em menos pedaços (hoje 1,23 token por
  palavra). Cada resenha fica mais curta, e o `block_size` cobre mais texto. Em compensação, cada
  token aparece menos vezes no treino, e o embedding tem o dobro de linhas para aprender.
- **Previsão:** **efeito pequeno** (dentro de ±5%), de direção incerta. Esta é a previsão menos
  confiante da lista.
- **Custo:** ~20 min.

## Fase 2: treinos longos

### E1: treinar mais tempo

- **Muda:** `train.max_steps` 6.000 → 30.000 (~10 épocas), com o early stopping (paciência 8) ligado.
- **Por quê:** é a consequência direta do diagnóstico de subajuste.
- **Previsão:** a **maior melhora** de todos os experimentos. A perplexity cai pelo menos 15% em
  relação à run 0. Se as curvas de treino e de validação começarem a se separar, aparece o primeiro
  sinal de sobreajuste, e o early stopping guarda o melhor ponto.
- **Custo:** ~1 h.

### E6: sem a nota (ablação)

- **Muda:** o modelo é treinado **sem** o token `<nota_k>`; a sequência vira `<bos> texto <eos>`.
  Precisa de um parâmetro novo, `train.conditioning`. Com a mesma duração do E1, a comparação é contra
  o E1.
- **Por quê:** é a prova central de que o modelo **usa** a nota, e não só escreve resenhas genéricas.
- **Previsão, com a conta:** sem a nota, as amostras pedidas como nota 1 e como nota 5 saem da mesma
  distribuição. Se o juiz classifica uma fração `p_neg` delas como negativa, `p_neu` como neutra e
  `p_pos` como positiva, a coerência de cada classe é essa fração. A macro fica
  `(p_neg + p_neu + p_pos) / 3 = 1/3`. **Previsão: coerência ≈ 0,33**, contra os 0,617 da run 0. A
  perplexity **piora um pouco**, porque a nota é informação: saber que a resenha é nota 1 torna
  "defeito" mais provável.
- **Custo:** ~1 h.

### Run final (consolidação, não é experimento)

Ela junta as mudanças **adotadas** nas fases 1 e 2, na duração do E1. É o candidato a checkpoint
final, escolhido pela validação. Se o E3 for adotado, o custo passa de 1 h, e conversamos antes.

## Emenda à fase 2 (19/11/2026, depois da fase 1 e antes de qualquer run da fase 2)

Aprovada pelo aluno. **Nenhuma hipótese nem previsão mudou**, só como e quanto tempo se roda:

1. **O E6 roda no orçamento curto (6.000 passos), contra a run 0**, e não longo contra o E1. A
   comparação continua limpa: é uma variável só (a nota), contra a referência cujo ruído o E0 já
   mediu, como manda a regra 1. O custo cai de ~1 h para ~13 min. A previsão continua a mesma:
   coerência ≈ 0,33 e perplexity um pouco pior.
2. **A duração da run final sai da curva do E1.** As quatro mudanças adotadas juntas custam **2,9× por
   passo** (366 ms contra 128 ms, medido). 30.000 passos levariam ~3 h, acima do limite de 1 h por run
   da regra 5. A run final para onde a curva do E1 mostrar que o ganho satura, até 30.000 passos, e o
   early stopping pode cortar antes. O total fica dentro das ~6 h da regra 5.

## Resultados

Preenchidos depois de cada run.

| Run | Mudança | Perplexity por palavra | Coerência macro | Distinct-2 | Cópia exata | Cópia de 8 palavras | Passos | Tempo | Previsão confirmada? | Adotada? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| run 0 | — | 218,5 | 0,617 | 0,68 | 0,3% | 1,5% | 6.000 | 13 min | — | referência |
| E0 | semente 43 | 219,3 | 0,642 | 0,70 | 0,2% | 0,8% | 6.000 | 13 min | sim | (mede o ruído) |
| E2 | LR 1e-3 | **141,7** | 0,673 | 0,68 | 0,0% | 1,4% | 6.000 | 13 min | sim | **sim** |
| E3 | `d_model` 256 | **140,1** | 0,670 | 0,67 | 0,0% | 1,3% | 6.000 | 26 min | sim (custo: 2×, não 3×) | sim, com ressalva de custo |
| E4 | dropout 0 | **193,1** | 0,608 | 0,71 | 0,1% | 1,1% | 6.000 | 10 min | direção sim, tamanho não (−12%, previsto 1 a 5%) | sim, vigiando o sobreajuste |
| E5 | vocabulário 8.000 | **174,5** | 0,627 | 0,69 | 0,0% | 1,0% | 6.000 | 21 min | **não** (−20%, previsto ±5%) | sim, com ressalva de custo |
| E6 | **sem a nota** | 224,6 | **0,312** | 0,68 | 0,0% | 1,5% | 6.000 | 13 min | **sim** (≈ 0,33 e perplexity um pouco pior) | — (ablação) |
| E1 | 30.000 passos | **97,9** | 0,676 | 0,65 | 0,1% | 2,6% | 30.000 | 64 min | **sim** (−55%, previsto ≥ 15%) | sim |
| **final** | E2 + E3 + E4 + E5, 15.000 passos | **69,3** | **0,721** | 0,68 | 0,0% | 3,1% | 15.000 | 66 min | — (consolidação) | **checkpoint final** |

Parâmetros: 1,32 M na run 0 e no E0, E2, E4 e E6; 4,22 M no E3; 1,83 M no E5 (a tabela de embedding
dobra com o vocabulário); 5,24 M na run final.

Os números desta tabela saem do MLflow. Como ele fica fora do Git, todas as runs estão exportadas em
`reports/experiments/runs.csv` (parâmetros e métricas finais) e `reports/experiments/curves.csv`
(as curvas de treino e validação passo a passo), para qualquer pessoa conferir.

### Leitura de cada run

**E0 (ruído entre sementes).** A perplexity mudou só **0,4%** (218,5 → 219,3), e a coerência mudou
**0,025** (0,617 → 0,642). Ambas ficaram dentro da previsão. Consequências para as próximas runs:

- **Perplexity:** pela regra 4, uma mudança precisa baixar a perplexity em mais de **~0,8%** (o dobro
  de 0,4%) para contar como melhora.
- **Coerência:** ela oscila quase 0,03 só por trocar a semente, então diferenças desse tamanho são
  ruído. É por isso que ela é guarda, e não métrica de decisão.
- **Comprimento e cópia:** a média de palavras (16,9 → 19,1) e a cópia de 8 palavras (1,5% → 0,8%)
  também variam com a semente. Diferenças pequenas nelas não serão lidas como efeito.

**E2 (LR 1e-3).**
- **Perplexity: 218,5 → 141,7 (−35%).** É cerca de 40 vezes o limite de ruído, e a previsão se
  confirmou.
- **Coerência:** subiu para 0,673, então a guarda passa. Não houve instabilidade: a loss caiu desde
  o começo, e a norma do gradiente ficou na mesma faixa da run 0.
- **A curva** (`val_loss` a cada 750 passos):
  - run 0: 6,23 → 5,14 → 4,82 → 4,65 → 4,52 → 4,43 → 4,39 → 4,36;
  - E2: 5,65 → 4,73 → 4,45 → 4,31 → 4,17 → 4,09 → 4,04 → 4,00.
- **A leitura:** o E2 chega no passo 2.250 onde a run 0 só chega no fim. Ela confirma o diagnóstico de
  subajuste: com LR 3e-4, cada passo andava pouco demais. A curva do E2 ainda desce no fim, então o
  E1 (mais passos) deve somar ganho.
- **Adotado.** Pela regra 1, as próximas runs da triagem continuam comparadas com a run 0, com LR
  3e-4. Assim, cada uma mede o efeito de uma variável só. A combinação com LR 1e-3 é testada na run
  final.

**E3 (`d_model` 256, 4,2 M de parâmetros).**
- **Perplexity: 218,5 → 140,1 (−36%)**, com o mesmo número de passos. A coerência foi para 0,670, e a
  guarda passa.
- **Custo:** o treino levou **26 min, o dobro** da run 0, e não o triplo previsto. As contas com
  matrizes maiores aproveitam melhor os 16 núcleos da CPU.
- **Sem sobreajuste:** treino e validação terminaram juntos (3,98 contra 3,97), mesmo com 3× mais
  parâmetros.
- **A leitura:** a previsão se confirmou. Um modelo maior aprende mais por passo. O E3 sozinho chega
  quase ao mesmo ponto que o E2 sozinho (140,1 contra 141,7), mas o E2 custa metade do tempo.
- **Adotado, com ressalva.** No orçamento de 1 h, o modelo maior treina metade dos passos. Se a run
  final juntar o E2, o E3 e a duração do E1, ela passa de 1 h, e o custo vai ser discutido antes,
  como manda a regra 5.

**E4 (dropout 0).**
- **Perplexity: 218,5 → 193,1 (−12%).** A direção estava prevista, mas o tamanho não: a previsão era
  de 1 a 5%. A coerência ficou em 0,608 (−0,009, dentro do ruído), e a guarda passa.
- **Custo:** o treino ficou mais rápido, 10 min em vez de 13, porque sem dropout há menos conta.
- **A leitura:** no subajuste, o dropout custava mais do que a previsão dizia. Ele joga fora 10% das
  ativações a cada passo, e com poucos passos cada sinal perdido pesa.
- **O primeiro sinal de distância entre treino e validação:** sem dropout, a loss de treino passa a ser
  medida "limpa", e ela ficou em **4,16, contra 4,24 na validação**. O modelo já vai um pouco melhor
  nos dados que viu. Com 6.000 passos isso não é problema, mas num treino 5× mais longo pode virar
  sobreajuste.
- **Adotado, vigiando o sobreajuste.** Se, no E1 ou na run final, a validação parar de cair enquanto o
  treino continua caindo, o dropout volta. O early stopping guarda o melhor ponto até lá.

**E5 (vocabulário 8.000).**
- **Perplexity: 218,5 → 174,5 (−20%).** **A previsão estava errada:** eu esperava um efeito pequeno,
  de ±5%. A coerência ficou em 0,627, e a guarda passa.
- **O que mudou no texto:** cada palavra virou menos tokens (1,23 → 1,13 por palavra). Com isso, menos
  resenhas passaram do `block_size`: 1.829 saíram do treino, contra 2.404, e na validação 232 precisaram
  de janela, contra 309.
- **A lição de medida:** a loss **por token** ficou **pior** (4,32 → 4,53), enquanto a perplexity **por
  palavra** ficou **melhor**. Não é contradição: com mais tokens no vocabulário, cada token carrega mais
  texto e é mais difícil de acertar. Medido na mesma quantidade de texto (por palavra), o modelo
  prevê melhor. É exatamente o motivo da regra 2. Escolhido pela loss por token, este experimento teria
  sido descartado por engano.
- **Custo:** 21 min, 1,6× a run 0. A camada de saída tem o dobro de linhas, e o softmax é feito sobre
  8.000 tokens a cada posição.
- **Adotado, com ressalva de custo.**

### Fechamento da fase 1

| Mudança | Ganho na perplexity | Custo de tempo | Previsão |
|---|---:|---:|---|
| LR 1e-3 (E2) | −35% | 1× | confirmada |
| `d_model` 256 (E3) | −36% | 2× | confirmada (custo menor que o previsto) |
| dropout 0 (E4) | −12% | 0,8× | direção certa, tamanho subestimado |
| vocabulário 8.000 (E5) | −20% | 1,6× | errada (efeito subestimado) |

As quatro mudanças passaram a regra do ruído com folga (o limite era ~0,8%) e a guarda da coerência.
Todas foram adotadas. O padrão das quatro conta a mesma história: **o modelo da Etapa 2 estava
subajustado**, e tudo o que deixa o aprendizado mais rápido ou o modelo mais capaz ajudou. Nada indicou
decoreba ainda.

**E6 (sem a nota, contra a run 0). A prova de que o modelo usa a nota.**
- **Coerência: 0,617 → 0,312.** A previsão feita pela conta era 1/3 ≈ 0,33. A diferença de 0,02 está
  dentro do ruído de amostragem medido no E0 (0,025): cada classe é medida em amostras sorteadas
  separadamente, então as três frações não somam exatamente 1 (aqui, 0,24 + 0,19 + 0,505 = 0,935).
- **Por classe:** sem a nota, o juiz chamou de positivas cerca de metade das amostras, não importando a
  nota pedida (0,505 nas notas 4 e 5). Isso é o modelo reproduzindo a proporção do próprio treino, em
  que 62% das resenhas são de nota 4 ou 5. As notas 1 e 2 caíram para 0,24, e a nota 3 para 0,19.
- **Perplexity: 218,5 → 224,6 (+2,8%).** É 7 vezes o ruído, e a previsão "piora um pouco" se
  confirmou. Saber a nota ajuda a prever o texto, mas pouco: a maior parte do que torna uma resenha
  previsível (gramática, vocabulário de produto) não depende do tom.
- **A leitura:** o token `<nota_k>` vale **+0,305 de coerência**, o que transforma um gerador "de tom
  aleatório" num gerador que acerta o tom em ~62% das vezes, contra um teto de 0,728 do juiz. É a
  evidência central do H2: a coerência vem da nota, e não de o juiz gostar de qualquer texto. O mesmo
  modelo, sem a nota, fica no acaso.

**E1 (30.000 passos, config da run 0).**
- **Perplexity: 218,5 → 97,9 (−55%).** A previsão (pelo menos 15%) se confirmou com folga, e esta foi
  a maior melhora da etapa. A coerência foi para 0,676.
- **Sem sobreajuste:** treino e validação andaram juntos até o fim, com diferença entre −0,07 e +0,04,
  o que é ruído de lote. O early stopping não disparou, e o melhor passo foi o 29.250.
- **A curva achata:**

  | Trecho | Queda da `val_loss` | Na perplexity |
  |---|---:|---:|
  | 10k → 15k | 0,131 | ~12% |
  | 15k → 20k | 0,062 | ~6% |
  | 20k → 25k | 0,052 | ~5% |
  | 25k → 30k | 0,020 | ~2% |

  Uma ressalva: o LR decai em cosseno até o passo 30.000, então parte do achatamento vem da própria
  agenda, que reduz o tamanho dos passos no fim.
- **O preço do treino longo:** a cópia de 8 palavras subiu de 1,5% para 2,6%, e o distinct-2 caiu de
  0,68 para 0,65. Com mais treino, o modelo reproduz mais trechos comuns. As duas medidas continuam
  dentro dos limiares (H5 ≤ 7%, H6 ≥ 0,59), mas são as que vigiar na run final.

### Decisão da duração da run final (antes de rodá-la)

**15.000 passos**, com o early stopping ligado. Os motivos:
- Na curva do E1 (LR 3e-4), o ganho cai para ~5% a cada 5.000 passos depois do passo 20.000.
- Com LR 1e-3, o E2 chegou aonde a run 0 chegava ~2,7× mais rápido. Então a configuração combinada
  deve alcançar o trecho plano bem antes de 30.000 passos.
- 15.000 passos × 366 ms dão ~1h35, e o total da etapa fica em ~5 h, dentro das 6 h da regra 5.

**A configuração:** LR 1e-3, `d_model` 256, dropout 0 e vocabulário 8.000. As outras opções ficam
iguais às da run 0.

**O risco:** o LR 1e-3 nunca foi testado junto com o modelo maior. O corte do gradiente protege contra
uma explosão. Se a loss subir nas primeiras avaliações, a run é interrompida e refeita com LR menor.

### Resultado da run final

**As cinco hipóteses medidas automaticamente passaram** (o H3, a rubrica humana, está na seção
seguinte):

| Hipótese | Limiar | Run 0 | Run final | Passou? |
|---|---|---:|---:|---|
| H1 perplexity por palavra | ≤ 72 (trigrama: 90,2) | 218,5 | **69,3** | sim |
| H2 coerência macro | ≥ 0,57 (teto do juiz em texto real: 0,728) | 0,617 | **0,721** | sim |
| H4 cópia exata (≥ 5 palavras) | ≤ 1% | 0,3% | **0,0%** | sim |
| H5 cópia de 8 palavras | ≤ 7% (referência humana: 5,5%) | 1,5% | **3,1%** | sim |
| H6 distinct-2 | ≥ 0,59 (texto real: 0,65) | 0,68 | **0,68** | sim |

Esta é a validação, e não o teste. O `test.parquet` continua fechado até a Etapa 4, e o número que vai
para o relatório final é o dele.

- **Perplexity: 218,5 → 69,3 (−68%)**, abaixo do trigrama (90,2) e da meta do H1. Por token, ela é
  42,6, com 1,13 token por palavra.
- **Coerência: 0,721**, quase no teto de 0,728 que o juiz alcança em resenhas reais. Por nota, ela é
  0,91 (nota 1), 0,68 (2), 0,525 (3), 0,785 (4) e 0,90 (5). A nota 3 é a mais difícil, como já
  acontecia com o juiz em texto real: uma resenha "neutra" mistura elogio e crítica.
- **Custo: 66 min de treino**, ou 262 ms por passo, abaixo dos 366 ms estimados na emenda. Somadas,
  as nove runs da etapa levaram ~4 h de treino, dentro das ~6 h da regra 5.
- **Sem instabilidade:** o LR 1e-3 com o modelo maior, o risco anotado antes da run, não causou picos.
  A loss caiu em toda avaliação.

**O sinal de alerta: treino e validação estão se afastando.** A validação ainda melhorava no último
passo, então o early stopping não disparou, e o checkpoint é o do passo 15.000. Mas o ritmo mudou:

| Trecho | Queda da `val_loss` | Distância média (validação − treino) |
|---|---:|---:|
| 2.500 → 5.000 | 0,239 | 0,07 |
| 5.000 → 7.500 | 0,165 | 0,11 |
| 7.500 → 10.000 | 0,118 | 0,16 |
| 10.000 → 12.500 | 0,080 | 0,21 |
| 12.500 → 15.000 | 0,029 | 0,31 |

Este é exatamente o risco anotado no E4: sem dropout e com 4× mais parâmetros, o modelo começa a
decorar o treino. Por enquanto ele ainda generaliza, porque a validação cai e a cópia está abaixo da
referência humana. Mas treinar mais nesta configuração tende a piorar a validação, e não a melhorar.
Parte do achatamento no fim também vem do cosseno, que reduz o LR para 1e-4 no último passo.

- **O que fica para o relatório:** a cópia de 8 palavras subiu junto com o tempo de treino (run 0:
  1,5%, E1: 2,6%, final: 3,1%). Ela continua abaixo dos 5,5% que resenhas humanas reais "copiam" do
  treino antes do dedupe, mas é a medida que vigiar se o modelo crescer.
- **O que não foi testado (por orçamento):** dropout 0,1 com o modelo grande e mais passos. Pela
  tabela acima, é o próximo experimento mais promissor, e ele entra no relatório como trabalho futuro.

### H3: rubrica humana, às cegas (sub-etapa 3.4)

Lote fixo em `reports/rubric/` com a rubrica v1.0 (`docs/rubric.md`):
- 20 amostras por nota de cada fonte: trigrama, Transformer final e, como controle, resenhas reais da
  validação;
- 300 textos embaralhados, com ids anônimos e semente 2026;
- o lote foi commitado antes da pontuação, e a chave (`key_v1.csv`) só foi aberta depois.

O resumo está em `reports/metrics/rubric.json`.

| Fonte | Amostras ruins | Fluência | Plausibilidade | Coerência | Repetição |
|---|---:|---:|---:|---:|---:|
| Trigrama | **85%** | 1,40 | 1,44 | 2,19 | 2,97 |
| **Transformer** | **37%** | 2,19 | 2,26 | 2,60 | 2,93 |
| Resenhas reais (controle) | 3% | 2,95 | 2,91 | 2,86 | 2,98 |

Amostra ruim é a que tem algum critério com nota 1. As colunas de critério são médias de 1 a 3.

**H3 passou:** 37% de amostras ruins contra 85% do trigrama. A diferença, de 48 pontos, tem intervalo
de 95% de ±12 pontos com 100 amostras por fonte, então não é sorte do sorteio.

- **O controle funciona.** As resenhas reais ficaram com 3% de ruins, o que mostra que a rubrica não
  pune o jeito informal de escrever do B2W. Os 37% do Transformer vêm, na maior parte, de defeito
  do modelo, e não da régua (quanto a régua pesa está na ressalva abaixo).
- **O que reprova o Transformer é a fluência, e não a nota.**
  - Das 37 amostras ruins, 33 têm fluência 1 e 18 têm plausibilidade 1, e muitas têm as duas.
  - Só 3 têm coerência 1, e 1 tem repetição 1.
  - Isso reforça o H2 por outro caminho: quando o texto se entende (fluência 2 ou 3), o tom quase
    nunca contradiz a nota (3 de 67 amostras com coerência 1). O problema é a forma. Das 33 amostras
    com fluência 1, 12 começam bem e se perdem, 12 já começam quebradas, 8 misturam assuntos e 1
    entra em loop. É o limite de um modelo pequeno (5,2
    M de parâmetros). A falta de contexto foi descartada como causa (ver `docs/error_analysis.md`).
- **Por nota:** as amostras ruins do Transformer são 35% (nota 1), 65% (2), 45% (3), 15% (4) e 25%
  (5). As notas 4 e 5 são as mais fáceis, provavelmente porque são as mais frequentes no treino (62%
  das resenhas).
  A nota 2, com só 6% do treino, é a pior. É o risco de desbalanceamento do charter, que aparece aqui
  e não na coerência macro.
- **Ressalva:** o avaliador é um só, o autor, e aplicou as três fronteiras esclarecidas pela v1.1 da
  rubrica (fluência, coerência na nota 3 e plausibilidade de textos curtos) com mais rigor do que ela
  pede. Isso infla os números absolutos. A parte da coerência na nota 3 dá para contar: pela v1.1,
  deixariam de ser ruins 2 resenhas reais (3% → 1%), 1 amostra do Transformer (37% → 36%) e nenhuma
  do trigrama. As partes da fluência e da plausibilidade não foram medidas, porque exigiriam
  repontuar. Elas provavelmente pesam mais no Transformer, cujos erros ficam mais perto da fronteira
  entre fluência 1 e 2 do que a salada do trigrama. Ou seja, o rigor joga contra o Transformer, e o
  H3 passou mesmo assim (ver as limitações em `docs/rubric.md`).
- O `reports/metrics/transformer_eval.json` continua com o H3 "pendente", porque é saída do estágio
  `evaluate` do DVC, e a rubrica é trabalho humano, fora do pipeline. O resultado do H3 está em
  `reports/metrics/rubric.json`.

As amostras ruins do Transformer são a matéria-prima da análise de erros (sub-etapa 3.5), em
`docs/error_analysis.md`.

## Fechamento da Etapa 3

**O que foi entregue:**
- Nove runs comparáveis (run 0, E0 a E6 e a run final), cada uma com a hipótese escrita antes. Estão
  no MLflow e exportadas para `reports/experiments/`.
- O checkpoint final (`final-combinada`, promovido com `dvc exp apply`):
  - LR 1e-3, `d_model` 256, sem dropout e vocabulário de 8.000 tokens;
  - 15.000 passos e 5,2 M de parâmetros.
- **H1 a H6 passaram na validação** (tabela na seção 7.1 do charter). O H3 vem da rubrica humana às
  cegas, com resenhas reais como controle.
- A análise de erros das amostras ruins (`docs/error_analysis.md`), com as causas medidas, e não só
  descritas.

**O que vai para a Etapa 4, decidido com a evidência desta etapa:**
1. **Inferência com top-k e temperatura menor**, contra a causa A da análise de erros. Os valores são
   escolhidos **na validação**, com as mesmas métricas (coerência, cópia de 8 palavras e distinct-2,
   que podem piorar), e só depois o teste é aberto.
2. **Abrir o teste uma única vez**, com o checkpoint final e os parâmetros de inferência já fixados.
   Uma nova pontuação humana, se houver, usa a rubrica v1.1.
3. Na API, um **limite de tamanho** da geração e uma **checagem de repetição** (causas B e E), como
   mitigação.
4. **Link público do checkpoint:** hoje o remote do DVC é uma pasta local.
5. API com `/health`, testes da API e Docker.

**Trabalho futuro, fora do orçamento da etapa:**
- dropout 0,1 com o modelo grande e mais passos, por causa do afastamento entre treino e validação
  na run final;
- condicionamento também pela categoria (causa C);
- peso por classe ou sobreamostragem da nota 2 (causa D).

**Nota de reprodutibilidade:** depois da run final, o `dataset.py` foi refatorado. O filtro de
comprimento virou a função `fits_in_context`, com a mesma conta e um teste de fronteira, sem mudar o
comportamento. A prova: desfazendo só essa refatoração, o arquivo volta exatamente ao hash
registrado na run final (md5 `38fde6d3…`, 7.447 bytes). O novo hash foi registrado com
`dvc commit -f train`, sem retreinar. A reprodução limpa da Etapa 5 vai confirmar a equivalência na
prática.
