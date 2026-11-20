# Rubrica humana

> Versão 1.0, escrita em 17/11/2026, **antes de qualquer amostra gerada existir** (nem do baseline,
> nem do Transformer). É por isso que ela vale como medida: se fosse escrita depois de ler as
> amostras, poderia ser ajustada, sem querer, para favorecer um dos modelos. A versão 1.1
> (20/11/2026), mais abaixo, esclarece três fronteiras e vale só para as avaliações seguintes.

## Para que serve

As métricas automáticas do charter (perplexity, coerência pelo juiz, distinct-n, cópia) não pegam
tudo. Um texto pode ter boa perplexity e não fazer sentido, ou enganar o juiz com uma palavra
positiva no meio de uma reclamação. A rubrica mede o que só uma pessoa lendo percebe.

## Como aplicar

1. **Amostra fixa.** Cada modelo avaliado gera **20 resenhas por nota** (100 no total), com semente
   fixa e os mesmos parâmetros de amostragem. A semente e os parâmetros ficam registrados junto com
   as notas da rubrica.
2. **Às cegas.** As amostras de todos os modelos comparados são misturadas numa planilha só, em
   ordem aleatória, sem a coluna de qual modelo gerou. A nota pedida aparece, porque a coerência
   depende dela.
3. **Uma amostra por vez, os quatro critérios.** Use as descrições e os exemplos abaixo, e não a
   comparação com a amostra anterior.
4. **Só depois de pontuar tudo**, a coluna do modelo é revelada e as médias são calculadas.

Formato da planilha (CSV): `sample_id, requested_rating, text, fluency, plausibility, coherence,
repetition, notes`. A coluna `model` fica num arquivo separado, ligado pelo `sample_id`.

## Critérios (de 1 a 3)

### 1. Fluência: dá para ler como português?

| Nota | Descrição | Exemplo inventado |
|---|---|---|
| 1 | Frases quebradas ou palavras soltas sem sentido. | "produto o de chegou bom entrega a muito" |
| 2 | Dá para entender, mas há trechos sem sentido ou erros que atrapalham a leitura. | "chegou rápido mas o produto que funciona não da carga bem" |
| 3 | Lê-se sem esforço. | "Chegou rápido e funciona bem, recomendo." |

Erro de digitação e falta de acento típicos de resenha real ("nao", "otimo", "vc") **não** tiram
ponto: o objetivo é imitar resenha de verdade, e não texto revisado.

### 2. Plausibilidade: poderia estar no site da loja?

| Nota | Descrição | Exemplo inventado |
|---|---|---|
| 1 | Não parece resenha: lista de palavras, texto de outro gênero, frase sem relação com compra. | "o governo anunciou ontem a nova medida" |
| 2 | Parece resenha, mas tem contradição interna ou mistura produtos incompatíveis. | "a tela do sofá é ótima e a bateria do liquidificador dura muito" |
| 3 | Uma pessoa acreditaria que é uma resenha real. | "Comprei para minha mãe, ela adorou. Só a caixa veio amassada." |

### 3. Coerência com a nota pedida

| Nota | Descrição | Exemplo inventado |
|---|---|---|
| 1 | O tom contradiz a nota. | Nota 1: "Excelente, superou minhas expectativas!" |
| 2 | Tom ambíguo ou neutro quando a nota é extrema, ou elogio entusiasmado sem ressalva na nota 3. | Nota 5: "Produto ok, chegou no prazo." |
| 3 | O tom combina com a nota. | Nota 2: "Funciona, mas esquenta muito e o acabamento é fraco." |

Referência para cada nota pedida:

- **Notas 1 e 2:** insatisfação. Na nota 2 cabe algum ponto positivo, mas o saldo é negativo.
- **Nota 3:** satisfação parcial: "razoável", "esperava mais", elogio com ressalva clara. A EDA
  mostrou que 88% das resenhas reais de nota 3 recomendam o produto, então um tom levemente positivo
  **com ressalva** vale 3. Elogio sem nenhuma ressalva vale 2.
- **Notas 4 e 5:** satisfação. Na nota 4 cabe uma ressalva pequena.

### 4. Repetição: o texto anda em círculo?

| Nota | Descrição | Exemplo inventado |
|---|---|---|
| 1 | A mesma palavra ou frase se repete 3 vezes ou mais, em loop. | "muito bom muito bom muito bom muito bom" |
| 2 | Alguma repetição perceptível, mas o texto avança. | "Chegou rápido, muito rápido, e é muito bom, muito bom mesmo." |
| 3 | Nenhuma repetição além do normal de uma resenha. | "Bom produto, entrega rápida." |

Repetição usada como ênfase, do jeito que resenhas reais fazem ("muito muito bom"), vale 3.

## Observações (coluna `notes`)

Além das notas, registre, quando aparecer:

- **cópia aparente**: o texto parece uma resenha real decorada (a métrica de cópia confere depois);
- **dado pessoal ou nome de loja/pessoa**;
- **conteúdo ofensivo**;
- **caso não previsto** pela rubrica (ver regras abaixo).

## Como resumir

- Média de cada critério, por modelo e por nota pedida.
- **Amostra ruim:** qualquer critério com nota 1. A fração de amostras ruins é o número principal
  da rubrica, e as amostras ruins alimentam a análise de erros do charter (pelo menos 10, uma a uma).

## Regras para manter a rubrica honesta

1. **Não mudar a rubrica depois de ler amostras.** Um caso que ela não prevê recebe a descrição mais
   próxima e é anotado como "caso não previsto".
2. Se uma mudança for mesmo necessária, ela vira uma **versão nova, com data**, e vale só para as
   avaliações seguintes. Avaliações antigas não são repontuadas com a versão nova.
3. Os modelos comparados entre si são sempre avaliados **com a mesma versão** da rubrica, na mesma
   planilha cega.

## Versão 1.1 (20/11/2026)

Escrita **depois** de pontuar o lote v1 e de abrir a chave dele. Por isso ela **não** vale para o
lote v1: as notas dele ficam como foram dadas (regra 2 acima). Ela vale para as avaliações seguintes,
por exemplo a do teste, na Etapa 4.

Os critérios e as notas de 1 a 3 não mudam. A v1.1 só esclarece três fronteiras que, na v1, davam
margem a mais de uma leitura. Diferente da v1.0, os exemplos abaixo não são inventados: foram
adaptados de casos do lote v1.

1. **Fluência, 1 ou 2.** O teste é: "consigo dizer, numa frase, o que a pessoa quis dizer?". Se sim, a
   nota é 2, mesmo com trechos sem sentido. Se não, é 1.
2. **Coerência na nota 3.** Na nota 3, um texto só positivo **ou** só negativo vale 2: ele não
   contradiz a nota, só fica na nota vizinha. A coerência 1 é para as notas 1, 2, 4 e 5, quando o tom
   é o oposto do pedido. Elogio com ressalva ("é bom, mas...") continua valendo 3.
3. **Plausibilidade de textos curtos.** Uma reclamação ou um elogio curto, do tipo que um cliente
   escreveria, vale 3, mesmo com erro de digitação ("Não mostra o tamanho do produto."). Um desabafo
   que não avalia nada vale 2 ("Nem chegou."). Um fragmento sem relação com compra vale 1 ("No dia 5.").

## Limitações

- **Um avaliador só** (o autor do projeto), que conhece os modelos. A avaliação às cegas e os
  exemplos fixos reduzem esse viés, mas não o eliminam. Se houver tempo na Etapa 5, uma segunda
  pessoa pontua 20 amostras, e a concordância entre as duas é reportada.
- No lote v1, as três fronteiras esclarecidas na versão 1.1 foram aplicadas com mais rigor do que a
  v1.1 pede:
  - mais textos receberam fluência 1;
  - reclamações na nota 3 receberam coerência 1;
  - textos curtos e verossímeis receberam plausibilidade 1.

  Isso aumenta a fração de amostras ruins. A parte da coerência na nota 3 dá para contar: pela v1.1,
  deixariam de ser ruins 2 resenhas reais (3% → 1%), 1 amostra do Transformer (37% → 36%) e nenhuma
  do trigrama. As outras duas partes exigiriam repontuar e não foram medidas. Elas provavelmente
  pesam mais no Transformer, cujos erros ficam mais perto da fronteira entre fluência 1 e 2 do que a
  salada do trigrama. Ou seja, o rigor joga contra o Transformer: a comparação do H3 fica
  conservadora, e os números absolutos estão inflados.
- 20 amostras por nota são poucas para diferenças pequenas. A rubrica serve para achar problemas
  que os números não mostram, e não para decidir sozinha qual modelo é melhor.
