# Charter do projeto nano-review

> Versão 0.3 (Etapa 1: distribuição das notas corrigida com os dados reais do `docs/data_manifest.md`). Este documento fixa o problema e a forma de medir o sucesso
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

B2W-Reviews01 (Kaggle), resenhas reais da Americanas/Submarino, licença CC BY-NC-SA 4.0 (exige
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
| Cópia do treino | O modelo está decorando? | Fração de amostras idênticas a uma resenha de treino e fração com algum trecho de 8 palavras presente no treino. |
| Rubrica humana | O que número nenhum pega: fluência, plausibilidade | Rubrica escrita **antes** de lermos amostras (até o início da Etapa 3), aplicada a uma amostra fixa. |
| Análise de erros | Por que o modelo erra, não só quanto | Pelo menos 10 amostras ruins analisadas uma a uma: nota pedida, texto gerado, o que está errado, hipótese da causa e ação possível. |

Duas cautelas registradas desde já:

1. **O juiz erra.** A coerência depende de um classificador treinado nos dados reais. A acurácia dele na
   validação real é o teto: um gerador não pode ser julgado mais coerente do que o juiz consegue medir.
   Por isso a acurácia do classificador é sempre reportada junto.
2. **Perplexity só se compara na mesma unidade.** O baseline n-grama da Etapa 1 trabalha com palavras e o
   Transformer vai trabalhar com tokens BPE. Perplexity por palavra e por token BPE não são comparáveis.
   Na Etapa 2 a comparação passa a usar a mesma tokenização (ou bits por caractere).

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
  dos pesos que treina o Transformer. O relatório explica esse treino passo a passo; a forma de
  implementar (à mão em PyTorch ou com scikit-learn) é decidida na sub-etapa 6.

## 7. Hipótese

> Em comparação com o baseline de n-gramas, o Transformer treinado do zero tem **perplexity menor** na
> validação e **coerência com a nota maior**, sem aumentar a taxa de cópia do treino nem reduzir a
> diversidade.

A hipótese é relativa ao baseline de propósito. Os limiares numéricos (por exemplo "pelo menos X pontos
a mais de coerência") são definidos depois que o baseline for medido e **antes** de treinar o
Transformer.

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

- [ ] Limiares numéricos da hipótese (depois do baseline, sub-etapa 6).
- [ ] Rubrica humana (antes da Etapa 3).
- [ ] Corte de comprimento das resenhas (depois da EDA, sub-etapa 4).
