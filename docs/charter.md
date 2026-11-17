# Charter do projeto nano-review

> Versão 0.1 (rascunho, Etapa 1). Este documento fixa o problema e a forma de medir o sucesso
> **antes** de existir modelo, para que a avaliação não seja ajustada depois para caber no resultado.

## 1. Problema e nicho

Quem desenvolve telas de avaliação, filtros de moderação ou classificadores de sentimento precisa de
muitos textos de resenha para testar o sistema. Dados reais têm problemas para esse uso: dados pessoais,
licença, pouca variedade em algumas notas (resenhas de nota 3 são raras) e o fato de que não dá para
pedir "mais 500 resenhas de nota 2".

O nano-review é um gerador de resenhas curtas de produto em português, condicionado à nota, treinado
**do zero** (Transformer só-decoder pequeno, sem modelo pré-treinado) em CPU.

## 2. Usuário e decisão

- **Usuário:** desenvolvedor ou analista de QA que precisa de textos sintéticos.
- **Decisão que o sistema apoia:** "este conjunto de textos serve para testar a minha tela, o meu filtro
  ou o meu classificador?". Para isso, o texto precisa soar como resenha, ser coerente com a nota pedida
  e variar entre amostras.

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

Duas cautelas registradas desde já:

1. **O juiz erra.** A coerência depende de um classificador treinado nos dados reais. A acurácia dele na
   validação real é o teto: um gerador não pode ser julgado mais coerente do que o juiz consegue medir.
   Por isso a acurácia do classificador é sempre reportada junto.
2. **Perplexity só se compara na mesma unidade.** O baseline n-grama da Etapa 1 trabalha com palavras e o
   Transformer vai trabalhar com tokens BPE. Perplexity por palavra e por token BPE não são comparáveis.
   Na Etapa 2 a comparação passa a usar a mesma tokenização (ou bits por caractere).

## 6. Baseline

- **Gerador:** modelo de n-gramas (trigrama de palavras com suavização) treinado no mesmo formato
  `<nota_k> texto <eos>`. É o "modelo de linguagem mais simples que funciona": se o Transformer não o
  superar, o Transformer não se justifica.
- **Juiz:** TF-IDF + regressão logística treinado no conjunto de treino real.

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
| Desbalanceamento (nota 5 domina) | O modelo aprende a gerar elogio para qualquer nota | Coerência macro por classe; avaliar amostragem balanceada ou peso por classe na Etapa 3 |
| Memorização | O "gerador" vira um copiador de resenhas reais | Métrica de cópia; dedupe antes do split; split por produto |
| Dados pessoais no texto | Vazar e-mail, telefone ou CPF de clientes reais | Descartar colunas de perfil do avaliador; filtrar o texto por regex na limpeza |
| CPU limita o tamanho do modelo | Texto pouco fluente | Resenhas curtas, vocabulário BPE pequeno, modelo pequeno; treinar cedo, não só no fim |
| Juiz fraco | Coerência medida com ruído | Reportar a acurácia do juiz; rubrica humana como segunda opinião |
| Uso indevido | Publicar avaliações falsas em loja real | Ver seção 9 |

## 9. Ética e fora do escopo

- **Não é objetivo:** gerar avaliações para publicação. Resenha falsa engana consumidores e, no Brasil,
  pode configurar publicidade enganosa (CDC). O uso pretendido é teste de software.
- Fora do escopo da v1: condicionamento por categoria ou produto, resenhas longas, uso comercial
  (a licença CC BY-NC-SA não permite), qualquer modelo pré-treinado como núcleo.

## 10. Decisões pendentes

- [ ] Limiares numéricos da hipótese (depois do baseline, sub-etapa 6).
- [ ] Rubrica humana (antes da Etapa 3).
- [ ] Corte de comprimento das resenhas (depois da EDA, sub-etapa 4).
