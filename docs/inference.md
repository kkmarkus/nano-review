# Inferência (Etapa 4)

> Como o checkpoint final da Etapa 3 vira o texto que o produto entrega. Cada decisão é tomada **sem
> tocar no teste**, e a regra de cada uma é escrita **antes** de olhar o resultado.

## 4.1 Decodificação: temperatura e top-k

### Por que mudar a forma de sortear

- A Etapa 3 mediu o modelo sorteando direto da distribuição que ele aprendeu (temperatura 1, sem
  top-k). Essa é a medida honesta do **modelo**, e por isso continua sendo a régua do H1 a H6.
- O **produto** pode trocar um pouco de variedade por texto mais seguro. A análise de erros
  (`docs/error_analysis.md`, causa A) associou as amostras ruins às palavras inventadas, que vêm de
  tokens BPE improváveis. O mecanismo ainda não foi testado, e esta varredura é o teste: se cortar a
  cauda da distribuição reduz as palavras inventadas, a hipótese ganha força.
- **O custo esperado:** texto mais seguro tende a ser menos variado (distinct-2 cai) e a repetir
  frases comuns do treino (a cópia sobe). Por isso o H5 e o H6 entram como restrição.

Os dois botões, implementados em `nano_review.inference.sampling`:
- **temperatura:** divide os logits antes do softmax. Abaixo de 1, o provável fica mais provável, e
  o raro quase some;
- **top-k:** só os `k` tokens mais prováveis podem ser sorteados; os outros têm chance zero.

### O que é variado

- Temperatura: 1,0; 0,9; 0,8 e 0,7.
- Top-k: sem top-k, 50 e 20.
- São 12 combinações, cada uma com 200 amostras por nota (1.000 no total), semente 42 e o mesmo
  checkpoint. Os valores ficam em `params.yaml` (seção `decoding`), e o estágio `decoding` do DVC
  roda a varredura.

**Conferência de sanidade:** a combinação temperatura 1 sem top-k, com a mesma semente, precisa
reproduzir as amostras da avaliação da Etapa 3 e, portanto, as mesmas métricas
(`reports/metrics/transformer_eval.json`). Um teste já garante que a amostragem nova sorteia os
mesmos tokens que a antiga nesse caso.

### Regra de escolha (fixada em 20/11/2026, antes de rodar a varredura)

1. **Elegíveis:** as combinações que passam H2 (coerência ≥ 0,57), H4 (cópia exata ≤ 1%), H5 (cópia
   de 8 palavras ≤ 7%) e H6 (distinct-2 ≥ 0,59). Os limiares são os do charter e não mudam.
2. Entre as elegíveis, vence a de **menos palavras inventadas por 100 palavras**.
3. **Empate:** vence a de maior distinct-2.
4. Se nenhuma combinação passar nas quatro, a decisão volta para o autor, com os números, e fica
   registrada aqui.

### O que esta escolha não mede

- **A fluência como uma pessoa a vê.** As palavras inventadas são um indicador parcial: pegam o
  defeito típico do BPE, mas não a salada de palavras reais nem a palavra real no lugar errado. Por
  isso a decodificação escolhida passa por um lote humano novo, com a rubrica v1.1, depois que o
  teste for aberto (4.3).
- **A perplexity (H1).** Ela não depende da decodificação: é a probabilidade que o modelo dá a
  textos reais, e não ao que ele sorteia.
- **Nada aqui lê a validação nem o teste.** As amostras são geradas pelo modelo; a cópia e o
  vocabulário vêm do treino; o juiz foi treinado no treino.

### Resultado (20/11/2026)

O estágio `decoding` levou 31 minutos. As métricas de cada combinação estão em
`reports/metrics/decoding.json`, as 12.000 amostras em `reports/samples/decoding.jsonl` e um run por
combinação no MLflow (`decoding-t1.0-k0` etc., exportados para `reports/experiments/`).

**A conferência de sanidade passou.** A combinação temperatura 1 sem top-k reproduziu exatamente
as métricas da Etapa 3: coerência 0,7208, distinct-2 0,6812, cópia de 8 palavras 3,1% e 19,4
palavras em média.

| Temperatura | Top-k | Coerência (H2) | Distinct-2 (H6) | Cópia exata (H4) | Cópia de 8 palavras (H5) | Inventadas por 100 palavras | Reprova |
|---:|---:|---:|---:|---:|---:|---:|---|
| **1,0** | **—** | 0,721 | **0,681** | 0,0% | 3,1% | 3,86 | — |
| 1,0 | 50 | 0,763 | 0,511 | 0,7% | 4,9% | 0,62 | H6 |
| 1,0 | 20 | 0,771 | 0,414 | 0,2% | 9,5% | 0,20 | H5, H6 |
| 0,9 | — | 0,740 | 0,573 | 0,3% | 5,5% | 1,94 | H6 |
| 0,9 | 50 | 0,803 | 0,456 | 1,1% | 6,7% | 0,30 | H4, H6 |
| 0,9 | 20 | 0,803 | 0,377 | 0,7% | 10,3% | 0,14 | H5, H6 |
| 0,8 | — | 0,783 | 0,489 | 0,8% | 9,2% | 0,94 | H5, H6 |
| 0,8 | 50 | 0,791 | 0,402 | 1,8% | 10,0% | 0,20 | H4, H5, H6 |
| 0,8 | 20 | 0,799 | 0,348 | 1,3% | 12,0% | 0,09 | H4, H5, H6 |
| 0,7 | — | 0,800 | 0,395 | 1,7% | 14,3% | 0,15 | H4, H5, H6 |
| 0,7 | 50 | 0,803 | 0,346 | 1,4% | 16,3% | 0,09 | H4, H5, H6 |
| 0,7 | 20 | 0,801 | 0,307 | 1,8% | 18,9% | 0,02 | H4, H5, H6 |

**Pela regra, a escolhida é temperatura 1, sem top-k**, a única que passa nas quatro hipóteses. O
produto entrega, por padrão, o mesmo sorteio que foi medido na Etapa 3.

### Leitura

1. **A causa A da análise de erros agora tem uma intervenção, e não só uma associação.** Cortar a
   cauda da distribuição derruba as palavras inventadas: com top-k 50, elas caem de 3,86 para 0,62
   por 100 palavras, seis vezes menos. A coerência pelo juiz também sobe: com top-k 50, de 0,72 para 0,76–0,80. Se
   isso melhora a fluência que **uma pessoa** percebe, é o que o lote humano da 4.3 vai medir.
2. **O preço é alto e chega cedo.** O primeiro passo para longe da temperatura 1 sem top-k já
   derruba o distinct-2 abaixo de 0,59: 0,573 com temperatura 0,9 e 0,511 com top-k 50. Baixando
   mais, a cópia do treino dispara, até 18,9% de trechos de 8 palavras. O modelo fica "seguro"
   porque passa a repetir as frases mais comuns do treino, e os textos também encurtam (de 19,4
   para 12,5 palavras em média).
3. **Para o usuário do projeto, essa troca não compensa como padrão.** Quem testa um classificador
   ou uma tela de moderação precisa de variedade. Texto mais limpo, mas repetitivo e copiado,
   testaria menos casos. É para isso que o H5 e o H6 entram como restrição.

**Limite da varredura:** a grade é grossa, e a mudança brusca fica entre a temperatura 1 sem top-k e
o primeiro vizinho. Pode existir uma combinação intermediária (temperatura 0,95, top-k 200) que passe
e invente menos palavras. Ela não foi testada para não mudar o experimento depois de ver o
resultado, e fica como trabalho futuro.

### Decisões que saem daqui

- **Padrão do produto:** temperatura 1, sem top-k.
- **A API aceita temperatura e top-k opcionais em cada pedido**, com esse padrão. Quem quiser texto
  mais limpo e aceitar menos variedade e mais cópia pode pedir, por exemplo, top-k 50. A tabela
  acima é a documentação dessa troca.
- **Lote humano v2 (rubrica v1.1, na 4.3):** 150 textos às cegas:
  - 10 por nota da temperatura 1 sem top-k, o padrão;
  - 10 por nota da temperatura 1 com top-k 50, a alternativa mais próxima, que só reprova no H6;
  - 10 por nota de resenhas reais do **teste**, como controle.

  O lote mede se cortar a cauda melhora a fluência percebida por uma pessoa.

  **Regra de exclusão (escrita em 20/11/2026, antes de qualquer sorteio do lote v2):** fica de fora
  do sorteio todo texto idêntico, depois de normalizar caixa, pontuação e espaços, a um dos 300
  textos do lote v1. O motivo: o `inference_check` usa a mesma semente da avaliação da Etapa 3, e a
  revisão independente mostrou que 86 dos 100 textos do Transformer já pontuados no v1 estão no
  conjunto padrão. Um texto já lido, com a chave já aberta, não é avaliação às cegas. A regra cobre
  os 300 textos, e não só os do Transformer, porque frases curtas e genéricas podem coincidir por
  acaso. Mesmo assim, sobram mais de 180 candidatos por nota.

## 4.2 O gerador do produto: validação e filtros

`nano_review.inference.generator.ReviewGenerator` é o que a API chama. Ele carrega o checkpoint e o
tokenizador uma vez e, a cada pedido, valida a entrada, sorteia o texto e o filtra.

### Validação do pedido

| Campo | Aceito | Padrão |
|---|---|---|
| Nota | inteiro de 1 a 5 | obrigatória |
| Quantidade | de 1 a 20 resenhas por pedido | 1 |
| Temperatura | maior que 0 e até 2 | 1,0 (4.1) |
| Top-k | de 1 ao tamanho do vocabulário (8.000); 0 pede "sem top-k" | sem top-k (4.1) |
| Semente | inteiro ≥ 0, opcional | aleatória |

Com semente, o mesmo pedido devolve os mesmos textos, no mesmo ambiente (a mesma versão do torch e
o mesmo número de threads; o Docker da 4.5 pode não bater com esta máquina). Isso serve a quem
precisa de dados de teste
reproduzíveis, como um teste automatizado que espera sempre a mesma resenha.

### Filtros (decididos em 20/11/2026, com os números do treino e das amostras da 4.1)

Um texto reprovado é descartado e sorteado de novo, até 5 vezes. Se as 5 tentativas forem
reprovadas, a API responde com erro (503) e o motivo: o produto nunca entrega um texto que falhou
num filtro.

| Filtro | Regra | Por quê |
|---|---|---|
| Tamanho (`too_long`) | mais de 83 tokens de texto sem chegar ao `<eos>` | 83 é o percentil 95 das resenhas do treino. Na 4.1, 2,9% das amostras padrão passavam disso, e a análise de erros associou texto longo a amostra ruim (causa B). É mitigação, e não cura. O limite também corta, de passagem, os textos que enchiam o contexto sem `<eos>` (1 em 1.000, contando os IDs gerados). |
| Vazio (`empty`) | nenhuma palavra | Não serve como resenha. |
| Repetição (`repetition`) | a mesma palavra 3 vezes seguidas | É a forma mais simples da repetição 1 da rubrica. Pega o loop do "bla bla bla" (causa E), e a ênfase normal ("muito muito bom") passa. Só 0,22% das resenhas reais do treino cairiam nesse filtro. O loop de uma frase inteira ("muito bom muito bom muito bom") **não** é checado: 0,16% das resenhas do treino têm um, e nenhuma das 2.000 resenhas entregues abaixo. |
| Dado pessoal (`personal_data`) | e-mail, telefone, CPF ou URL | Os mesmos padrões da limpeza. O treino não tem nenhum (a limpeza tirou), e as amostras da 4.1 também não, mas o modelo pode compor um número que parece telefone, e o custo de entregar um dado pessoal é alto. O texto é regenerado, e não mascarado: um "[EMAIL]" no meio da resenha não serve como dado de teste. |

O estágio `inference_check` do DVC mede o pipeline completo (decodificação mais filtros), com a
mesma decodificação e os mesmos filtros da API: 1.000 resenhas com o padrão e 1.000 com top-k 50, a
alternativa da 4.1. A validação do pedido fica de fora, porque o estágio chama direto a função que
gera uma resenha.

**A URL é procurada no texto em minúsculas.** O padrão da limpeza não pega "WWW." nem "HTTPS://".
O treino não tem nenhum desses casos, mas o filtro do produto não pode depender disso.

### Resultado do pipeline completo (20/11/2026)

O estágio `inference_check` levou 6 minutos. Os números estão em `reports/metrics/inference.json`,
as 2.000 resenhas entregues em `reports/samples/inference.jsonl` e um run por combinação no MLflow
(`inference-t1.0-k0` e `inference-t1.0-k50`).

| | Padrão (T 1,0, sem top-k) | Alternativa (T 1,0, top-k 50) |
|---|---:|---:|
| Coerência (H2 ≥ 0,57): texto cru → filtrado | 0,721 → **0,728** | 0,763 → 0,766 |
| Distinct-2 (H6 ≥ 0,59) | 0,681 → **0,684** | 0,511 → 0,508 |
| Cópia exata (H4 ≤ 1%) | 0,0% → **0,0%** | 0,7% → 0,7% |
| Cópia de 8 palavras (H5 ≤ 7%) | 3,1% → **2,9%** | 4,9% → 5,0% |
| Inventadas por 100 palavras | 3,86 → 3,81 | 0,62 → 0,60 |
| Palavras por resenha (média) | 19,4 → 17,6 | 16,4 → 16,2 |
| Resenhas que precisaram de mais de uma tentativa | 3,0% | 0,6% |
| Tentativas descartadas (tamanho / repetição / vazio / dado pessoal) | 29 / 1 / 0 / 0 | 6 / 0 / 0 / 0 |
| Resenhas que esgotaram as 5 tentativas (erro 503) | **0** | 0 |
| Tempo por resenha nesta máquina (média / mediana / p95) | 186 / 125 / 554 ms | 152 / 111 / 395 ms |

**Leitura:**
- **Os filtros não quebram o que a 4.1 aprovou.** O padrão continua passando em H2, H4, H5 e H6, e a
  alternativa com top-k 50 continua reprovando só no H6.
- **Quase todo o trabalho dos filtros é o limite de tamanho:** 29 das 30 tentativas descartadas no
  padrão. O filtro de repetição pegou um caso, e o de dado pessoal não pegou nenhum. Ele continua
  lá pelo custo de um erro, e não pela frequência.
- **O custo é pequeno:** no padrão, só 3% das resenhas precisaram de uma segunda tentativa, nenhuma
  de uma terceira, e nenhuma esgotou as 5. O erro 503 deve ser raríssimo.
- **O texto encurta um pouco** (19,4 → 17,6 palavras em média), porque o limite corta a cauda longa.
  As resenhas reais do treino têm mediana de 21 tokens, e as geradas, de 18.
- **O tempo** foi medido com 16 núcleos, uma resenha por vez, e varia um pouco entre rodadas (na
  primeira, a mediana foi de 127 ms). Um pedido de 20 resenhas leva perto de 20 vezes a **média**, e
  não a mediana, porque a soma é puxada pelos textos longos: uns 3,7 segundos. Dentro do Docker, com
  menos núcleos, deve ser mais lento, e será medido na 4.5.
- **Reprodutibilidade:** depois das correções da revisão independente (validação mais rigorosa,
  tempo médio, URL em minúsculas), o estágio foi rodado de novo, e as 2.000 resenhas saíram
  idênticas às da primeira rodada. Os 2 runs antigos do MLflow foram marcados como apagados, para
  não ficarem duplicados.

**Conferência de reprodutibilidade:** depois da varredura da 4.1, dois arquivos do estágio
`decoding` mudaram: o `sampling.py` ganhou o limite de tokens novos, e o `decoding.py` ganhou um
parâmetro de prefixo no nome dos runs do MLflow (para o `inference_check` reaproveitar a função). O
DVC marcou o estágio como alterado. Em vez de rodar os 31 minutos de
novo (o que duplicaria os 12 runs no MLflow), a combinação T 1,0 com top-k 50 foi gerada de novo com
o código atual: as 1.000 amostras saíram idênticas, texto a texto, às gravadas. A combinação padrão
já é coberta por um teste (a amostragem nova sorteia os mesmos tokens da régua da Etapa 3). O hash
novo foi registrado com `dvc commit -f decoding`.

## 4.4 A API

`nano_review.api.app` serve o `ReviewGenerator` por HTTP, com FastAPI. As rotas e os campos estão
no README. As decisões de desenho:

- **O modelo carrega uma vez, quando a API sobe** (o `lifespan` do FastAPI), e não a cada pedido.
  Carregar 5 milhões de pesos custa mais que gerar uma resenha. O uvicorn só aceita conexões depois
  que o modelo carregou, e uma falha no carregamento derruba a subida. Um cliente vê então "conexão
  recusada", e não um 503. A checagem de saúde do Docker (4.5) precisa tratar os dois casos como
  "fora do ar". O 503 do `/health` existe para quando o app roda sem esse ciclo de subida, como nos
  testes.
- **A validação acontece duas vezes.** A primeira é no formato do pedido, com pydantic em modo
  `strict`: `"5"`, `true` ou `3.0` não viram nota, e campo desconhecido é erro. Ela gera o 422 e a
  documentação do `/docs`. A segunda é no gerador, que não confia em quem o chama. Um erro dessa
  segunda validação (por exemplo, um top-k maior que o vocabulário) também vira 422, e nunca 500.
- **Dois erros 500 encontrados na revisão independente, e corrigidos:**
  - **Temperatura positiva minúscula.** Com `1e-40`, a divisão dos logits estourava o float32, e o
    sorteio quebrava. A API agora exige temperatura de pelo menos 0,1, o mesmo mínimo da página. O
    gerador não foi mudado, porque dois estágios do DVC dependem dele, um deles congelado.
  - **`NaN` e `Infinity` no corpo.** O JSON do Python aceita esses valores, e o 422 padrão do
    FastAPI tentava devolver o valor recebido, que não cabe numa resposta JSON. O 422 agora devolve
    só o tipo, o lugar e a mensagem do erro.
- **O 503 traz os motivos** quando as 5 tentativas de um texto caem nos filtros, e derruba o pedido
  inteiro: o produto não entrega menos resenhas do que as pedidas sem avisar. Com a decodificação
  padrão, isso não aconteceu nenhuma vez em 3.000 resenhas (4.2 e teste final). Longe do padrão,
  acontece: em 20 pedidos de uma resenha (5 notas × 4 sementes), deram 503 dois com temperatura
  0,1, um com 2,0 e nenhum com 0,5 ou 1,0. A página avisa isso perto das opções avançadas.
- **A semente usada volta na resposta.** Sem semente, a API sorteia uma e a devolve. O resultado
  continua aleatório, mas um bom texto pode ser repetido depois. A semente sorteada fica abaixo de
  2⁵³, o maior inteiro que o JavaScript guarda sem arredondar; sem isso, a página mostraria uma
  semente que não repete o texto.
- **Toda resposta diz o que o texto é** (`"synthetic": true` e um aviso). É a seção de ética do
  charter aplicada ao produto. A página mostra o mesmo aviso no topo.
- **A rota de geração é uma função comum (`def`), e não `async`.** Gerar é trabalho de CPU, e o
  FastAPI roda funções comuns numa thread separada. Assim o `/health` continua respondendo durante
  uma geração.
- **A página são três arquivos, sem bibliotecas externas:** a estrutura (`index.html`), o estilo
  (`style.css`) e o comportamento (`app.js`). A API serve o CSS e o JavaScript em `/static`, e a
  página funciona offline na demonstração. O texto do modelo entra na página como texto
  (`textContent`), e nunca como HTML: nada do que o modelo escreve vira código no navegador.
- **Os testes não dependem do checkpoint real.** A API recebe o carregador como parâmetro
  (`create_app(load)`), e os testes passam um modelo minúsculo.
- **Limite conhecido, a concorrência.** Pedidos simultâneos dividem a mesma CPU: o torch já usa
  vários núcleos por pedido, e vários pedidos ao mesmo tempo disputam os mesmos núcleos. Para a
  demonstração, com um usuário, isso não pesa. O modelo é compartilhado entre os pedidos, e isso é
  seguro porque ele fica sempre em modo de avaliação e cada pedido tem o seu próprio sorteador.

**Teste de fumaça com o modelo real (20/11/2026):** o `/health` respondeu com 5.240.320 parâmetros e o
hash do checkpoint igual ao registrado no DVC. O `/generate` devolveu 5 resenhas em 1,4 s, todas na
primeira tentativa. Um pedido inválido deu 422, e a página e o `/docs` responderam 200.
