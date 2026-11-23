# Reprodução limpa do zero (Etapa 5)

> Executada em 23/11/2026, a partir do commit `6994f2f`. Os critérios da parte C foram escritos
> **antes** de o pipeline rodar. Depois das correções, um segundo clone do GitHub, no commit
> `664ef01`, refez as partes A e B só pelo README publicado: instalação em 4,5 min, versões
> idênticas às do `.venv` original, 308 testes passando e o MD5 do CSV conferido. O código do
> pipeline não mudou entre os dois commits, então a parte C não foi repetida.

## Objetivo

Mostrar que quem tem só o link do repositório chega aos mesmos dados, ao mesmo modelo e aos mesmos
números seguindo apenas o [README](../README.md). Cada atrito encontrado virou correção no README.

**Como o teste ficou honesto:**

- Clone novo do GitHub em outra pasta (`~/repro/`), venv novo e `pip --no-cache-dir`, para baixar
  todas as bibliotecas de novo.
- O remote do DVC do clone foi apontado para uma pasta inexistente (`.dvc/config.local`, fora do
  Git), e o cache do DVC começou vazio. O clone não tinha como usar os dados já baixados.
- O modelo retreinado foi guardado à parte antes de baixar o do Release, para os dois serem
  comparados.

## Resultado

| Parte | O que foi feito | Resultado |
|---|---|---|
| A. Ambiente | venv, torch CPU e dependências só pelo README; `make check` | 308 testes passam; versões **idênticas** às do `.venv` original (depois da correção 1) |
| B. Dados brutos | CSV baixado da URL fixada com `dvc update` | MD5 `3d6bb9c7…` e SHA-256 `821fb0bf…` iguais aos do [manifesto](data_manifest.md); byte a byte igual ao original |
| C. Pipeline | `dvc repro` de todos os estágios, menos o `final_eval` | **reprodução exata**: todas as saídas bit a bit iguais, inclusive o checkpoint |
| D. Produto | modelo do Release, API pelo `uvicorn` e pelo Docker | SHA-256 `OK`; Release idêntico ao modelo retreinado; contêiner `healthy` |

### Parte C, estágio por estágio

Critérios escritos antes de rodar: os estágios de dados precisam sair bit a bit iguais. O checkpoint
pode mudar de hash (somas em várias threads da CPU podem mudar de ordem), e nesse caso a perplexity
de validação precisaria ficar a até 1% da original, com o mesmo veredito em H1 a H6.

| Estágio | Duração no clone | Saídas comparadas com o `dvc.lock` original |
|---|---|---|
| `clean` | 13 s | iguais |
| `split` | 2 s | iguais (`train`, `validation`, `test`) |
| `judge` | 33 s | iguais |
| `tokenizer` | 18 s | iguais |
| `baseline` | 2,2 min | iguais |
| `train` | 72,5 min | `transformer.pt` e curva iguais; ver a nota sobre o tempo de relógio |
| `evaluate` | 3,9 min | iguais (H1 a H6) |
| `decoding` | 30,7 min | iguais |
| `inference_check` | 5,4 min | amostras iguais; ver a nota sobre o tempo de relógio |
| `final_eval` | não roda | congelado: o teste só é aberto uma vez ([`final_test.md`](final_test.md)) |

A cláusula de tolerância não precisou ser usada: o checkpoint saiu com o mesmo hash.

**Tempo de relógio.** Dois arquivos de métricas aparecem como diferentes no `dvc.lock`, mas só pelo
que depende da carga da máquina: `train_seconds` (4.297 s contra 3.936 s) e `elapsed_seconds` no
`train.json` e no `train_log.csv`, os milissegundos por resenha no `inference.json` e o id do run no
MLflow. Loss de treino, loss de validação, learning rate e norma do gradiente são iguais nos 60
pontos da curva.

**Estágios pulados.** Na primeira passada, o DVC pulou `baseline`, `decoding` e `inference_check`:
as saídas deles estão no Git e as entradas não mudaram, então pela lógica do DVC não havia nada a
refazer. Para a comparação não ser trivial, os três foram forçados com `dvc repro -f -s <estágio>`.

**Por que saiu idêntico.** A semente fixa (`set_seed`) define a inicialização, o dropout, a ordem dos
lotes e a amostragem; o split é determinístico; e o lock de versões garante as mesmas bibliotecas
fazendo as mesmas contas. O `set_seed` não liga `torch.use_deterministic_algorithms`, e só valeria
ligar se aparecesse divergência. Nesta máquina não apareceu.

### Parte D, produto

| Verificação | Resultado |
|---|---|
| Download do Release `v0.1.0` e `sha256sum -c` | 12 s, os dois arquivos `OK` |
| Release comparado com o modelo retreinado na parte C | byte a byte idênticos |
| API pelo `uvicorn` | `/health` ok (MD5 `9181cbbd…`), página 200, nota 9 dá 422, a mesma semente repete o texto, `"synthetic": true` |
| `docker build --no-cache` | 169 s, imagem de 1,99 GB |
| `make docker-run` | API responde em ~8 s, usuário `app` (uid 1000, sem root), estado `healthy` |
| Mesmo pedido (nota 5, semente 7) no `uvicorn` e no Docker | os mesmos textos |

## Atritos encontrados e corrigidos

1. **As versões mudavam a cada instalação.** O `requirements.txt` só tem versões mínimas
   (`torch>=2.4`…). Na instalação nova, as bibliotecas principais coincidiram, mas 16 dependências
   indiretas vieram mais novas. Com o tempo, o torch e o pandas também mudariam, e com eles os
   números. **Correção:** `requirements-lock.txt` com as versões exatas do `.venv` que gerou os
   resultados, o torch fixado em `2.15.0` no README e o alvo `make lock` para regenerar o lock. O
   `requirements.txt` continua com os mínimos: ele diz do que o projeto depende, e o lock diz com
   quais versões os números foram obtidos.
2. **O `dvc pull` do README falhava num clone novo** ("missing cache files"), porque o remote do DVC
   é uma pasta desta máquina. **Correção:** o README baixa o CSV da fonte com
   `dvc update data/raw/B2W-Reviews01.csv.dvc`, confere o MD5 e gera o resto com `dvc repro`.
3. **O tempo do pipeline estava subestimado.** O README dizia ~66 min de treino. No clone foram
   72,5 min, e o pipeline inteiro levou ~2 h (o `decoding` sozinho, 31 min). **Correção:** o README
   passou a dar as duas estimativas.

Cuidados que não viraram correção, mas valem para a demonstração:

- **Porta ocupada.** No primeiro teste da API, outro contêiner já estava na porta 8000: o `uvicorn`
  do clone não subiu, e os pedidos foram respondidos pelo outro processo. Só o log do `uvicorn`
  mostrou o erro. Antes de subir a API, conferir a porta (`ss -ltn | grep 8000`).
- **Cache do Docker.** Um `make docker-build` numa máquina que já construiu a imagem reaproveita as
  camadas e leva 1 s. Para provar que o `Dockerfile` funciona do zero, é preciso `--no-cache`.

## Limitações

- **Mesma máquina.** A reprodução rodou no mesmo computador (AMD Ryzen 7 4800H, 16 núcleos, WSL2,
  Ubuntu 26.04, Python 3.12.14). Ela prova o caminho "clone, README, resultado", não outro sistema
  operacional nem outra CPU. Em outro processador, as somas em ponto flutuante podem sair em outra
  ordem, e o checkpoint pode não ser bit a bit igual. Nesse caso vale o critério de tolerância da
  parte C.
- **A imagem Docker** cobre parte dessa lacuna para a API: ela fixa sistema, Python e bibliotecas.
- **A fonte dos dados é externa.** Se o GitHub da americanas-tech sair do ar, o `dvc update` falha. O
  MD5 do README pelo menos detecta se o arquivo mudou.
- **A rubrica humana não se reproduz por definição.** As planilhas pontuadas estão versionadas em
  `reports/rubric/`, e os comandos que montam os lotes cegos estão na [referência](reference.md).
- **O `final_eval` não foi refeito**, de propósito. Como o checkpoint saiu idêntico, o resultado do
  teste vale para o modelo reproduzido sem abrir o teste de novo.
