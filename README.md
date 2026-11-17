# nano-review

Transformer pequeno, **treinado do zero** (sem modelo pré-treinado), que gera resenhas de
produto em português a partir de uma nota de 1 a 5. Projeto integrador individual.

- **Dados:** B2W-Reviews01 (licença CC BY-NC-SA 4.0; origem e esquema ficam no manifesto em `docs/`).
- **Usuário:** desenvolvedor que precisa de textos sintéticos para testar telas, moderação ou
  classificadores de sentimento. A IA não serve para publicar avaliações falsas.

## Status

Etapa 0 concluída: esqueleto do repositório, ambiente e porta de qualidade (`make check`).

## Como preparar o ambiente (WSL/Ubuntu, Python 3.12)

Mantenha o projeto dentro do sistema de arquivos do WSL (por exemplo `~/projetos/nano-review`),
fora de `/mnt/c` e de pastas sincronizadas pelo OneDrive.

```bash
# Se o comando abaixo reclamar de ensurepip/venv: sudo apt install python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# O torch (build CPU) vem do índice próprio da PyTorch; instalar antes do resto.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt
make check
```

`make check` roda `ruff` (estilo), `mypy --strict` (tipos) e `pytest` (testes). Ele precisa
passar antes de qualquer commit ou entrega.

## Estrutura

```
src/nano_review/   código do pacote (cada subpasta nasce na etapa que a usa)
tests/             testes automatizados
docs/              charter, manifesto de dados e relatório
notebooks/         exploração de dados
configs/           hiperparâmetros em YAML
```

Ainda não existem `data/` e `models/`: serão criadas na Etapa 1, já sob controle do DVC.
