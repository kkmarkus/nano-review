// Comportamento da página: escolher a nota, chamar o /generate e mostrar as resenhas.
// Sem bibliotecas externas, para a demonstração funcionar offline.
let rating = 5;

const numberFormat = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 2 });

// Os botões de nota funcionam como um grupo: só um fica marcado.
document.querySelectorAll("#ratings button").forEach((button) => {
  button.addEventListener("click", () => {
    rating = Number(button.dataset.rating);
    document.querySelectorAll("#ratings button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b === button));
    });
  });
});

// Campo vazio quer dizer "use o padrão": ele não entra no pedido. Campo preenchido precisa ser um
// número de verdade: "1.5" num campo inteiro ou "12abc" viram erro, e não um número cortado.
function readField(id, { integer }) {
  const input = document.getElementById(id);
  // Num campo numérico, o navegador entrega texto inválido ("12abc") como vazio; o badInput avisa.
  if (input.validity.badInput) return { error: true };
  const raw = input.value.trim();
  if (raw === "") return { value: undefined };
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isSafeInteger(value))) {
    return { error: true };
  }
  return { value };
}

// A API devolve o erro de validação como lista (pydantic) ou como texto (o gerador).
function errorMessage(status, detail) {
  if (status === 422 && Array.isArray(detail)) {
    return "Pedido inválido: " + detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ");
  }
  if (status === 503 && detail && detail.message) {
    return "Não foi possível gerar agora: " + detail.message;
  }
  return typeof detail === "string" ? detail : `Erro ${status}`;
}

function showReview(review) {
  const item = document.createElement("div");
  item.className = "review";
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = `nota ${review.rating}`;
  // textContent, e não innerHTML: o texto do modelo nunca vira código na página.
  const text = document.createElement("p");
  text.textContent = review.text;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copiar";
  copy.addEventListener("click", async () => {
    // A área de transferência só existe em página segura (localhost ou https).
    if (!navigator.clipboard) {
      copy.textContent = "Indisponível";
      return;
    }
    await navigator.clipboard.writeText(review.text);
    copy.textContent = "Copiado";
  });
  item.append(tag, text, copy);
  document.getElementById("results").append(item);
}

function summaryText(data) {
  const count = data.reviews.length;
  const noun = count === 1 ? "resenha" : "resenhas";
  const topK = data.top_k === null ? "sem top-k" : `top-k ${data.top_k}`;
  return (
    `${count} ${noun} em ${numberFormat.format(data.seconds)} s · ` +
    `temperatura ${numberFormat.format(data.temperature)} · ${topK} · semente ${data.seed}`
  );
}

const button = document.getElementById("generate");

button.addEventListener("click", async () => {
  const error = document.getElementById("error");
  const results = document.getElementById("results");
  const summary = document.getElementById("summary");
  error.textContent = "";
  summary.textContent = "";

  const n = readField("n", { integer: true });
  const temperature = readField("temperature", { integer: false });
  const topK = readField("top_k", { integer: true });
  const seed = readField("seed", { integer: true });
  if (n.error || n.value === undefined || n.value < 1 || n.value > 20) {
    error.textContent = "A quantidade precisa ser um inteiro de 1 a 20.";
    return;
  }
  if (temperature.error || topK.error || seed.error) {
    error.textContent = "Temperatura, top-k e semente precisam ser números (top-k e semente, inteiros).";
    return;
  }

  // Um pedido por vez: dois cliques seguidos não podem misturar duas respostas na lista.
  button.disabled = true;
  results.textContent = "";
  summary.textContent = "Gerando…";
  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rating,
        n: n.value,
        temperature: temperature.value,
        top_k: topK.value,
        seed: seed.value,
      }),
    });
    // Um erro inesperado do servidor pode vir sem JSON; a página precisa dizer algo mesmo assim.
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      summary.textContent = "";
      error.textContent = errorMessage(response.status, data.detail);
      return;
    }
    data.reviews.forEach(showReview);
    summary.textContent = summaryText(data);
  } catch {
    summary.textContent = "";
    error.textContent = "Não foi possível falar com a API. Ela está rodando?";
  } finally {
    button.disabled = false;
  }
});

// O /health diz se o modelo está carregado antes de o usuário pedir qualquer coisa.
fetch("/health")
  .then((response) => (response.ok ? response.json() : Promise.reject(response.status)))
  .then((health) => {
    const millions = numberFormat.format(Math.round(health.parameters / 1e5) / 10);
    document.getElementById("health").textContent = `modelo carregado · ${millions} M parâmetros`;
  })
  .catch(() => {
    document.getElementById("health").textContent = "modelo indisponível";
  });
