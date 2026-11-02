// Логика демо «Формы и параметры блока» (модуль 12, уроки 12.4 и 12.20). Формы тензоров по шагам урока 12.4,
// параметры TransformerBlock и SimpleLLM урока 12.20: блок - две LayerNorm (4·D), attention 4·D² (без bias),
// FFN 2·D·ff + ff + D; модель - эмбеддинги токенов V·D и позиций C·D, N блоков, final LayerNorm 2·D,
// выходной слой D·V без bias. T - длина входа, C - контекст (max_seq_length).
// Чистые функции без DOM - сверяются с golden.json.

export const PRESETS = [
  { key: "lesson", B: 2, T: 5, C: 5, D: 16, H: 2, k: 4, N: 1, V: 5 },
  { key: "model", B: 2, T: 4, C: 4, D: 16, H: 2, k: 4, N: 2, V: 50257 },
  { key: "gpt2", B: 1, T: 16, C: 1024, D: 768, H: 12, k: 4, N: 12, V: 50257 },
];
const GPT2 = PRESETS[2];
const GPT2_REAL = 124439808;
export const PARSE = "Словарь V - целое от 1 до 100 000, контекст C - от 1 до 4 096.";

function parseInt_(text, top) {
  const t = String(text).trim();
  if (!/^\d+$/.test(t)) return null;
  const v = Number(t);
  return v >= 1 && v <= top ? v : null;
}

export const parseVocab = (text) => parseInt_(text, 100000);
export const parseCtx = (text) => parseInt_(text, 4096);

/** Разряды через пробел: 1614752 → «1 614 752». */
export const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}

const shape = (...dims) => `[${dims.join(", ")}]`;

/** Цепочка форм от token_ids до logits; если D не делится на H, размер головы - «?». */
export function shapes(p) {
  const { B, T, D, H, V, N } = p;
  const ff = p.k * D;
  const hd = D % H === 0 ? D / H : "?";
  return [
    ["token_ids", shape(B, T)], ["эмбеддинги токенов + позиций", shape(B, T, D)],
    ["LayerNorm 1", shape(B, T, D)], ["Q, K, V", shape(B, T, D)], ["головы Q, K, V", shape(B, H, T, hd)],
    ["scores и веса", shape(B, H, T, T)], ["веса · V", shape(B, H, T, hd)], ["склейка голов, W_O", shape(B, T, D)],
    ["residual 1", shape(B, T, D)], ["LayerNorm 2", shape(B, T, D)], ["FFN: Linear 1, ReLU", shape(B, T, ff)],
    ["FFN: Linear 2", shape(B, T, D)], ["residual 2 = выход блока", shape(B, T, D)],
    [`после ${N} ${plural(N, "блока", "блоков", "блоков")}: final LayerNorm`, shape(B, T, D)], ["logits", shape(B, T, V)],
  ].map(([step, sh]) => ({ step, shape: sh }));
}

export function blockParams(D, H, ff) {
  const ln = 4 * D;
  const attn = 4 * D * D;
  const ffn = 2 * D * ff + ff + D;
  return { ln, attn, ffn, total: ln + attn + ffn };
}

export function modelParams(p) {
  const { D } = p;
  const b = blockParams(D, p.H, p.k * D);
  const out = { tok: p.V * D, pos: p.C * D, blocks: p.N * b.total, final: 2 * D, head: D * p.V };
  out.total = out.tok + out.pos + out.blocks + out.final + out.head;
  return out;
}

/** Текст assert из MultiHeadAttention урока 10.12 или пустая строка. */
export function checkText(D, H) {
  return D % H === 0 ? "" : `embed_dim (${D}) должен делиться на num_heads (${H}) без остатка`;
}

/** Таблица параметров: названия из кода (code = true) или по-русски. */
export function paramRows(p, code) {
  const m = modelParams(p);
  const { D, V, C, N } = p;
  const names = code
    ? ["token_embedding", "position_embedding", `blocks (${N} × TransformerBlock)`, "final_norm", "lm_head", "get_num_parameters()"]
    : ["эмбеддинги токенов", "эмбеддинги позиций", `${N} ${plural(N, "блок", "блока", "блоков")}`, "final LayerNorm", "выходной слой", "всего"];
  const formulas = [`${V} × ${D}`, `${C} × ${D}`, `${N} × ${group(blockParams(D, p.H, p.k * D).total)}`, `2 × ${D}`, `${D} × ${V}`, ""];
  return ["tok", "pos", "blocks", "final", "head", "total"].map((key, i) => ({ label: names[i], formula: formulas[i], value: group(m[key]) }));
}

export function blockText(p, code) {
  const { D } = p;
  const ff = p.k * D;
  const b = blockParams(D, p.H, ff);
  const [ln, attn, ffn] = code ? ["norm_1 + norm_2", "attention", "ffn"] : ["две LayerNorm", "attention", "FFN"];
  return `один блок: ${ln} 4 × ${D} = ${group(b.ln)}, ${attn} 4 × ${D}² = ${group(b.attn)}, `
    + `${ffn} 2 × ${D} × ${ff} + ${ff} + ${D} = ${group(b.ffn)}, всего ${group(b.total)}`;
}

export function captionText(p) {
  if (p.D % p.H) return "Классы урока 12.20 остановятся на assert: выберите число голов, на которое делится D.";
  const m = modelParams(p);
  if (["C", "D", "H", "k", "N", "V"].every((key) => p[key] === GPT2[key])) {
    return `По нашему классу у GPT-2 выходит ${group(m.total)} параметров. Настоящая GPT-2 делит матрицу `
      + `эмбеддингов с выходным слоем и добавляет bias в attention, у неё ${group(GPT2_REAL)}.`;
  }
  const pct = (v) => ((100 * v) / m.total).toFixed(1);
  const { N } = p;
  return `Блок не меняет форму [${p.B}, ${p.T}, ${p.D}], поэтому блоки можно ставить друг за другом. `
    + `Эмбеддинги токенов и выходной слой - ${pct(m.tok + m.head)}% параметров, `
    + `${N} ${plural(N, "блок", "блока", "блоков")} - ${pct(m.blocks)}%.`;
}
