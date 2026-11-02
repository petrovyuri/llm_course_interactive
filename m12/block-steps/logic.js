// Логика демо «Блок по шагам» (модуль 12, уроки 12.9-12.19). Пример урока 12.8: «кот», «любил», «сметану»,
// embed_dim 4, одна голова, W_Q = W_K = W_V = I, деление на √4 = 2, Pre-LN, LayerNorm без γ, β и eps,
// FFN 4 → 3 → 4 без bias, выходной слой W_vocab 4×5.
// Правило уроков: итог каждого шага считается точно и показывается с 3 знаками (ROUND_HALF_UP), строки
// расчёта - из показанных чисел; «=» стоит, только если выражение из показанных чисел точно равно итогу.
// Код урока 12.20 (nn.LayerNorm с eps = 1e-5) в 3-м знаке иногда даёт другое - тогда в конце «(код: …)».
// Чистые функции без DOM - сверяются с golden.json.

export const STAGES = ["ln1", "qkv", "scores", "softmax", "attn", "res1", "ln2", "ffn", "res2", "logits", "probs"];
export const TOKENS = ["кот", "любил", "сметану"];
export const VOCAB = ["кот", "любил", "сметану", "он", "был"];
export const X = [[500, 200, 800, 100], [400, 900, 300, 700], [600, 100, 500, 400]];   // тысячные
export const W1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]];                          // целые
export const W2 = [[0, 0, 0, 0], [0, 0, 0, 0], [2, 1, 3, -2]];                           // десятые
export const WV = [[2, 0, 1, 3, 0], [1, 3, 0, 1, 2], [0, 2, 4, 1, 1], [1, 0, 0, 2, 3]];  // десятые
const SUB = ["₁", "₂", "₃"];
const R3 = [0, 1, 2];
const R4 = [0, 1, 2, 3];
const R5 = [0, 1, 2, 3, 4];
const GPT_MASK = "В GPT маска стоит всегда, урок для простоты считает без неё.";
const MATRIX = { ln1: "xhat", qkv: "xhat", scores: "S", softmax: "A", attn: "attn", res1: "h",
  ln2: "hhat", ffn: "ffn", res2: "y", logits: "logits", probs: "p" };
const CODE_KEYS = ["xhat", "S", "A", "attn", "h", "hhat", "z", "relu", "ffn", "y", "logits", "p"];

/** num / den с округлением половины от нуля (den > 0). */
function halfUp(num, den) {
  const q = Math.floor((2 * Math.abs(num) + den) / (2 * den));
  return num < 0 ? -q : q;
}

/** Целое в единицах 10^-k → строка с k знаками: units(365, 3) = «0.365», units(-2, 1) = «-0.2». */
export function units(n, k = 4) {
  const s = String(Math.abs(n)).padStart(k + 1, "0");
  return `${n < 0 ? "-" : ""}${s.slice(0, -k)}.${s.slice(-k)}`;
}

/** Число → целые тысячные, ROUND_HALF_UP. */
function th(x) {
  const q = Math.floor(Math.abs(x) * 1000 + 0.5);
  return x < 0 && q ? -q : q;
}

const sum = (a) => a.reduce((s, v) => s + v, 0);
const n3 = (v) => units(v, 3);
const par = (v) => (v < 0 ? `(${n3(v)})` : n3(v));
const tenth = (v) => (v < 0 ? `(${units(v, 1)})` : units(v, 1));
const rel = (equal) => (equal ? "=" : "≈");
const same = (a, b) => (Array.isArray(a) ? a.length === b.length && a.every((v, i) => same(v, b[i])) : a === b);

/** Слагаемые с пробелами: «0.250 - 0.012 + 0.320», ноль - «0». */
function terms(ts) {
  return ts.map((t, i) => {
    const s = t === 0 ? "0" : n3(Math.abs(t));
    return (i === 0 ? (t < 0 ? "-" : "") : t < 0 ? " - " : " + ") + s;
  }).join("");
}

function vec(vs, zero = "0.000") {
  return `[${vs.map((v) => (v === 0 ? zero : n3(v))).join(", ")}]`;
}

function layerNorm(rows, eps) {
  return rows.map((r) => {
    const mu = sum(r) / 4;
    const v = sum(r.map((a) => (a - mu) ** 2)) / 4;
    const s = Math.sqrt(v + eps);
    return { mu, v, s, n: r.map((a) => (a - mu) / s) };
  });
}

/** Softmax строки; null - закрытая маской клетка (вес 0). */
function softmax(row) {
  const e = row.map((s) => (s === null ? 0 : Math.exp(s)));
  const total = sum(e);
  return e.map((v) => v / total);
}

/** Прямой проход примера: eps = 0 - уроки, eps = 1e-5 - nn.LayerNorm кода. */
function forward(mask, eps) {
  const x = X.map((r) => r.map((v) => v / 1000));
  const ln1 = layerNorm(x, eps);
  const xhat = ln1.map((o) => o.n);
  const S = R3.map((i) => R3.map((j) => (mask && j > i ? null : sum(R4.map((d) => xhat[i][d] * xhat[j][d])) / 2)));
  const A = S.map(softmax);
  const attn = R3.map((i) => R4.map((d) => sum(R3.map((j) => A[i][j] * xhat[j][d]))));
  const h = R3.map((i) => R4.map((d) => x[i][d] + attn[i][d]));
  const ln2 = layerNorm(h, eps);
  const hhat = ln2.map((o) => o.n);
  const z = R3.map((i) => R3.map((k) => sum(R4.map((d) => hhat[i][d] * W1[d][k]))));
  const relu = z.map((r) => r.map((v) => Math.max(0, v)));
  const ffn = R3.map((i) => R4.map((d) => sum(R3.map((k) => relu[i][k] * W2[k][d])) / 10));
  const y = R3.map((i) => R4.map((d) => h[i][d] + ffn[i][d]));
  const logits = R3.map((i) => R5.map((v) => sum(R4.map((d) => y[i][d] * WV[d][v])) / 10));
  return { ln1, ln2, xhat, S, A, attn, h, hhat, z, relu, ffn, y, logits, p: logits.map(softmax) };
}

const chains = new Map();
const codes = new Map();

/** Все шаги урока в целых тысячных; с маской закрытые scores и экспоненты - null, их веса - 0. */
export function chain(mask) {
  if (chains.has(mask)) return chains.get(mask);
  const f = forward(mask, 0);
  const c = {};
  for (const key of CODE_KEYS) c[key] = f[key].map((row) => row.map((v) => (v === null ? null : th(v))));
  for (const [key, ln] of [["", f.ln1], ["2", f.ln2]]) {
    c[`mu${key}`] = ln.map((o) => th(o.mu));
    c[`var${key}`] = ln.map((o) => th(o.v));
    c[`sigma${key}`] = ln.map((o) => th(o.s));
  }
  const xh = c.xhat;
  c.P = R3.map((i) => R3.map((j) => R4.map((d) => halfUp(xh[i][d] * xh[j][d], 1000))));
  c.E = c.S.map((row) => row.map((s) => (s === null ? null : th(Math.exp(s / 1000)))));
  c.sum = c.E.map((row) => sum(row.filter((v) => v !== null)));
  c.T = R3.map((i) => R4.map((d) => R3.map((j) => halfUp(c.A[i][j] * xh[j][d], 1000))));
  c.L = R3.map((i) => R5.map((v) => R4.map((d) => halfUp(c.y[i][d] * WV[d][v], 10))));
  c.E2 = c.logits.map((row) => row.map((s) => th(Math.exp(s / 1000))));
  c.sum2 = c.E2.map(sum);
  // LayerNorm 2 в строках расчёта - с 4 знаками: с 3 знаками (h - μ) / √σ² расходится с ĥ на калькуляторе
  const th4 = (x) => { const q = Math.floor(Math.abs(x) * 10000 + 0.5); return x < 0 && q ? -q : q; };
  c.h4 = f.h.map((row) => row.map(th4));
  c.mu2_4 = f.ln2.map((o) => th4(o.mu));
  c.d2_4 = f.h.map((row, i) => row.map((v) => th4(v - f.ln2[i].mu)));
  c.q2_4 = f.h.map((row, i) => row.map((v) => th4((v - f.ln2[i].mu) ** 2)));
  c.var2_4 = f.ln2.map((o) => th4(o.v));
  chains.set(mask, c);
  return c;
}

/** Числа кода без округлений: TransformerBlock урока 12.20 (eps = 1e-5) и выходной слой. */
export function code(mask) {
  const f = forward(mask, 1e-5);
  return Object.fromEntries(CODE_KEYS.map((key) => [key, f[key]]));
}

function codeTh(mask) {
  if (!codes.has(mask)) {
    const k = code(mask);
    codes.set(mask, Object.fromEntries(CODE_KEYS.map((key) => [key, k[key].map((row) => row.map((v) => (v === null ? null : th(v))))])));
  }
  return codes.get(mask);
}

/** Хвост «(код: …)», если 3 знака кода другие; manual, coded - число или список. */
function withCode(text, manual, coded) {
  if (same(manual, coded)) return text;
  return `${text} (код: ${Array.isArray(manual) ? vec(coded) : n3(coded)})`;
}

/** Главная таблица этапа: 3 строки токенов. */
export function matrix(stage, mask) {
  return chain(mask)[MATRIX[stage]].map((row) => row.map((v) => (v === null ? "-∞" : n3(v))));
}

/** Точная десятичная запись без лишних нулей: exact(40000, 5) = «0.4». */
function exact(v, k) {
  const t = units(v, k).replace(/0+$/, "").replace(/\.$/, "");
  return t === "0" || t === "-0" ? "0" : t;
}

/** LayerNorm 1 точно: μ и разности в 1e-5, квадраты в 1e-10, дисперсия в 1e-12 (вход - тысячные). */
function ln1Ints(i) {
  const mu5 = sum(X[i]) * 25;
  const d = X[i].map((v) => v * 100 - mu5);
  const q = d.map((v) => v * v);
  return { mu5, d, q, var12: sum(q) * 25 };
}

function lnLines(c, k, i, second) {
  if (second) {
    const v4 = units(c.var2_4[i], 4);
    return [`μ = (${c.h[i].map(par).join(" + ")}) / 4 ${rel(sum(c.h[i]) === 4 * c.mu2[i])} ${n3(c.mu2[i])}`,
      `h - μ ≈ [${c.d2_4[i].map((v) => units(v, 4)).join(", ")}]`,
      `σ² = (${c.q2_4[i].map((v) => units(v, 4)).join(" + ")}) / 4 ≈ ${v4}`,
      `σ = √${v4} ≈ ${n3(c.sigma2[i])}`,
      withCode(`ĥ ≈ ${vec(c.hhat[i])}`, c.hhat[i], k.hhat[i])];
  }
  const { mu5, d, q, var12 } = ln1Ints(i);
  const s3 = c.sigma[i];
  return [`μ = (${X[i].map((v) => units(v / 100, 1)).join(" + ")}) / 4 = ${exact(mu5, 5)}`,
    `x - μ = [${d.map((v) => exact(v, 5)).join(", ")}]`,
    `σ² = (${q.map((v) => exact(v, 10)).join(" + ")}) / 4 = ${exact(var12, 12)}`,
    `σ = √${exact(var12, 12)} ${rel(s3 * s3 * 10 ** 6 === var12)} ${n3(s3)}`,
    withCode(`x̂ ≈ ${vec(c.xhat[i])}`, c.xhat[i], k.xhat[i])];
}

function scoreLine(c, k, i, j) {
  const s = `s${SUB[i]}${SUB[j]}`;
  if (c.S[i][j] === null) return `${s} = -∞: маска закрывает «${TOKENS[j]}»`;
  const a = c.xhat[i];
  const b = c.xhat[j];
  const P = c.P[i][j];
  const prods = R4.map((d) => `${par(a[d])}·${par(b[d])}`).join(" + ");
  const exact = R4.every((d) => (a[d] * b[d]) % 1000 === 0);
  return withCode(`${s} = (${prods}) / 2 ${rel(exact)} (${terms(P)}) / 2 ${rel(sum(P) === 2 * c.S[i][j])} ${n3(c.S[i][j])}`,
    c.S[i][j], k.S[i][j]);
}

function sumLine(exps, values, total) {
  if (values.length === 1) return `Σ = ${exps[0]} ≈ ${n3(total)}`;
  return `Σ = ${exps.join(" + ")} ≈ ${terms(values)} = ${n3(total)}`;
}

function weightLine(c, k, i, j) {
  const a = `a${SUB[i]}${SUB[j]}`;
  if (c.E[i][j] === null) return `${a} = 0: маска закрывает «${TOKENS[j]}»`;
  const e = c.E[i][j];
  const total = c.sum[i];
  const A = c.A[i][j];
  return withCode(`${a} = ${n3(e)} / ${n3(total)} ${rel(e * 1000 === A * total)} ${n3(A)}`, A, k.A[i][j]);
}

function attnLine(c, k, i, d) {
  const A = c.A[i];
  const T = c.T[i][d];
  const v = c.attn[i][d];
  const prods = R3.map((j) => `${n3(A[j])}·${par(c.xhat[j][d])}`).join(" + ");
  const exact = R3.every((j) => (A[j] * c.xhat[j][d]) % 1000 === 0);
  return withCode(`attn${SUB[i]}[${d + 1}] = ${prods} ${rel(exact)} ${terms(T)} ${rel(sum(T) === v)} ${n3(v)}`, v, k.attn[i][d]);
}

function resLine(c, k, i, d, second) {
  const [a, b, r, key, left] = second
    ? [c.h[i][d], c.ffn[i][d], c.y[i][d], "y", n3(c.h[i][d])]
    : [X[i][d], c.attn[i][d], c.h[i][d], "h", units(X[i][d] / 100, 1)];
  return withCode(`${key}${SUB[i]}[${d + 1}] = ${left} + ${par(b)} ${rel(a + b === r)} ${n3(r)}`, r, k[key][i][d]);
}

function zLine(c, k, i, j) {
  const ts = R4.filter((d) => W1[d][j]).map((d) => c.hhat[i][d]);
  const z = c.z[i][j];
  return withCode(`z${SUB[j]} = ${terms(ts)} ${rel(sum(ts) === z)} ${n3(z)}`, z, k.z[i][j]);
}

const active = (c, i) => R3.filter((j) => c.relu[i][j] && W2[j].some(Boolean));

/** Почему выход FFN нулевой: все z обнулены или выжившие z стоят против нулевых строк W₂. */
function whyZero(c, i) {
  const alive = R3.filter((j) => c.relu[i][j]);
  if (!alive.length) return "ReLU обнулил все z";
  const names = alive.map((j) => `z${SUB[j]}`).join(", ");
  return `${names} ${alive.length === 1 ? "попадает на нулевую строку" : "попадают на нулевые строки"} W₂`;
}

function ffnLine(c, k, i) {
  const act = active(c, i);
  const f = c.ffn[i];
  if (!act.length) return withCode(`ffn = ReLU(z) · W₂ = ${vec(f)}: ${whyZero(c, i)}`, f, k.ffn[i]);
  const parts = act.map((j) => `${n3(c.relu[i][j])} · [${W2[j].map((w) => units(w, 1)).join(", ")}]`).join(" + ");
  const exact = R4.every((d) => sum(act.map((j) => c.relu[i][j] * W2[j][d])) === f[d] * 10);
  return withCode(`ffn = ReLU(z) · W₂ = ${parts} ${rel(exact)} ${vec(f)}`, f, k.ffn[i]);
}

function ffnCell(c, k, i, d) {
  const act = active(c, i);
  const f = c.ffn[i][d];
  if (!act.length) return `ffn${SUB[i]}[${d + 1}] = 0: ${whyZero(c, i)}`;
  const prods = act.map((j) => `${n3(c.relu[i][j])}·${tenth(W2[j][d])}`).join(" + ");
  const exact = sum(act.map((j) => c.relu[i][j] * W2[j][d])) === f * 10;
  return withCode(`ffn${SUB[i]}[${d + 1}] = ${prods} ${rel(exact)} ${n3(f)}`, f, k.ffn[i][d]);
}

function logitLine(c, k, i, v) {
  const y = c.y[i];
  const L = c.L[i][v];
  const lg = c.logits[i][v];
  const prods = R4.map((d) => `${par(y[d])}·${units(WV[d][v], 1)}`).join(" + ");
  const exact = R4.every((d) => (y[d] * WV[d][v]) % 10 === 0);
  return withCode(`logit(${VOCAB[v]}) = ${prods} ${rel(exact)} ${terms(L)} ${rel(sum(L) === lg)} ${n3(lg)}`, lg, k.logits[i][v]);
}

function probLine(c, k, i, v) {
  const e = c.E2[i][v];
  const total = c.sum2[i];
  const p = c.p[i][v];
  return withCode(`p(${VOCAB[v]}) = ${n3(e)} / ${n3(total)} ${rel(e * 1000 === p * total)} ${n3(p)}`, p, k.p[i][v]);
}

/** Строки расчёта этапа для токена i (их по одной показывает «▶ по шагам»). */
export function lines(stage, i, mask) {
  const c = chain(mask);
  const k = codeTh(mask);
  if (stage === "ln1" || stage === "ln2") return lnLines(c, k, i, stage === "ln2");
  if (stage === "qkv") {
    return ["W_Q = W_K = W_V = I, поэтому Q = K = V = x̂",
      withCode(`q${SUB[i]} = k${SUB[i]} = v${SUB[i]} ≈ ${vec(c.xhat[i])}`, c.xhat[i], k.xhat[i])];
  }
  if (stage === "scores") return R3.map((j) => scoreLine(c, k, i, j));
  if (stage === "softmax") {
    const open = R3.filter((j) => c.E[i][j] !== null);
    return [sumLine(open.map((j) => `e^${par(c.S[i][j])}`), open.map((j) => c.E[i][j]), c.sum[i]),
      ...R3.map((j) => weightLine(c, k, i, j))];
  }
  if (stage === "attn") return R4.map((d) => attnLine(c, k, i, d));
  if (stage === "res1" || stage === "res2") return R4.map((d) => resLine(c, k, i, d, stage === "res2"));
  if (stage === "ffn") return [...R3.map((j) => zLine(c, k, i, j)), `ReLU(z) = ${vec(c.relu[i], "0")}`, ffnLine(c, k, i)];
  if (stage === "logits") return R5.map((v) => logitLine(c, k, i, v));
  return [sumLine(c.logits[i].map((v) => `e^${par(v)}`), c.E2[i], c.sum2[i]), ...R5.map((v) => probLine(c, k, i, v))];
}

/** Строка расчёта клетки [i][j] главной таблицы этапа. */
export function cellOf(stage, i, j, mask) {
  const c = chain(mask);
  const k = codeTh(mask);
  if (stage === "ln2") {
    const text = `ĥ${SUB[i]}[${j + 1}] = (${units(c.h4[i][j], 4)} - ${units(c.mu2_4[i], 4)}) / `
      + `√${units(c.var2_4[i], 4)} ≈ ${n3(c.hhat[i][j])}`;
    return withCode(text, c.hhat[i][j], k.hhat[i][j]);
  }
  if (stage === "ln1") {
    const { mu5, var12 } = ln1Ints(i);
    return withCode(`x̂${SUB[i]}[${j + 1}] = (${units(X[i][j] / 100, 1)} - ${exact(mu5, 5)}) / √${exact(var12, 12)} ≈ ${n3(c.xhat[i][j])}`,
      c.xhat[i][j], k.xhat[i][j]);
  }
  if (stage === "qkv") {
    const s = `${SUB[i]}[${j + 1}]`;
    return withCode(`q${s} = k${s} = v${s} = x̂${s} ≈ ${n3(c.xhat[i][j])}`, c.xhat[i][j], k.xhat[i][j]);
  }
  if (stage === "scores") return scoreLine(c, k, i, j);
  if (stage === "softmax") return weightLine(c, k, i, j);
  if (stage === "attn") return attnLine(c, k, i, j);
  if (stage === "res1" || stage === "res2") return resLine(c, k, i, j, stage === "res2");
  if (stage === "ffn") return ffnCell(c, k, i, j);
  if (stage === "logits") return logitLine(c, k, i, j);
  return probLine(c, k, i, j);
}

export function captionText(stage, i, mask) {
  const t = TOKENS[i];
  const p = chain(mask).p[i];
  const win = VOCAB[R5.reduce((best, v) => (p[v] > p[best] ? v : best), 0)];
  const base = {
    ln1: `LayerNorm вычитает из строки «${t}» среднее и делит на σ: у x̂ среднее 0 и разброс 1.`,
    qkv: `W_Q, W_K и W_V здесь единичные, поэтому q, k и v у «${t}» совпадают с x̂.`,
    scores: `Score - скалярное произведение q «${t}» и k каждого слова, делённое на √4 = 2.`,
    softmax: `Softmax превращает scores строки «${t}» в веса: все положительные, в сумме 1.`,
    attn: `Выход attention для «${t}» - смесь векторов v трёх слов с весами из softmax.`,
    res1: `Residual: к исходному вектору «${t}» прибавляется выход attention.`,
    ln2: "Вторая LayerNorm нормирует h перед FFN, а сам h идёт в обход FFN ко второму residual.",
    ffn: "FFN: z = ĥ · W₁, ReLU обнуляет отрицательные z, затем второй слой W₂.",
    res2: "Второй residual: к h прибавляется выход FFN, получается выход блока y.",
    logits: "Выходной слой умножает y на столбцы W_vocab: по одному логиту на каждое слово словаря.",
    probs: `Softmax по логитам даёт вероятности следующего слова после «${t}», больше всех у «${win}».`,
  }[stage];
  if (!mask || STAGES.indexOf(stage) < STAGES.indexOf("scores")) return base;
  if (["scores", "softmax", "attn"].includes(stage)) {
    if (i === 0) return `С маской «${t}» видит только себя: будущие слова закрыты. ${GPT_MASK}`;
    if (i === TOKENS.length - 1) return `С маской «${t}» видит все слова до себя: будущих у него нет, маска ничего не меняет. ${GPT_MASK}`;
    return `С маской «${t}» видит «${TOKENS[0]}» и себя, а слово «${TOKENS[2]}» ещё впереди и закрыто. ${GPT_MASK}`;
  }
  if (same(chain(true)[MATRIX[stage]][i], chain(false)[MATRIX[stage]][i])) return base;
  return `${base} С маской числа другие, чем в уроке.`;
}
