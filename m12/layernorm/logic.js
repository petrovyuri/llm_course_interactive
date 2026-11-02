// Логика демо «LayerNorm своими руками» (модуль 12, урок 12.9). Вектор из 4 чисел до 2 знаков (целые сотые):
// μ, разности, квадраты и дисперсия - точные десятичные дроби, σ, x̂ и выход γ · x̂ + β - 3 знака
// (ROUND_HALF_UP); строка nn.LayerNorm (eps = 1e-5) - 4 знака, в ней видна разница из-за eps.
// Чистые функции без DOM - сверяются с golden.json.

export const PRESETS = [{ label: "кот", x: [0.5, 0.2, 0.8, 0.1] }, { label: "любил", x: [0.4, 0.9, 0.3, 0.7] },
  { label: "сметану", x: [0.6, 0.1, 0.5, 0.4] }, { label: "одинаковые", x: [0.5, 0.5, 0.5, 0.5] }];
export const PARSE = "Числа от -9 до 9, не больше двух знаков после точки.";
const EPS = 1e-5;

export function parseValue(text) {
  const t = String(text).trim().replace(",", ".");
  if (!/^-?(\d+(\.\d{0,2})?|\.\d{1,2})$/.test(t)) return null;
  const v = Number(t);
  return Math.abs(v) <= 9 ? v : null;
}

/** Целое в единицах 10^-k → строка с k знаками. */
function units(n, k) {
  const s = String(Math.abs(n)).padStart(k + 1, "0");
  return `${n < 0 ? "-" : ""}${s.slice(0, -k)}.${s.slice(-k)}`;
}

/** Точная десятичная запись без лишних нулей: exact(4000, 4) = «0.4», exact(0, 4) = «0». */
function exact(v, k) {
  const s = units(v, k).replace(/0+$/, "").replace(/\.$/, "");
  return s === "0" || s === "-0" ? "0" : s;
}

/** Сотые → μ в 1e-4, разности в 1e-4, квадраты в 1e-8, дисперсия в 1e-10. */
function ints(x) {
  const h = x.map((v) => Math.round(v * 100));
  const mu4 = h.reduce((s, v) => s + v, 0) * 25;
  const d = h.map((v) => v * 100 - mu4);
  const q = d.map((v) => v * v);
  return { h, mu4, d, q, var10: q.reduce((s, v) => s + v, 0) * 25 };
}

export function stats(x) {
  const { mu4, var10 } = ints(x);
  const v = var10 / 1e10;
  return { mu: mu4 / 1e4, var: v, sigma: Math.sqrt(v) };
}

/** x̂ = (x - μ) / σ без eps; null, если σ = 0. */
export function normed(x) {
  const { d, var10 } = ints(x);
  if (var10 === 0) return null;
  const sigma = Math.sqrt(var10 / 1e10);
  return d.map((v) => v / 1e4 / sigma);
}

export function output(x, gamma, beta) {
  const n = normed(x);
  return n === null ? null : n.map((v) => gamma * v + beta);
}

function th(v) {
  const q = Math.floor(Math.abs(v) * 1000 + 0.5);
  return v < 0 && q ? -q : q;
}

const vec3 = (vs) => `[${vs.map((v) => units(th(v), 3)).join(", ")}]`;

/** nn.LayerNorm: (x - μ) / √(σ² + eps) · γ + β. */
function code(x, gamma, beta) {
  const { d, var10 } = ints(x);
  const s = Math.sqrt(var10 / 1e10 + EPS);
  return d.map((v) => gamma * (v / 1e4 / s) + beta);
}

/** γ и β: «1», «0.5», отрицательное - в скобках. */
function num(v) {
  const s = exact(Math.round(v * 10), 1);
  return v < 0 ? `(${s})` : s;
}

export function lines(x, gamma, beta) {
  const { h, mu4, d, q, var10 } = ints(x);
  const out = [`μ = (${h.map((v) => exact(v, 2)).join(" + ")}) / 4 = ${exact(mu4, 4)}`,
    `x - μ = [${d.map((v) => exact(v, 4)).join(", ")}]`,
    `σ² = (${q.map((v) => exact(v, 8)).join(" + ")}) / 4 = ${exact(var10, 10)}`];
  if (var10 === 0) {
    out.push("σ = √0 = 0", "x̂ = (x - μ) / σ: σ = 0, а на ноль делить нельзя");
  } else {
    const s3 = th(Math.sqrt(var10 / 1e10));
    out.push(`σ = √${exact(var10, 10)} ${s3 * s3 * 10000 === var10 ? "=" : "≈"} ${units(s3, 3)}`,
      `x̂ = (x - μ) / σ ≈ ${vec3(normed(x))}`,
      `y = γ · x̂ + β = ${num(gamma)} · x̂ + ${num(beta)} ≈ ${vec3(output(x, gamma, beta))}`);
  }
  out.push(`nn.LayerNorm(4), eps = 0.00001: [${code(x, gamma, beta).map((v) => v.toFixed(4)).join(", ")}]`);
  return out;
}

export function captionText(x) {
  if (ints(x).var10 === 0) {
    return "Все числа одинаковые: σ = 0, и без eps делить не на что. nn.LayerNorm прибавляет к дисперсии "
      + "eps и выдаёт нули, а после γ и β получается вектор из одних β.";
  }
  return "У x̂ среднее 0 и σ = 1: LayerNorm убрал сдвиг и масштаб, а порядок чисел остался прежним.";
}
