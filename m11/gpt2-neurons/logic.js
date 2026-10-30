// Логика демо «Нейроны FFN настоящей GPT-2» (модуль 11, урок «Архитектура Feed-Forward Network»).
// Активации скрытого слоя FFN GPT-2 после GELU (12 слоёв × 5 токенов × 3072 нейрона) на фразе
// «The cat loved sour cream» посчитаны в эталоне и упакованы по байту в data.js: код 0…17 -
// отрицательное -c/100, 18…117 - (c - 18)/100, 118…254 - 1 + (c - 118)/10, 255 - выброс из OUTLIERS.
// Все подписи и счётчики считаются по этим значениям. Чистые функции без DOM - сверяются с golden.json.
import { fixed } from "../../kit/format.js";
import { CODES, OUTLIERS, TOKENS } from "./data.js";

export { TOKENS };
export const LAYERS = 12;
export const WIDTH = 3072;

const BYTES = Uint8Array.from(atob(CODES), (ch) => ch.charCodeAt(0));
const EXTRA = new Map(OUTLIERS.map(([l, t, n, v]) => [`${l},${t},${n}`, v]));
const code = (l, t, n) => BYTES[(l * 5 + t) * WIDTH + n];

export function decode(c) {
  if (c <= 17) return -c / 100;
  if (c <= 117) return (c - 18) / 100;
  return 1 + (c - 118) / 10;
}

export function value(l, t, n) {
  const c = code(l, t, n);
  return c === 255 ? EXTRA.get(`${l},${t},${n}`) : decode(c);
}

/** Знак сохраняется и у значений, округлённых до -0.00. */
export function isNegative(l, t, n) {
  return code(l, t, n) <= 17;
}

/** Значения нейронов слоя l для токена t. */
export function row(l, t) {
  return Array.from({ length: WIDTH }, (_, n) => value(l, t, n));
}

/** Показ: меньше 1 - 2 знака, от 1 - 1 знак (такова точность упаковки). */
function num(v) {
  return v < 1 ? fixed(v, 2) : fixed(v, 1);
}

export function stats(l, t) {
  let strong = 0;
  let near = 0;
  let neg = 0;
  for (let n = 0; n < WIDTH; n += 1) {
    const v = value(l, t, n);
    if (v > 0.5) strong += 1;
    if (Math.abs(v) < 0.1) near += 1;
    if (isNegative(l, t, n)) neg += 1;
  }
  return { strong, near, neg };
}

export function statsText(l, t) {
  const s = stats(l, t);
  const pct = (k) => fixed((100 * k) / WIDTH, 1);
  return [`сильный отклик (> 0.5): ${s.strong} из ${WIDTH} (${pct(s.strong)}%)`,
    `около нуля (|v| < 0.1): ${s.near} из ${WIDTH} (${pct(s.near)}%)`,
    `отрицательных: ${s.neg} из ${WIDTH} (${pct(s.neg)}%)`];
}

/** Пять самых сильных нейронов: по убыванию значения, при равенстве - меньший номер. */
export function top5(l, t) {
  return row(l, t).map((v, n) => ({ n, v })).sort((a, b) => b.v - a.v || a.n - b.n).slice(0, 5);
}

export function topText(l, t) {
  return top5(l, t).map(({ n, v }) => `#${n}: ${num(v)}`);
}

export function neuronText(l, n) {
  return `Нейрон ${n}, слой ${l}: ${TOKENS.map((tok, t) => `${tok} ${num(value(l, t, n))}`).join(", ")}`;
}

export function captionText(l, t) {
  return `Сильно откликаются немногие нейроны: у «${TOKENS[t]}» в слое ${l} - ${stats(l, t).strong} из ${WIDTH}.`;
}
