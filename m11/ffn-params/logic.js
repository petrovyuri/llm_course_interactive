// Логика демо «Параметры FFN и Attention» (модуль 11, уроки «Архитектура», «Сравнение» и «Код»).
// FFN: W1 [d×ff] + W2 [ff×d] = 2 × d × ff (формула урока 7), в коде с bias ещё + ff + d, как
// get_num_parameters() урока 8. Attention - 4 × d², как MultiHeadAttention модуля 10. Итог на все
// блоки - по формуле урока, без bias. Чистые функции без DOM - сверяются с golden.json.
import { fixed, trimZeros } from "../../kit/format.js";

export const MAX_DIM = 12288;
export const PRESETS = [
  { key: "example", label: "пример 4/8", d: 4, k: 2, blocks: 1 },
  { key: "model", label: "наша модель", d: 16, k: 4, blocks: 1 },
  { key: "gpt2", label: "GPT-2", d: 768, k: 4, blocks: 12 },
  { key: "gpt3", label: "GPT-3", d: 12288, k: 4, blocks: 96 },
];
export const PARSE = "embed_dim - целое от 1 до 12288.";

/** Поле embed_dim → целое 1…12288 или null. */
export function parseDim(text) {
  const t = String(text).trim();
  if (!/^\d+$/.test(t)) return null;
  const v = Number(t);
  return v >= 1 && v <= MAX_DIM ? v : null;
}

/** Разряды через пробел: 1207959552 → «1 207 959 552». */
export const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}

/** «≈ 116 млрд», «≈ 4.7 млн», для меньших чисел - пусто. */
export function approx(n) {
  if (n >= 1e9) return `≈ ${trimZeros(fixed(n / 1e9, 1))} млрд`;
  if (n >= 1e6) return `≈ ${trimZeros(fixed(n / 1e6, 1))} млн`;
  return "";
}

const ffn = (d, k, bias = false) => 2 * d * k * d + (bias ? k * d + d : 0);
const attn = (d) => 4 * d * d;

export const ffnText = (d, k) => `FFN: 2 × ${d} × ${k * d} = ${group(ffn(d, k))}`;
export const biasText = (d, k) => `в коде с bias: ${group(ffn(d, k))} + ${k * d} + ${d} = ${group(ffn(d, k, true))}`;
export const attnText = (d) => `Attention: 4 × ${d}² = ${group(attn(d))}`;

/** Доля FFN в блоке (bias - как в коде урока 8). */
export function share(d, k, bias) {
  const f = ffn(d, k, bias);
  return f / (f + attn(d));
}

export const shareText = (d, k, bias) => `FFN - ${fixed(100 * share(d, k, bias), 1)}% параметров блока`;

export function totalText(d, k, blocks) {
  if (blocks === 1) return "";
  const part = (n) => group(n) + (approx(n) ? ` (${approx(n)})` : "");
  return `${blocks} ${plural(blocks, "блок", "блока", "блоков")}: FFN ${part(ffn(d, k) * blocks)}, Attention ${part(attn(d) * blocks)}`;
}

export function captionText(d, k) {
  if (k === 4) return "При ff_dim = 4 × embed_dim на FFN приходится две трети параметров блока.";
  return `ff_dim = ${k} × embed_dim: у FFN ${2 * k} × d², у Attention 4 × d².`;
}
