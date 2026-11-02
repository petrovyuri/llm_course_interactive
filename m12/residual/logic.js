// Логика демо «Зачем residual» (модуль 12, урок 12.3). Данные (data.js, считает golden_residual.py):
// стопка из N блоков TransformerBlock урока 12.20 (embed_dim 16, 2 головы, ff_dim 64) со случайными
// весами начальной инициализации, вход - 5 случайных токенов; режим «res» - как в уроке (x + attn, x + ffn),
// «plain» - те же подмодули без сложения. Показатели - |x| выхода и косинус выхода и входа.
// Чистые функции без DOM - сверяются с golden.json.

import { IN_NORMS, NS, RUNS } from "./data.js";

/** Показатели стопки из n блоков: |x| выхода и косинус выхода и входа. */
export function row(seed, n, mode) {
  return RUNS[seed][mode][NS.indexOf(n)];
}

export function normText(v) {
  return v.toFixed(v >= 1 ? 2 : 3);
}

export function cosText(c) {
  return c.toFixed(3);
}

export function blocksWord(n) {
  if (n % 10 === 1 && n % 100 !== 11) return "блок";
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return "блока";
  return "блоков";
}

export function captionText(seed, n) {
  const r = row(seed, n, "res");
  const p = row(seed, n, "plain");
  return `${n} ${blocksWord(n)}: с residual выход почти равен входу (косинус ${cosText(r.cos)}). Без residual `
    + `от входа остался шум (косинус ${cosText(p.cos)}), а длина вектора упала с ${normText(IN_NORMS[seed])} `
    + `до ${normText(p.norm)}.`;
}
