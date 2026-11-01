// Логика демо «Каждый токен сам по себе» (модуль 11, урок «Зачем нужен Feed-Forward Network?»).
// Один и тот же вход X (выход Multi-Head Attention модуля 10) идёт в Attention с весами модуля 10 и в
// FFN с весами урока 11.3; к строке одного токена прибавлен сдвиг δ. Attention повторяет forward урока
// 10.12 (деление на √2 во float32, каузальная маска - прибавка -∞), FFN - класс урока 11.8 с bias = 0.
// Счёт в double; «не изменилась» - точное равенство строки с исходной.
// Чистые функции без DOM - сверяются с golden.json.
import { fixed } from "../../kit/format.js";

export const TOKENS = ["The", "cat", "loved"];
const X = [[7484, 8508, 7281, 8055], [7580, 8647, 7365, 8145], [7588, 8658, 7372, 8154]];   // десятитысячные
const W1 = [[5, 2, 1, 3, 4, 2, 1, 2], [3, 6, 2, 1, 2, 5, 3, 1], [1, 1, 7, 2, 3, 1, 6, 2], [2, 3, 1, 5, 1, 2, 1, 7]];
const W2 = [[5, 2, 1, 3], [3, 6, 2, 1], [1, 1, 7, 2], [2, 3, 1, 5], [4, 2, 3, 1], [2, 5, 1, 3], [1, 3, 6, 2], [3, 1, 2, 6]];
// Матрицы двух голов модуля 10 склеены по столбцам: [W⁰ | W¹], десятые
const WQ = [[5, 2, 4, 3], [3, 6, 2, 5], [1, 1, 3, 2], [2, 3, 5, 1]];
const WK = [[4, 3, 5, 2], [2, 5, 3, 4], [3, 2, 2, 3], [1, 4, 4, 2]];
const WV = [[6, 1, 7, 2], [1, 7, 2, 6], [2, 2, 1, 3], [3, 1, 2, 4]];
const WO = [[5, 2, 1, 3], [3, 6, 2, 1], [1, 1, 7, 2], [2, 3, 1, 5]];
const SCALE = Math.fround(Math.sqrt(2));   // torch.sqrt(torch.tensor(2.0)) во float32

const tenth = (M) => M.map((row) => row.map((v) => v / 10));
const [w1, w2, wq, wk, wv, wo] = [W1, W2, WQ, WK, WV, WO].map(tenth);

/** a [n×k] · b [k×m]; сумма слагаемых по порядку, как матричное умножение torch на малых матрицах. */
function matmul(a, b) {
  return a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
}

/** X + δ (в десятых) ко всем 4 числам строки t. */
export function inputWith(t, d10) {
  return X.map((row, i) => row.map((v) => (i === t ? v / 10000 + d10 / 10 : v / 10000)));
}

export function ffnOut(t, d10) {
  const hidden = matmul(inputWith(t, d10), w1).map((row) => row.map((v) => Math.max(0, v + 0)));
  return matmul(hidden, w2).map((row) => row.map((v) => v + 0));
}

export function attnOut(t, d10, mask) {
  const x = inputWith(t, d10);
  const [Q, K, V] = [wq, wk, wv].map((w) => matmul(x, w));
  const concat = x.map(() => []);
  for (const h of [0, 1]) {
    const cols = [2 * h, 2 * h + 1];
    const q = Q.map((r) => cols.map((c) => r[c]));
    const k = K.map((r) => cols.map((c) => r[c]));
    const v = V.map((r) => cols.map((c) => r[c]));
    const scores = q.map((qi, i) => k.map((kj, j) => (qi[0] * kj[0] + qi[1] * kj[1]) / SCALE + (mask && j > i ? -Infinity : 0)));
    const weights = scores.map((row) => {
      const max = Math.max(...row);
      const e = row.map((s) => Math.exp(s - max));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map((v2) => v2 / sum);
    });
    matmul(weights, v).forEach((z, i) => concat[i].push(...z));
  }
  return matmul(concat, wo);
}

function outOf(kind, t, d10, mask) {
  return kind === "ffn" ? ffnOut(t, d10) : attnOut(t, d10, mask);
}

/** Наибольшее |Δ| по каждой строке выхода против δ = 0. */
export function deltas(kind, t, d10, mask) {
  const now = outOf(kind, t, d10, mask);
  const base = outOf(kind, t, 0, mask);
  return now.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - base[i][j]))));
}

function changed(kind, t, d10, mask) {
  const now = outOf(kind, t, d10, mask);
  const base = outOf(kind, t, 0, mask);
  return now.map((row, i) => row.some((v, j) => v !== base[i][j]));
}

export function changeText(kind, t, d10, mask) {
  const d = deltas(kind, t, d10, mask);
  return changed(kind, t, d10, mask).map((c, i) => (c ? `изменилась на ${fixed(d[i], 4)}` : "не изменилась"));
}

export function captionText(t, d10, mask) {
  if (d10 === 0) return `Сдвиньте ползунок: δ прибавится ко всем 4 числам строки «${TOKENS[t]}».`;
  const rows = changed("attn", t, d10, mask).flatMap((c, i) => (c ? [`«${TOKENS[i]}»`] : []));
  const part = rows.length === 3 ? "поменялись все три строки"
    : rows.length === 2 ? `поменялись строки ${rows[0]} и ${rows[1]}` : `поменялась только строка ${rows[0]}`;
  return `Attention - совещание: изменили «${TOKENS[t]}», и ${part}. `
    + `FFN - самостоятельная работа: поменялась только строка «${TOKENS[t]}».`;
}
