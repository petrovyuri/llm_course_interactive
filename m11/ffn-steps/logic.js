// Логика демо «FFN по шагам» (модуль 11, уроки «Шаг 1» - «Сравнение»). Вход X - выход Multi-Head
// Attention из модуля 10, W1 [4×8] и W2 [8×4] - из урока 3, bias = 0 (сдвиг b₁ есть только на вкладке ReLU).
// Счёт точный и целый: X в десятитысячных, W в десятых, Intermediate и ReLU в стотысячных, Output в
// миллионных; показ - ROUND_HALF_UP до 4 знаков, как в уроках 3-7. Класс FeedForwardNetwork урока 8
// даёт те же 4 знака. Произведения в строках расчёта округлены каждое, сумма - точная.
// Чистые функции без DOM - сверяются с golden.json.

export const TOKENS = ["The", "cat", "loved"];
export const X = [[7484, 8508, 7281, 8055], [7580, 8647, 7365, 8145], [7588, 8658, 7372, 8154]];
export const W1 = [[5, 2, 1, 3, 4, 2, 1, 2], [3, 6, 2, 1, 2, 5, 3, 1], [1, 1, 7, 2, 3, 1, 6, 2], [2, 3, 1, 5, 1, 2, 1, 7]];
export const W2 = [[5, 2, 1, 3], [3, 6, 2, 1], [1, 1, 7, 2], [2, 3, 1, 5], [4, 2, 3, 1], [2, 5, 1, 3], [1, 3, 6, 2], [3, 1, 2, 6]];
export const STAGES = ["input", "linear1", "relu", "linear2", "result"];
const SUB = ["₀", "₁", "₂"];
const R4 = [0, 1, 2, 3];
const R8 = [0, 1, 2, 3, 4, 5, 6, 7];

/** num / den с округлением половины от нуля (den > 0). */
function halfUp(num, den) {
  const q = Math.floor((2 * Math.abs(num) + den) / (2 * den));
  return num < 0 ? -q : q;
}

/** Целое в единицах 10^-k → строка с k знаками: units(3448) = «0.3448», units(-9, 1) = «-0.9». */
export function units(n, k = 4) {
  const s = String(Math.abs(n)).padStart(k + 1, "0");
  return `${n < 0 ? "-" : ""}${s.slice(0, -k)}.${s.slice(-k)}`;
}

/** Целое в единицах 10^-k (k ≥ 4) → 4 знака, ROUND_HALF_UP: show(86335, 5) = «0.8634». */
export function show(n, k) {
  return units(halfUp(n, 10 ** (k - 4)), 4);
}

/** Intermediate = X · W1 (+ сдвиг b₁ в десятых ко всем числам), в стотысячных. */
export function inter(shift10 = 0) {
  return X.map((row) => R8.map((j) => R4.reduce((s, k) => s + row[k] * W1[k][j], 0) + shift10 * 10000));
}

export function relu(M) {
  return M.map((row) => row.map((v) => Math.max(0, v)));
}

/** Output = ReLU · W2, в миллионных. */
export function out(R) {
  return R.map((row) => R4.map((j) => R8.reduce((s, k) => s + row[k] * W2[k][j], 0)));
}

/** Матрица строками: k ≥ 4 - показ 4 знака, иначе как есть (веса - 1 знак). */
export function matrixText(M, k) {
  return M.map((row) => row.map((v) => (k >= 4 ? show(v, k) : units(v, k))));
}

export function interLine(i, j) {
  const terms = R4.map((k) => `${units(X[i][k], 4)} × ${units(W1[k][j], 1)}`).join(" + ");
  const prods = R4.map((k) => show(X[i][k] * W1[k][j], 5)).join(" + ");
  return `int${SUB[i]}[${j + 1}] = ${terms} = ${prods} = ${show(inter()[i][j], 5)}`;
}

export function outLine(i, j) {
  const R = relu(inter());
  const terms = R8.map((k) => `${show(R[i][k], 5)} × ${units(W2[k][j], 1)}`).join(" + ");
  const prods = R8.map((k) => show(R[i][k] * W2[k][j], 6)).join(" + ");
  return `out${SUB[i]}[${j + 1}] = ${terms} = ${prods} = ${show(out(R)[i][j], 6)}`;
}

/** Вкладка ReLU для токена i при сдвиге b₁: до и после ReLU, сколько обнулено, выход после W2. */
export function reluView(i, shift10) {
  const M = inter(shift10);
  return {
    pre: M[i].map((v) => show(v, 5)),
    post: M[i].map((v) => show(Math.max(0, v), 5)),
    zeros: M[i].filter((v) => v < 0).length,
    out: out(relu(M))[i].map((v) => show(v, 6)),
  };
}

export function zerosText(i, shift10) {
  const { zeros } = reluView(i, shift10);
  if (shift10 === 0) return "Отрицательных нет: ReLU ничего не обнулил.";
  if (zeros === 0) return `b₁ = ${units(shift10, 1)}: все 8 чисел ещё положительные, ReLU ничего не обнулил.`;
  return `b₁ = ${units(shift10, 1)}: обнулено ${zeros} из 8, выход «${TOKENS[i]}» стал другим.`;
}

export function captionText(stage, i) {
  const t = TOKENS[i];
  return {
    input: "На вход идёт выход Multi-Head Attention из прошлого модуля: 3 токена по 4 числа. W₁ расширяет 4 → 8, W₂ сжимает 8 → 4.",
    linear1: `Строка «${t}» умножается на каждый из 8 столбцов W₁: из 4 чисел получается 8.`,
    relu: "ReLU оставляет положительные числа и обнуляет отрицательные.",
    linear2: `Каждое число выхода «${t}» собрано из всех 8 чисел после ReLU.`,
    result: `Форма та же, [3×4], а числа другие. Строка выхода «${t}» посчитана только из строки входа «${t}».`,
  }[stage];
}
