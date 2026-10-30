// Логика демо «Из прямых - ломаная» (модуль 11, урок «Функция активации ReLU»). Сеть - класс
// FeedForwardNetwork урока 8 с embed_dim = 1, ff_dim = n: одно число на входе, n нейронов с ReLU,
// одно на выходе. Веса: узлы k_i = -1 + 2i/n, W1 = 1, b1 = -k_i, W2 - приращения наклона отрезков,
// b2 = f(-1), так что выход - ломаная через значения цели в узлах. Без ReLU выход - прямая.
// Чистые функции без DOM - сверяются с golden.json.
import { fixed } from "../../kit/format.js";

export const TARGETS = [
  { key: "abs", label: "модуль |x|" },
  { key: "hill", label: "горка 1 - x²" },
  { key: "wave", label: "волна sin(πx)" },
];
export const XS = Array.from({ length: 201 }, (_, j) => -1 + j / 100);

export function target(key, x) {
  if (key === "abs") return Math.abs(x);
  if (key === "hill") return 1 - x * x;
  return Math.sin(Math.PI * x);
}

export function weights(key, n) {
  const knots = n ? Array.from({ length: n + 1 }, (_, i) => -1 + (2 * i) / n) : [];
  const slopes = Array.from({ length: n }, (_, i) => (target(key, knots[i + 1]) - target(key, knots[i])) / (2 / n));
  return {
    w1: Array(n).fill(1),
    b1: knots.slice(0, n).map((k) => -k),
    w2: slopes.map((s, i) => (i === 0 ? s : s - slopes[i - 1])),
    b2: target(key, -1),
  };
}

/** Выход сети в точке x: как forward класса урока 8 (без ReLU - два линейных слоя подряд). */
export function net(key, n, x, useRelu) {
  const w = weights(key, n);
  let y = 0;
  for (let i = 0; i < n; i += 1) {
    const h = x * w.w1[i] + w.b1[i];
    y += (useRelu ? Math.max(0, h) : h) * w.w2[i];
  }
  return y + w.b2;
}

export function curve(key, n, useRelu) {
  return XS.map((x) => net(key, n, x, useRelu));
}

/** Вклад нейрона i: w2 × ReLU(w1·x + b1). */
export function piece(key, n, i) {
  const w = weights(key, n);
  return XS.map((x) => w.w2[i] * Math.max(0, x * w.w1[i] + w.b1[i]));
}

export function maxError(key, n, useRelu) {
  return Math.max(...XS.map((x) => Math.abs(net(key, n, x, useRelu) - target(key, x))));
}

export function errorText(key, n, useRelu) {
  const e = maxError(key, n, useRelu);
  return e < 0.005 ? "совпадает с целью" : `наибольшее отклонение: ${fixed(e, 2)}`;
}

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}

export function captionText(key, n, useRelu) {
  if (!useRelu) return "Без ReLU два линейных слоя равны одному: при любом числе нейронов получается прямая.";
  if (n === 0) return "Без нейронов на выходе только bias - горизонтальная линия.";
  return `${n} ${plural(n, "нейрон", "нейрона", "нейронов")} - ${n} ${plural(n, "отрезок", "отрезка", "отрезков")} ломаной. `
    + "Чем больше нейронов, тем точнее ломаная повторяет цель.";
}
