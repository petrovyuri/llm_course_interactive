// Логика демо «SGD и Adam: размер шага» (модуль 13, урок 13.6). Вытянутая долина f(w₁, w₂) = 0.1·w₁² + 2·w₂²,
// старт (-3.5, 1.5). SGD: w ← w - η · g. Adam - формулы torch.optim.Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8);
// на странице m̂ и v̂, как в уроке. SGD считается улетевшим при |w| > 100, дальше путь не продолжается.
// Чистые функции без DOM - сверяются с golden.json (эталон - torch.optim в float64).

import { fixed, trimZeros } from "../../kit/format.js";

export const START = [-3.5, 1.5];
export const LRS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 1];
export const MAX_STEPS = 200;
const FAR = 100;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

export const loss = (w) => 0.1 * w[0] ** 2 + 2 * w[1] ** 2;
export const grad = (w) => [0.2 * w[0], 4 * w[1]];
const far = (w) => Math.abs(w[0]) > FAR || Math.abs(w[1]) > FAR;

/** Шаги 0…n (или до вылета SGD): [{w, loss, g, m, v, mh, vh, delta}], у шага 0 - только w и loss. */
export function run(kind, eta, n) {
  let w = [...START];
  let m = [0, 0], v = [0, 0];
  const out = [{ w, loss: loss(w) }];
  for (let t = 1; t <= n; t++) {
    const g = grad(w);
    if (kind === "sgd") {
      const delta = g.map((x) => -eta * x);
      w = w.map((x, i) => x + delta[i]);
      out.push({ w, loss: loss(w), g, delta });
      if (far(w)) break;
      continue;
    }
    m = m.map((x, i) => x + (1 - B1) * (g[i] - x));
    v = v.map((x, i) => B2 * x + (1 - B2) * g[i] * g[i]);
    const bc1 = 1 - B1 ** t, bc2 = 1 - B2 ** t;
    const denom = v.map((x) => Math.sqrt(x) / Math.sqrt(bc2) + EPS);
    const w2 = w.map((x, i) => x - (eta / bc1) * m[i] / denom[i]);
    const delta = w2.map((x, i) => x - w[i]);
    w = w2;
    out.push({ w, loss: loss(w), g, m: [...m], v: [...v], mh: m.map((x) => x / bc1), vh: v.map((x) => x / bc2), delta });
  }
  return out;
}

export const path = (kind, eta, n) => run(kind, eta, n).map((s) => ({ w: s.w, loss: s.loss }));
export const diverged = (kind, eta, n) => far(run(kind, eta, n).at(-1).w);

function sig(x, digits = 4) {
  if (x === 0) return "0";
  if (Math.abs(x) < 1e-6) return "≈0";
  return trimZeros(x.toPrecision(digits));
}

/** Строки таблицы последнего шага: [{label, a, b}] - значения по w₁ и w₂. */
export function stepRows(kind, eta, n) {
  const r = run(kind, eta, n);
  const s = r.at(-1), t = r.length - 1;
  const row = (label, xs) => ({ label, a: sig(xs[0]), b: sig(xs[1]) });
  const rows = [];
  if (t > 0) {
    rows.push(row("g", s.g));
    if (kind === "adam") {
      rows.push(row("m", s.m), row("v", s.v), row(`m̂ = m / (1 - 0.9^${t})`, s.mh), row(`v̂ = v / (1 - 0.999^${t})`, s.vh),
        row("сдвиг -η·m̂ / (√v̂ + ε)", s.delta));
    } else {
      rows.push(row("сдвиг -η·g", s.delta));
    }
  }
  rows.push({ label: "w", a: fixed(s.w[0], 3), b: fixed(s.w[1], 3) });
  return rows;
}

export function captionText(eta, n) {
  if (n === 0) {
    return "Обе точки стоят на старте, loss 5.725. Нажмите «шаг» или «20 шагов»: SGD сдвигает вес на η · g, "
      + "а Adam - примерно на η по каждой оси.";
  }
  const laNum = run("adam", eta, n).at(-1).loss;
  const la = sig(laNum, 3);
  if (eta > 0.5) {
    if (diverged("sgd", eta, n)) {
      return "SGD разошёлся: при η > 0.5 каждый прыжок по крутой оси w₂ длиннее предыдущего, и точка улетела "
        + `за карту. Adam при том же η: loss ${la}.`;
    }
    return `SGD раскачивается: при η > 0.5 каждый прыжок по крутой оси w₂ длиннее предыдущего. Adam при том же η: loss ${la}.`;
  }
  const lsNum = run("sgd", eta, n).at(-1).loss;
  const ls = sig(lsNum, 3);
  if (ls === la) return `Шаг ${n}: loss у SGD и у Adam ${la}${la === "≈0" ? ": оба дошли до минимума." : ", одинаковый."}`;
  const head = `Шаг ${n}: loss у SGD ${ls}, у Adam ${la}.`;
  if (laNum < lsNum) return `${head} Adam впереди: его шаг по каждой оси около η, а у SGD шаг η · g, и на пологой оси w₁ он ползёт.`;
  return `${head} SGD впереди. Adam не обязан быть быстрее: на этой карте его плюс в том, что он не разлетается `
    + "ни при одном η из ползунка.";
}
