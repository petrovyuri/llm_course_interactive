// Логика демо «Хвост и ядро на настоящей модели» (модуль 15, урок 15.6). Распределение модели урока 14.2 (data.js)
// для трёх окон по всем 50 257 токенам: 50 первых логитов точно, остальные - тысячные, сжатые разностями с числом
// повторов. Softmax со сдвигом на максимум, суммы - слева направо в порядке убывания (как эталон).
// Top-k и top-p - как filter_top_k и filter_top_p урока 15.6. Чистые функции без DOM - сверяются с golden.json.

import { fixed, pText, tokenLabel } from "../../kit/format.js";
import { WINDOWS } from "./data.js";

const N_SHOW = 10;
export const T10S = Array.from({ length: 30 }, (_, i) => i + 1);
export const KEYS = WINDOWS.map((w) => w.key);
const find = (key) => WINDOWS.find((w) => w.key === key);
export const N = WINDOWS[0].n;

export const parseWindow = (text) => (KEYS.includes(text) ? text : "prompt");
export const windowText = (key) => find(key).text;

const cache = new Map();
function decoded(key) {
  if (!cache.has(`d:${key}`)) {
    const w = find(key);
    const vals = w.top.map((x) => x.z);
    let v = null;
    for (let j = 0; j < w.tail.length; j += 2) {
      v = v === null ? w.tail[j] : v - w.tail[j];
      for (let c = 0; c < w.tail[j + 1]; c++) vals.push(v / 1000);
    }
    cache.set(`d:${key}`, Float64Array.from(vals).sort().reverse());
  }
  return cache.get(`d:${key}`);
}

// Распределение держим одно (последнее): 30 массивов по 50 257 чисел на окно - это десятки мегабайт на телефоне.
let last = { id: null, p: null };
function probs(key, t10) {
  const id = `p:${key}:${t10}`;
  if (last.id !== id) {
    const z = decoded(key);
    const m = (z[0] * 10) / t10;
    const e = new Float64Array(z.length);
    let total = 0;
    for (let i = 0; i < z.length; i++) {
      e[i] = Math.exp((z[i] * 10) / t10 - m);
      total += e[i];
    }
    for (let i = 0; i < z.length; i++) e[i] /= total;
    last = { id, p: e };
  }
  return last.p;
}

export const topRows = (key, t10) => find(key).top.slice(0, N_SHOW).map((x, j) => ({ label: tokenLabel(x.t), p: pText(probs(key, t10)[j]) }));

export function tailMass(key, t10) {
  const p = probs(key, t10);
  let total = 0;
  for (let i = N_SHOW; i < p.length; i++) total += p[i];
  return total;
}

export function filtered(key, t10, mode, k, p100) {
  const p = probs(key, t10);
  let count;
  if (mode === "topk") {
    const z = decoded(key);
    const threshold = z[k - 1];
    count = 0;
    for (const x of z) if (x >= threshold) count += 1;
  } else if (mode === "topp") {
    let cum = 0;
    count = 0;
    for (let j = 0; j < p.length; j++) {
      if (j > 0 && cum >= p100 / 100) break;
      cum += p[j];
      count += 1;
    }
    return { count, mass: cum };
  } else {
    count = N;
  }
  let mass = 0;
  for (let i = 0; i < count; i++) mass += p[i];
  return { count, mass };
}

export function coreCurve(key, p100) {
  const id = `c:${key}:${p100}`;
  if (!cache.has(id)) cache.set(id, T10S.map((t) => [t, filtered(key, t, "topp", 5, p100).count]));
  return cache.get(id).map((x) => [...x]);
}

const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}
const tokensWord = (n) => `${group(n)} ${plural(n, "токен", "токена", "токенов")}`;
const tokensGen = (n) => `${group(n)} ${plural(n, "токена", "токенов", "токенов")}`;
const weight = (p) => (pText(p).startsWith("<") ? "меньше 0.0001" : pText(p));

export function captionText(key, t10, mode, k, p100) {
  const w = find(key);
  const p = probs(key, t10);
  let s = `Окно «${w.text}», T = ${t10 / 10}: у лидера «${tokenLabel(w.top[0].t)}» ${pText(p[0])}.`;
  const f = filtered(key, t10, mode, k, p100);
  if (mode === "none") return `${s} Без фильтра в рулетке все ${group(N)} токенов, а хвост после первой десятки весит ${weight(tailMass(key, t10))}.`;
  if (mode === "topk") return `${s} Top-k = ${k}: в рулетке ${tokensWord(f.count)} при любой T.`;
  s += ` Top-p = ${p100 / 100}: в ядре ${tokensWord(f.count)}.`;
  const c1 = filtered(key, 10, "topp", k, p100).count;
  const c2 = filtered(key, 20, "topp", k, p100).count;
  if (c2 > c1) {
    if (t10 === 20) s += ` При T = 1 ядро было из ${tokensGen(c1)}: температура выравнивает распределение, и ядро растёт.`;
    else if (t10 === 10) s += ` При T = 2 оно вырастет до ${tokensGen(c2)}: температура выравнивает распределение.`;
    else s += ` При T = 1 ядро из ${tokensGen(c1)}, а при T = 2 уже из ${tokensGen(c2)}: температура выравнивает распределение, и ядро растёт.`;
  }
  return s;
}
