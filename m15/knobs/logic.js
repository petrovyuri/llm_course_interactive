// Логика демо «Температура, top-k и top-p» (модуль 15, уроки 15.3-15.5). Учебный пример модуля 12: словарь из 5 слов
// и логиты урока 12.18. Логиты в тысячных, T в десятых, p в сотых: z/T показывается целочисленным округлением
// HALF_UP, экспонента - от показанного z/T, итоги (сумма, вероятности, накопленная сумма, масса ядра) - double.
// Порядок, как в уроке 15.4: температура, фильтр, softmax. Чистые функции без DOM - сверяются с golden.json.

import { fixed } from "../../kit/format.js";

export const WORDS = ["кот", "любил", "сметану", "он", "был"];
const Z = [85, 200, 944, 334, -154];
export const STAGES = ["temperature", "topk", "topp"];
export const DEFAULTS = {
  temperature: { t10: 5, k: 2, p100: 70 },
  topk: { t10: 10, k: 2, p100: 70 },
  topp: { t10: 10, k: 2, p100: 70 },
};

export const parseStage = (text) => (STAGES.includes(text) ? text : "temperature");

/** Целое число тысячных → «-0.077». */
function milStr(q) {
  const a = Math.abs(q);
  return `${q < 0 ? "-" : ""}${Math.floor(a / 1000)}.${String(a % 1000).padStart(3, "0")}`;
}

/** z/T в тысячных с округлением HALF_UP: z_mil / (t10 / 10) = 10·z_mil / t10. */
function ztMil(z, t10) {
  const q = Math.floor((20 * Math.abs(z) + t10) / (2 * t10));
  return z < 0 ? -q : q;
}

const f3 = (x) => fixed(x, 3);
function tokensWord(n) {
  const q = n % 10;
  return `${n} ${q === 1 && n % 100 !== 11 ? "токен" : q >= 2 && q <= 4 && !(n % 100 >= 12 && n % 100 <= 14) ? "токена" : "токенов"}`;
}
const toMil = (s) => (s.startsWith("-") ? -1 : 1) * Number(s.replace("-", "").replace(".", ""));
const andList = (words) => {
  const q = words.map((w) => `«${w}»`);
  return q.length === 1 ? q[0] : `${q.slice(0, -1).join(", ")} и ${q[q.length - 1]}`;
};

function calc(stage, t10, k, p100) {
  const eAll = Z.map((z) => Math.exp(z / (100 * t10)));
  let kept = [true, true, true, true, true];
  if (stage === "topk") {
    const threshold = [...Z].sort((a, b) => b - a)[k - 1];
    kept = Z.map((z) => z >= threshold);
  }
  const e = eAll.map((x, i) => (kept[i] ? x : 0));
  let total = 0;
  for (const x of e) total += x;
  let allTotal = 0;
  for (const x of eAll) allTotal += x;
  const pT = eAll.map((x) => x / allTotal);
  let p = e.map((x) => x / total);
  const order = [0, 1, 2, 3, 4].sort((a, b) => pT[b] - pT[a]);
  const cum = [0, 0, 0, 0, 0];
  let mass = 1;
  if (stage === "topp") {
    let run = 0;
    order.forEach((i, j) => {
      kept[i] = j === 0 || run < p100 / 100;
      run += pT[i];
      cum[i] = run;
    });
    mass = 0;
    for (const i of order) if (kept[i]) mass += pT[i];
    p = pT.map((x, i) => (kept[i] ? x / mass : 0));
  }
  return { kept, total, pT, p, order, cum, mass };
}

export function rows(stage, t10, k, p100) {
  const c = calc(stage, t10, k, p100);
  const ref = stage === "temperature" ? calc("temperature", 10, k, p100).p : c.pT;   // метка «p до»
  const idx = stage === "topp" ? c.order : [0, 1, 2, 3, 4];
  return idx.map((i) => {
    const q = ztMil(Z[i], t10);
    const shown = c.kept[i] || stage !== "topk";
    return {
      word: WORDS[i], z: milStr(Z[i]), zt: shown ? milStr(q) : "-inf", e: shown ? f3(Math.exp(q / 1000)) : "0",
      pT: f3(c.pT[i]), cum: stage === "topp" ? f3(c.cum[i]) : "", kept: c.kept[i], p: f3(c.p[i]), w: c.p[i], ref: ref[i],
    };
  });
}

export function sumLine(stage, t10, k, p100) {
  const c = calc(stage, t10, k, p100);
  if (stage === "topp") {
    const parts = c.order.filter((i) => c.kept[i]).map((i) => f3(c.pT[i]));
    const mass = f3(c.mass);
    const sign = parts.reduce((s, x) => s + toMil(x), 0) === toMil(mass) ? "=" : "≈";
    const line = parts.length > 1 ? `масса ядра: ${parts.join(" + ")} ${sign} ${mass}` : `масса ядра: ${mass}`;
    return `${line}; вероятности в ядре делим на ${mass}`;
  }
  const parts = [0, 1, 2, 3, 4].filter((i) => c.kept[i]).map((i) => f3(Math.exp(ztMil(Z[i], t10) / 1000)));
  const total = f3(c.total);
  if (parts.length === 1) return `сумма: ${total}`;
  const sign = parts.reduce((s, x) => s + toMil(x), 0) === toMil(total) ? "=" : "≈";
  return `сумма: ${parts.join(" + ")} ${sign} ${total}`;
}

export function captionText(stage, t10, k, p100) {
  const c = calc(stage, t10, k, p100);
  const t = String(t10 / 10);
  const lead = c.order[0];
  if (stage === "temperature") {
    const p1 = calc("temperature", 10, k, p100).p;
    let s;
    if (t10 === 10) {
      s = `T = 1: обычный softmax, вероятности как в модуле 12. У лидера «${WORDS[lead]}» ${f3(c.p[lead])}.`;
    } else if (t10 < 10) {
      s = `T = ${t} < 1: различия между логитами усиливаются, распределение острее. `
        + `У лидера «${WORDS[lead]}» ${f3(c.p[lead])} вместо ${f3(p1[lead])}.`;
    } else {
      const last = c.order[4];
      s = `T = ${t} > 1: различия сглаживаются, распределение ровнее. У лидера «${WORDS[lead]}» `
        + `${f3(c.p[lead])} вместо ${f3(p1[lead])}, у слова «${WORDS[last]}» ${f3(c.p[last])} вместо ${f3(p1[last])}.`;
    }
    if (c.p[lead] >= 0.99) s += " Почти вся вероятность у лидера: при T → 0 сэмплирование превращается в greedy.";
    if (Math.max(...c.p) - Math.min(...c.p) < 0.1) s += " Распределение близко к равномерному: при очень большой T выбор почти случаен.";
    return s;
  }
  if (stage === "topk") {
    const alive = WORDS.filter((_, i) => c.kept[i]);
    if (alive.length === 1) return `k = 1: остаётся один лидер «${alive[0]}» с вероятностью 1 - это снова greedy.`;
    if (alive.length === 5) return `k = 5 - весь словарь: фильтра нет, вероятности те же, что после температуры T = ${t}.`;
    let kept = 0;
    for (let i = 0; i < 5; i++) if (c.kept[i]) kept += c.pT[i];
    const given = f3(1 - kept) === "0.000" ? "меньше 0.001" : f3(1 - kept);
    return `k = ${k}: в рулетке ${andList(alive)}, у остальных -inf и вероятность 0. `
      + `Выбывшие отдали ${given} массы, а соотношение выживших не изменилось.`;
  }
  const core = c.order.filter((i) => c.kept[i]);
  const pp = String(p100 / 100);
  let s;
  if (core.length === 1) {
    s = `p = ${pp}: у лидера «${WORDS[lead]}» уже ${f3(c.pT[lead])} ≥ ${pp}, в ядре один лидер - это greedy.`;
  } else if (core.length === 5) {
    s = `p = ${pp}: в ядро вошли все 5 токенов, фильтра нет.`;
  } else {
    s = `p = ${pp}: ядро из ${core.length} токенов (${andList(core.map((i) => WORDS[i]))}), их масса ${f3(c.mass)}. `
      + `Остальные получают -inf, вероятности в ядре делим на ${f3(c.mass)}.`;
  }
  if (t10 !== 10) {
    const m1 = calc("topp", 10, k, p100).kept.filter(Boolean).length;
    if (m1 !== core.length) s += ` При T = 1 в ядре было бы ${tokensWord(m1)}: чем выше T, тем ровнее распределение и тем шире ядро.`;
  }
  return s;
}
