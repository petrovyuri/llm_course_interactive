// Логика демо «Cross-entropy: цена ошибки» (модуль 13, уроки 13.4 и 13.8). Четыре токена с логитами,
// softmax и L = -ln p правильного токена; у «случайной модели» ещё rest токенов с логитом 0 (словарь GPT-2).
// Показ - 3 знака (toFixed); в строке L = -ln p вероятность - с наименьшей точностью, при которой
// калькулятор даёт тот же L. Пресет batch - две позиции примера модуля 12 с маской (data.js).
// Чистые функции без DOM - сверяются с golden.json.

import { fixed, pForLoss } from "../../kit/format.js";
import { BATCH } from "./data.js";

const TOKENS4 = ["токен 1", "токен 2", "токен 3", "токен 4"];
export const PRESETS = [
  { key: "lesson", label: "урок 13.4", labels: TOKENS4, logits: [2.0, 1.0, 0.1, -1.0], target: 0, rest: 0 },
  { key: "overview", label: "урок 13.2", labels: ["кот", "сметану", "молоко", "спал"], logits: [1.2, 0.4, 2.1, -0.3], target: 1, rest: 0 },
  { key: "random", label: "случайная модель", labels: TOKENS4, logits: [0, 0, 0, 0], target: 0, rest: 50253 },
  { key: "batch", label: "батч (13.8)" },
];
export { BATCH, pForLoss };
const SUB = "₁₂";

/** Разряды целой части через неразрывный пробел: «50257.000» → «50 257.000» (число не рвётся при переносе). */
export const group = (s) => String(s).replace(/\B(?=(\d{3})+(?!\d))/g, "\u00a0");

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}
const tokens = (n) => plural(n, "токен", "токена", "токенов");

export function probs(logits, rest) {
  const e = logits.map((z) => Math.exp(z));
  const s = e.reduce((a, v) => a + v, 0) + rest;
  return e.map((v) => v / s);
}

export const lossOf = (logits, rest, target) => -Math.log(probs(logits, rest)[target]);

export const pBar = (p) => (p >= 0.001 ? fixed(p, 3) : "< 0.001");
const pWord = (p) => (p >= 0.001 ? fixed(p, 3) : "меньше 0.001");
/** Строка с 3 знаками → целые тысячные. */
const th = (s) => Math.round(Number(s) * 1000);

const SCALE = 12;
const NBSP = String.fromCharCode(160);

/** Десятичная строка (разряды через неразрывный пробел) → BigInt в единицах 1e-12, без ошибок двоичной записи. */
function units(s) {
  const t = s.split(NBSP).join("");
  const neg = t.startsWith("-");
  const [a, b = ""] = t.replace("-", "").split(".");
  const v = BigInt(a || "0") * 10n ** BigInt(SCALE) + BigInt((b + "0".repeat(SCALE)).slice(0, SCALE));
  return neg ? -v : v;
}

/** Экспонента для строки расчёта: от 1 - 3 знака после точки с разрядами, меньше 1 - 3 значащие цифры
 * (e^-5 = 0.00674, а не 0.007: иначе деление на калькуляторе уходит от показанного p). */
const eText = (v) => (v >= 1 ? group(fixed(v, 3)) : v.toPrecision(3));

export function lines(st) {
  const { logits, rest, target: t } = st;
  const e = logits.map((z) => eText(Math.exp(z)));
  const zero = logits.every((z) => z === 0);
  const out = [`e^z ${zero ? "=" : "≈"} [${e.join(", ")}]`];
  if (rest) out.push(`ещё ${group(rest)} ${tokens(rest)}: ${group(rest)} · e^0 = ${group(rest)}`);
  const total = logits.reduce((a, z) => a + Math.exp(z), 0) + rest;
  const shown = group(fixed(total, 3));
  const one = 10n ** BigInt(SCALE);
  const rel = e.reduce((a, v) => a + units(v), 0n) + BigInt(rest) * one === units(shown) ? "=" : "≈";
  out.push(`Σ = ${[...e, ...(rest ? [group(rest)] : [])].join(" + ")} ${rel} ${shown}`);
  const p = probs(logits, rest)[t];
  const ps = pForLoss(p);
  const exact = units(e[t]) * one === units(ps) * units(shown);
  out.push(`p(«${st.labels[t]}») = ${e[t]} / ${shown} ${exact ? "=" : "≈"} ${ps}`);
  out.push(`L = -ln ${ps} ≈ ${fixed(-Math.log(p), 3)}`);
  return out;
}

/** «из 50 253 токенов», «из 21 токена». */
const tokensGen = (n) => (n % 10 === 1 && n % 100 !== 11 ? "токена" : "токенов");

export function captionText(st) {
  const { logits, rest, target: t, labels } = st;
  const p = probs(logits, rest);
  const L = fixed(-Math.log(p[t]), 3);
  const pt = pForLoss(p[t]);
  const n = logits.length + rest;
  const total = logits.reduce((a, z) => a + Math.exp(z), 0) + rest;
  if (logits.every((z) => z === logits[0]) && (!rest || logits[0] === 0)) {
    const s = `Все логиты равны, и вероятность делится поровну между ${group(n)} токенами: `
      + `p = 1 / ${group(n)}, L = ln ${group(n)} ≈ ${L}.`;
    return s + (rest ? " Так начинает необученная модель. В уроке 13.9 первая эпоха печатает 10.39: это среднее "
      + "за эпоху, и модель за неё уже немного учится." : "");
  }
  const tail = " Loss растёт, когда правильный ответ не главный кандидат.";
  const top = Math.max(...logits);
  if (rest && top < 0) {
    return `Каждый из остальных ${group(rest)} ${tokensGen(rest)} (логит 0) вероятнее любого из четырёх, `
      + `а правильный ответ «${labels[t]}» получил ${pt}: L = ${L}.${tail}`;
  }
  const leaders = logits.map((z, i) => (z === top ? i : -1)).filter((i) => i >= 0);
  const hiddenTie = Boolean(rest) && top === 0;
  if (leaders.includes(t)) {
    const head = leaders.length > 1 || hiddenTie ? `«${labels[t]}» делит первое место` : `На первом месте «${labels[t]}»`;
    let s = `${head}: p = ${pt}, L = ${L}. Чем ближе p к 1, тем ближе loss к 0.`;
    if (rest) s += ` Остальные ${group(rest)} ${tokens(rest)} вместе забирают ${pWord(rest / total)}.`;
    return s;
  }
  const names = [...leaders.map((i) => `«${labels[i]}»`), ...(hiddenTie ? [`остальные ${group(rest)} ${tokens(rest)}`] : [])];
  const who = names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} и ${names.at(-1)}`;
  const each = names.length > 1 ? " у каждого" : "";
  return `Больше всего модель верит в ${who} (${pWord(p[leaders[0]])}${each}), а правильный ответ «${labels[t]}» `
    + `получил ${pt}: L = ${L}.${tail}`;
}

export function batchRows(targets) {
  return targets.map((tg, i) => {
    const p = BATCH.p[i];
    return { context: BATCH.contexts[i], word: BATCH.vocab[tg], cells: p.map(pBar),
      line: `L${SUB[i]} = -ln ${pForLoss(p[tg])} ≈ ${fixed(-Math.log(p[tg]), 3)}` };
  });
}

const batchLosses = (targets) => targets.map((tg, i) => -Math.log(BATCH.p[i][tg]));

export function batchLine(targets) {
  const [a, b] = batchLosses(targets);
  const m = fixed((a + b) / 2, 3);
  const rel = th(fixed(a, 3)) + th(fixed(b, 3)) === 2 * th(m) ? "=" : "≈";
  return `L = (${fixed(a, 3)} + ${fixed(b, 3)}) / 2 ${rel} ${m}`;
}

export function batchCaption(targets) {
  const ls = batchLosses(targets);
  const i = ls[0] >= ls[1] ? 0 : 1;
  const p = BATCH.p[i][targets[i]];
  return `Loss батча - среднее по двум позициям. Хуже всего модель справилась после «${BATCH.contexts[i]}»: `
    + `правильное «${BATCH.vocab[targets[i]]}» получило ${pWord(p)}, L = ${fixed(ls[i], 3)}.`;
}

export const parsePreset = (text) => (PRESETS.some((p) => p.key === text) ? text : "lesson");
