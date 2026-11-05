// Логика демо «Генерация с ручками» (модуль 15, уроки 15.1 и 15.6). Записанные прогоны generate() урока 15.6 на
// модели урока 14.2 (data.js): промт «The cat was», 40 новых токенов, greedy и сэмплирование с T, top-k 5, top-p 0.9
// и разными seed. Для каждого шага - место выбранного токена и p в распределении, из которого тянули.
// Круг - первое повторённое окно из 4 токенов, как в m14/generate-steps. Чистые функции без DOM - сверяются с golden.json.

import { pText, tokenLabel } from "../../kit/format.js";
import { N_PROMPT, RUNS, VOCAB } from "./data.js";

export const TEMPS = [5, 7, 10, 15, 20];
export const FILTERS = ["none", "topk", "topp"];
export const SEEDS = [1, 2, 3, 4, 42];
const MAX_LEN = 4;
const FILTER_TEXT = { none: "без фильтра", topk: "top-k 5", topp: "top-p 0.9" };
const VIEW_GREEDY = [["greedy, запуск 1", "greedy:1"], ["greedy, запуск 2", "greedy:2"],
  ["T = 1, seed 1", "10|none|1"], ["T = 1, seed 2", "10|none|2"], ["T = 1, seed 4", "10|none|4"]];

export const runKey = (greedy, t10, filter, seed) => (greedy ? "greedy:1" : `${t10}|${filter}|${seed}`);
export const parseView = (text) => (text === "greedy" ? "greedy" : "");

function cycleOf(key) {
  const ids = RUNS[key].ids;
  const seen = new Map();
  for (let k = 0; k < ids.length - N_PROMPT; k++) {
    const win = ids.slice(0, N_PROMPT + k).slice(-MAX_LEN).join(",");
    if (seen.has(win)) return { k1: seen.get(win), k2: k };
    seen.set(win, k);
  }
  return null;
}

/** Круг по-настоящему: после первого повтора окна текст до конца повторяет себя с периодом k2 - k1.
 * У greedy это следует из повтора окна; у сэмплирования рулетка может свернуть, и круга нет. */
function periodic(key) {
  const c = cycleOf(key);
  if (!c) return null;
  const gen = RUNS[key].ids.slice(N_PROMPT);
  const period = c.k2 - c.k1;
  for (let g = c.k1; g < gen.length - period; g++) if (gen[g] !== gen[g + period]) return null;
  return c;
}

export function tokensView(key) {
  const r = RUNS[key];
  const c = periodic(key);
  return r.tokens.map((t, i) => {
    const g = i - N_PROMPT;
    return { label: tokenLabel(VOCAB[t]), text: VOCAB[t], kind: g < 0 ? "prompt" : "gen", rank: g < 0 ? null : r.ranks[g], inCycle: Boolean(c) && g >= c.k1 };
  });
}

function modeText(key) {
  if (key.startsWith("greedy")) return "greedy";
  const [t10, filter] = key.split("|");
  return `T = ${Number(t10) / 10}, ${FILTER_TEXT[filter]}`;
}

export function stepLine(key, g) {
  const r = RUNS[key];
  const x = VOCAB[r.tokens[N_PROMPT + g]];
  const lead = VOCAB[r.leader[g]];
  if (r.ranks[g] === 1) return `шаг ${g + 1}: выбран лидер «${tokenLabel(x)}», ${pEq(r.p[g])} (${modeText(key)})`;
  return `шаг ${g + 1}: выбран «${tokenLabel(x)}», №${group(r.ranks[g])} по вероятности, ${pEq(r.p[g])}; `
    + `лидер «${tokenLabel(lead)}», ${pEq(r.pl[g])} (${modeText(key)})`;
}

export const greedyView = () => VIEW_GREEDY.map(([title, key]) => ({ title, key }));

const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function pEq(p) {
  const t = pText(p);
  return t.startsWith("<") ? "p < 0.0001" : `p = ${t}`;
}
function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}

export function captionText(key) {
  const r = RUNS[key];
  const c = cycleOf(key);
  if (key.startsWith("greedy")) {
    let s = "Greedy на каждом шаге берёт лидера, поэтому запуск всегда один и тот же.";
    if (c) s += ` С шага ${c.k1 + 1} окно из 4 токенов повторяется, и текст идёт по кругу из ${c.k2 - c.k1} токенов.`;
    return s;
  }
  const off = r.ranks.filter((x) => x > 1).length;
  const far = Math.max(...r.ranks);
  const [, filter, seed] = key.split("|");
  let s = `${modeText(key)}, seed ${seed}: не лидер выбран в ${off} ${plural(off, "шаге", "шагах", "шагах")} из ${r.ranks.length}, самое дальнее место - №${group(far)}.`;
  if (periodic(key)) s += " Окно из 4 токенов повторилось, и текст пошёл по кругу.";
  else if (c) s += ` Окно из 4 токенов повторилось на шаге ${c.k2 + 1}, но рулетка с круга ушла.`;
  else s += " Окно из 4 токенов ни разу не повторилось.";
  if (filter === "none" && far > 100) s += " Без фильтра рулетка достаёт токены из хвоста, дальше 100-го места.";
  else if (filter === "topk") s += " Top-k 5 дальше 5-го места не пускает.";
  else if (filter === "topp" && far > 100) s += " Top-p 0.9 тянет только из ядра, но при такой T ядро разрослось, и хвост снова в рулетке: место дальше 100-го.";
  else if (filter === "topp") s += " Top-p 0.9 тянет только из ядра.";
  return s;
}

export function viewCaption() {
  const a = RUNS["greedy:1"].ids;
  const b = RUNS["greedy:2"].ids;
  const same = a.length === b.length && a.every((x, i) => x === b[i]) ? "совпадают до символа" : "различаются";
  const loop = cycleOf("greedy:1") ? " и уходят в круг" : "";
  const texts = new Set(VIEW_GREEDY.slice(2).map(([, k]) => RUNS[k].ids.join(",")));
  return `Два запуска greedy ${same}${loop}. Три запуска с T = 1 и разными seed дают ${texts.size} разных текста: `
    + "оранжевым отмечены шаги, где рулетка выбрала не лидера.";
}
