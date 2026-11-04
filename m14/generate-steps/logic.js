// Логика демо «Генерация по шагам» (модуль 14, уроки 14.1, 14.3 и 14.4). Исправленный generate() урока 14.3
// по одному токену на модели урока 14.2 (данные - data.js): после k новых токенов модель видит окно из 4 последних
// токенов, таблица - распределение следующего шага, argmax подсвечен. Круг - первое повторённое окно: при greedy
// и окне в 4 токена одинаковое окно даёт одинаковое продолжение. Чистые функции без DOM - сверяются с golden.json.

import { pText, tokenLabel } from "../../kit/format.js";
import { MAX_LEN, N_STEPS, STEPS } from "./data.js";

export { N_STEPS };
export const DEFAULT = "barsik";
export const LOOP_KEY = "gaunt";
const find = (key) => STEPS.find((s) => s.key === key);

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}
const tokensWord = (n) => `${n} ${plural(n, "токен", "токена", "токенов")}`;

export const prompts = () => STEPS.map((s) => ({ key: s.key, text: s.prompt }));
export const parsePrompt = (text) => (STEPS.some((s) => s.key === text) ? text : DEFAULT);
export const parseView = (text) => (text === "loop" ? "loop" : "");

export function cycleOf(key) {
  const s = find(key);
  const full = [...s.prompt_ids, ...s.ids];
  const n = s.prompt_ids.length;
  const seen = new Map();
  for (let k = 0; k < full.length - n; k++) {
    const win = full.slice(0, n + k).slice(-MAX_LEN).join(",");
    if (seen.has(win)) return { k1: seen.get(win), k2: k, period: k - seen.get(win) };
    seen.set(win, k);
  }
  return null;
}

export function tokensView(key, k) {
  const s = find(key);
  const n = s.prompt_ids.length;
  const c = cycleOf(key);
  return s.tokens.slice(0, n + k).map((t, i) => ({
    label: tokenLabel(t), kind: i < n ? "prompt" : "gen", inWindow: i >= n + k - MAX_LEN,
    isNew: k > 0 && i === n + k - 1, inCycle: Boolean(c) && k >= c.k2 && i >= n + c.k1,
  }));
}

export function topRows(key, k) {
  if (k >= N_STEPS) return [];
  return find(key).top[k].map((x, j) => ({ label: tokenLabel(x.t), p: pText(x.p), chosen: j === 0 }));
}

const plain = (tokens) => tokens.join("").replaceAll(String.fromCharCode(10), "↵").trim();

function windowText(key, k) {
  const s = find(key);
  const n = s.prompt_ids.length;
  return plain(s.tokens.slice(Math.max(0, n + k - MAX_LEN), n + k));
}

export function captionText(key, k, view) {
  const s = find(key);
  const n = s.prompt_ids.length;
  const c = cycleOf(key);
  if (c && k >= c.k2) {
    let t = `Текущее окно «${windowText(key, c.k2)}» уже было перед шагом ${c.k1 + 1}. Argmax каждый раз выбирает `
      + `одно и то же, поэтому дальше текст идёт по кругу из ${tokensWord(c.period)}.`;
    if (view === "loop") {
      t += " Так greedy-генерация и зацикливается: на один промт она всегда идёт одним путём. "
        + "Лечит это сэмплирование из модуля 15.";
    }
    return t;
  }
  if (k >= N_STEPS) return "60 новых токенов - предел демо. Окно за это время ни разу не повторилось.";
  const nxt = s.top[k][0];
  let t;
  if (k === 0) {
    t = `Промт «${s.prompt}» - это ${tokensWord(n)}.`;
    if (n > MAX_LEN) {
      const cut = n - MAX_LEN === 1 ? "первый токен" : `первые ${tokensWord(n - MAX_LEN)}`;
      t += ` Модель видит только последние 4, а срез [-4:] отрезает ${cut}.`;
    }
    t += ` Сейчас модель выберет «${tokenLabel(nxt.t)}» (${pText(nxt.p)}).`;
  } else {
    t = `Шаг ${k + 1}: модель видит только «${windowText(key, k)}» и выберет «${tokenLabel(nxt.t)}» (${pText(nxt.p)}).`;
  }
  if (nxt.p < 0.3) t += " Уверенности мало, вариантов несколько, но argmax всё равно берёт лидера.";
  if (k === 0) t += " Нажмите «шаг».";
  return t;
}
