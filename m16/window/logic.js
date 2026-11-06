// Логика демо «Что видит модель в окне» (модуль 16, урок 16.5). Данные (data.js) - 12 обучающих примеров токенами
// gpt2: промпт по шаблону урока, ответ и EOS, как в QADataset. Окно w - сколько последних токенов модель видит,
// когда предсказывает следующий. Чистые функции без DOM - сверяются с golden.json.

import { tokenLabel } from "../../kit/format.js";
import { EXAMPLES, QUESTIONS } from "./data.js";

const EOS = "<|endoftext|>";
const PARAMS_32 = 6534528;
const EMBED = 64;
export const MAX_W = 32;
export const PRESETS = [4, 32];
export { EXAMPLES, QUESTIONS };

const label = (t) => (t === EOS ? "EOS" : tokenLabel(t));
const promptKey = (e, w) => e.ids.slice(0, e.n).slice(-w).join(",");

/** 6534528 → «6 534 528». */
const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function tokensWord(n) {
  const q = n % 10;
  if (q === 1 && n % 100 !== 11) return `${n} токен`;
  if (q >= 2 && q <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return `${n} токена`;
  return `${n} токенов`;
}

export function windowView(ei, w, j) {
  const e = EXAMPLES[ei];
  const target = e.n + j;
  return e.tokens.map((t, i) => ({
    label: label(t), kind: i < e.n ? "prompt" : (t === EOS ? "eos" : "answer"), inWindow: target - w <= i && i < target, target: i === target,
  }));
}

export function contexts(w) {
  const keys = EXAMPLES.map((e) => promptKey(e, w));
  return EXAMPLES.map((e, i) => ({ labels: e.tokens.slice(0, e.n).slice(-w).map(label), cut: e.n > w, same: keys.indexOf(keys[i]) }));
}

export const distinct = (w) => new Set(EXAMPLES.map((e) => promptKey(e, w))).size;
/** Предсказывая EOS, модель видит весь пример, если окно вмещает все токены до EOS. */
export const fits = (ei, w) => EXAMPLES[ei].ids.length - 1 <= w;
export const params = (w) => PARAMS_32 - (MAX_W - w) * EMBED;

export function infoLine(ei, w) {
  const e = EXAMPLES[ei];
  return `различимы ${distinct(w)} из 12 · пример ${tokensWord(e.ids.length - 1)} + EOS: ${fits(ei, w) ? "помещается" : "не помещается"} · `
    + `позиции ${w}×${EMBED} = ${group(w * EMBED)} · параметров ${group(params(w))}`;
}

export function captionText(ei, w, j) {
  const e = EXAMPLES[ei];
  const k = distinct(w);
  let s;
  if (k === 1) {
    s = `При окне ${w} все 12 вопросов перед ответом выглядят одинаково: `
      + `${contexts(w)[0].labels.map((t) => `«${t}»`).join(", ")}. Ответить по-разному модель не может.`;
  } else {
    let first12 = 1;
    while (distinct(first12) !== 12) first12 += 1;
    if (k === 12) {
      s = `При окне ${w} перед ответом различимы все 12 вопросов${w === first12 ? ": меньшего окна не хватает." : "."}`;
    } else {
      s = `При окне ${w} перед ответом различимы ${k} вопросов из 12, все 12 - начиная с окна ${first12}.`;
    }
  }
  const pos = e.n + j;
  let seen = 0;
  for (let i = Math.max(0, pos - w); i < pos; i++) if (i < e.n) seen += 1;
  s += ` Для «${label(e.tokens[pos])}» модель видит ${seen} из ${e.n} токенов промпта.`;
  return s;
}
