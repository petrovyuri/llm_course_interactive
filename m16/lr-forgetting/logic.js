// Логика демо «Выучить и не забыть» (модуль 16, урок 16.6). Данные (data.js) - три дообучения от базы урока на
// 12 парах с lr 1e-5, 1e-4 и 1e-3 (seed 42, как finetune() в demo_improve.py): после каждой эпохи loss обучения,
// сколько ответов верны, loss на рассказе и продолжение «The cat was». Чистые функции без DOM - сверяются с golden.json.

import { fixed } from "../../kit/format.js";
import { ANSWERS, PAIRS, RUNS, VOCAB } from "./data.js";

const EOS = "<|endoftext|>";
export const LRS = ["1e-5", "1e-4", "1e-3"];
export const EPOCHS = 200;
export { PAIRS };

const matches = (got, want) => got.replace(/\.+$/, "") === want.replace(/\.+$/, "");

function timesWord(n) {
  const q = n % 10;
  return q >= 2 && q <= 4 && !(n % 100 >= 12 && n % 100 <= 14) ? `${n} раза` : `${n} раз`;
}

/** Сколько из 12 ответов начинаются верно: засчитанный ответ или верный ответ, после которого модель пишет дальше. */
export function start(lr, epoch) {
  return PAIRS.filter(([, want], qi) => {
    const a = ANSWERS[RUNS[lr].answers[epoch][qi]];
    return matches(a, want) || a.startsWith(want);
  }).length;
}

export function curve(lr, metric) {
  if (metric === "start") return RUNS[lr].acc.map((_, e) => [e, start(lr, e)]);
  return RUNS[lr][metric].map((v, e) => [e, v]).filter(([, v]) => v !== null);
}

export function firstFull(lr) {
  const e = RUNS[lr].acc.indexOf(12);
  return e === -1 ? null : e;
}

export function state(lr, epoch, qi) {
  const r = RUNS[lr];
  const answer = ANSWERS[r.answers[epoch][qi]];
  const strings = r.cont[epoch].map((t) => VOCAB[t]);
  return {
    acc: r.acc[epoch], start: start(lr, epoch), story: fixed(r.story[epoch], 3), train: r.train[epoch] === null ? "-" : fixed(r.train[epoch], 3),
    answer, right: matches(answer, PAIRS[qi][1]),
    continuation: strings.map((s) => (s === EOS ? "‹EOS›" : s)).join("").replaceAll("\n", "↵").trim(),
    eos: strings.includes(EOS),
  };
}

export function captionText(lr, epoch, qi) {
  const r = RUNS[lr];
  const st = state(lr, epoch, qi);
  const s0 = r.story[0];
  let s = `lr ${lr}, эпоха ${epoch}: верны ${st.acc} из 12 ответов, loss на рассказе ${st.story}`;
  if (epoch === 0) {
    s += " - это база урока до дообучения.";
  } else {
    const ratio = r.story[epoch] / s0;
    s += ratio < 1.5 ? ", почти как до дообучения."
      : `, в ${timesWord(Math.floor(ratio + 0.5))} больше, чем до дообучения (${fixed(s0, 3)}).`;
  }
  if (st.start > st.acc) s += ` Начало ответа верно у ${st.start} из 12, но вовремя ставить EOS модель ещё не научилась.`;
  const ff = firstFull(lr);
  if (ff === null) s += ` За ${EPOCHS} эпох засчитанных ответов нет: шаг слишком маленький.`;
  else if (epoch < ff) s += ` Все 12 ответов станут верны на эпохе ${ff}.`;
  else s += ` Все 12 ответов верны с эпохи ${ff}.`;
  if (st.eos) s += " На «The cat was» модель ставит EOS, как после ответа.";
  return s;
}
