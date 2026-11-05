// Логика демо «Настоящее обучение по эпохам» (модуль 13, урок 13.9). Данные - записанный прогон train.py урока
// (data.js): пять прогонов по 100 эпох, loss «как печатает train.py», loss на новом тексте и прогноз для
// «The cat loved sour» на каждой эпохе. Подписи строятся по данным. Чистые функции без DOM - сверяются с golden.json.

import { fixed, pText, tokenLabel } from "../../kit/format.js";
import { LESSON, NEW_TEXT, NEXT, PROMPT, RUNS } from "./data.js";

export { LESSON, NEW_TEXT, PROMPT, RUNS, pText, tokenLabel };
export const EPOCHS = 100;
const run = (key) => RUNS.find((r) => r.key === key);
const l2 = (x) => fixed(x, 2);

function argmin(xs) {
  let best = 0;
  xs.forEach((x, i) => { if (x < xs[best]) best = i; });
  return best;
}

export function bestTest(key) {
  const r = run(key);
  const e = argmin(r.test);
  return { epoch: e, loss: r.test[e] };
}

export function bestTrain(key) {
  const r = run(key);
  const e = argmin(r.train);
  return { epoch: e + 1, loss: r.train[e] };
}

/** Эпоха первого крупного скачка loss на тексте урока (больше 0.1 за эпоху) или null. */
function firstJump(r) {
  for (let e = 1; e < r.train.length; e++) if (r.train[e] - r.train[e - 1] > 0.1) return e + 1;
  return null;
}

export function lossAt(key, epoch) {
  const r = run(key);
  return { train: epoch > 0 ? r.train[epoch - 1] : null, test: r.test[epoch] };
}

export function predRows(key, epoch) {
  const r = run(key);
  return PROMPT.map((_, pos) => ({
    context: PROMPT.slice(0, pos + 1).join(""), want: tokenLabel(NEXT[pos]), right: pText(r.right[epoch][pos]),
    top: r.top[epoch][pos].map((x) => ({ label: tokenLabel(x.t), p: pText(x.p), hit: x.t === NEXT[pos] })),
  }));
}

export function captionText(key, epoch) {
  const r = run(key);
  const lesson = run("3e-4");
  if (epoch === 0) {
    return `Эпоха 0: веса случайные. Loss на новом тексте ${l2(r.test[0])} ≈ ln 50 257: модель выбирает `
      + "наугад, и прогноз - случайные токены.";
  }
  const train = r.train[epoch - 1], test = r.test[epoch];
  if (key === "nozero") {
    const bt = bestTrain(key);
    if (epoch <= bt.epoch) {
      const faster = train < lesson.train[epoch - 1];
      let s = `Без zero_grad в .grad копятся градиенты всех прошлых шагов. Пока loss падает: на эпохе ${epoch} он `
        + `${l2(train)}${faster ? ", даже ниже, чем с zero_grad" : ""}.`;
      const bs = bestTest(key);
      if (epoch > bs.epoch && test - bs.loss >= 0.1) s += ` Но на новом тексте loss уже растёт: ${l2(test)}.`;
      return s;
    }
    return `Без zero_grad обучение ломается: лучший loss ${l2(bt.loss)} был на эпохе ${bt.epoch}, а к эпохе `
      + `${epoch} он вырос до ${l2(train)}. С zero_grad на этой эпохе - ${l2(lesson.train[epoch - 1])}.`;
  }
  const bt = bestTest(key);
  let s;
  if (epoch <= bt.epoch) {
    s = `Эпоха ${epoch}: loss на тексте урока ${l2(train)}, на новом тексте ${l2(test)}. Пока loss падает на обоих `
      + "текстах: модель учит общее, что есть и в новом тексте.";
  } else if (test - bt.loss < 0.1) {
    s = `Эпоха ${epoch}: loss на тексте урока ${l2(train)}, на новом тексте ${l2(test)} - около своего минимума `
      + `${l2(bt.loss)} (эпоха ${bt.epoch}).`;
  } else {
    s = `Эпоха ${epoch}: на тексте урока loss ${l2(train)}, а на новом тексте он вырос с ${l2(bt.loss)} `
      + `(эпоха ${bt.epoch}) до ${l2(test)}. Модель запоминает текст урока - это переобучение.`;
  }
  if (key === "3e-5") s += " Шаг маленький, и модель учится медленно.";
  const jump = firstJump(r);
  if (key === "3e-2" && jump !== null && epoch >= jump) s += " Шаг слишком большой: loss скачет.";
  return s;
}

export const parseRun = (text) => (RUNS.some((r) => r.key === text) ? text : "3e-4");
