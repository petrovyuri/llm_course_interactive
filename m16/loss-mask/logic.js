// Логика демо «Маска -100 по токенам» (модуль 16, урок 16.4). Данные (data.js) - позиции 12 примеров QADataset
// с ошибкой cross-entropy и вероятностью следующего токена у базы и у дообученной модели урока, и ответы без стопа
// моделей, дообученных с маской и без неё. Правило чисел - как в модуле 12: показ 3 знака, строка расчёта из
// показанных чисел, «=» только при точном равенстве. Чистые функции без DOM - сверяются с golden.json.

import { fixed, pText, tokenLabel } from "../../kit/format.js";
import { NOSTOP, PAIRS, ROWS, VOCAB } from "./data.js";

const EOS = "<|endoftext|>";
const MODEL_TEXT = { base: "до дообучения", chat: "после дообучения" };
export { PAIRS };

const counted = (r, masked) => r.label !== -100 || !masked;

export function rows(pi, model, masked) {
  const data = ROWS[model][pi];
  const first = data.findIndex((r) => r.label !== -100);
  return data.map((r, i) => {
    const on = counted(r, masked);
    const next = VOCAB[r.next];
    let what;
    if (!on) what = "читаем, не учимся";
    else if (next === EOS) what = "учим: дальше EOS, ответ закончен";
    else what = `${masked && i === first ? "здесь учим" : "учим"}: дальше «${tokenLabel(next)}»`;
    return {
      label: tokenLabel(VOCAB[r.t]), id: r.id, labelText: on ? String(r.nextId) : "-100", what,
      ce: fixed(r.ce, 3), p: pText(r.p), counted: on,
    };
  });
}

function loss(pi, model, masked) {
  const ces = ROWS[model][pi].filter((r) => counted(r, masked)).map((r) => r.ce);
  let s = 0;
  for (const x of ces) s += x;
  return s / ces.length;
}

/** «-1.234» → -1234 (целые тысячные показанного числа). */
function mil(s) {
  const [a, b] = s.replace("-", "").split(".");
  return (s.startsWith("-") ? -1 : 1) * (Number(a) * 1000 + Number(b));
}

export function lossLine(pi, model, masked) {
  const shown = ROWS[model][pi].filter((r) => counted(r, masked)).map((r) => fixed(r.ce, 3));
  const total = fixed(loss(pi, model, masked), 3);
  const s = shown.reduce((acc, x) => acc + mil(x), 0);
  const sign = s % shown.length === 0 && s / shown.length === mil(total) ? "=" : "≈";
  return `loss = (${shown.join(" + ")}) / ${shown.length} ${sign} ${total}`;
}

export function questionShare(pi, model) {
  let q = 0;
  let all = 0;
  for (const r of ROWS[model][pi]) {
    if (r.label === -100) q += r.ce;
    all += r.ce;
  }
  return q / all;
}

function split(idx) {
  const strings = idx.map((t) => VOCAB[t]);
  const first = strings.indexOf(EOS);
  const cut = first === -1 ? strings.length : first;
  const after = strings.slice(cut);
  const ans = after.findIndex((s) => s.trim() === "Answer");
  return {
    answer: strings.slice(0, cut).join("").trim(),
    after: after.map((s) => (s === EOS ? "‹EOS›" : s)).join("").replaceAll("\n", "↵").trim(),
    writesAnswer: ans !== -1,
    askFragment: ans !== -1 && after.slice(0, ans).some((s) => s.includes("?")),
  };
}

export const noMaskRows = () => PAIRS.map(([q], pi) => ({
  q, masked: split(NOSTOP.masked[pi]), unmasked: split(NOSTOP.unmasked[pi]),
}));

const matches = (got, want) => got.replace(/\.+$/, "") === want.replace(/\.+$/, "");

/** Доля в процентах; 99.5% и больше (но не 100%) - «больше 99%». */
function shareText(share) {
  const pct = Math.round(100 * share);
  return pct === 100 && share < 1 ? "больше 99%" : `${pct}%`;
}

export function captionText(tab, pi, model, masked) {
  if (tab === "nomask") {
    const rs = noMaskRows();
    const right = (k) => rs.filter((r, i) => matches(r[k].answer, PAIRS[i][1])).length;
    const writes = (k) => rs.filter((r) => r[k].writesAnswer).length;
    const asks = rs.filter((r) => r.unmasked.askFragment).length;
    let s = `Обе модели дообучены одинаково, разница только в маске. До EOS ответы верны у ${right("masked")} из 12 `
      + `с маской и у ${right("unmasked")} из 12 без неё. `
      + `Но без стопа модель без маски в ${writes("unmasked")} продолжениях из 12 сама дописывает слово «Answer» из разметки`;
    s += asks ? `, а в ${asks} из них перед ним ещё и обрывок вопроса со знаком «?». ` : ". ";
    s += writes("masked") === 0 ? "У модели с маской таких продолжений нет." : `У модели с маской таких продолжений ${writes("masked")}.`;
    return s;
  }
  const n = ROWS[model][pi].filter((r) => counted(r, masked)).length;
  const total = fixed(loss(pi, model, masked), 3);
  if (masked) {
    let s = `С маской loss ${MODEL_TEXT[model]} - среднее по ${n} позициям ответа: ${total}. Вопрос модель читает, но ошибка на нём в loss не входит.`;
    if (model === "chat") {
      const q = ROWS[model][pi].filter((r) => r.label === -100).map((r) => r.ce);
      let sum = 0;
      for (const x of q) sum += x;
      s += ` Ошибка на вопросе так и осталась большой, в среднем ${fixed(sum / q.length, 3)}: угадывать вопросы модель не училась.`;
    }
    return s;
  }
  return `Без маски loss ${MODEL_TEXT[model]} - среднее по всем ${n} позициям: ${total}. На вопрос приходится `
    + `${shareText(questionShare(pi, model))} суммы ошибок: большая часть обучения ушла бы на то, чтобы угадывать вопрос.`;
}
