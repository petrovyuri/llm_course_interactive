// Логика демо «Три набора данных» (модуль 16, урок 16.6). Данные (data.js) - ответы трёх режимов demo_improve.py
// (A - 12 пар, B - плюс вторая формулировка, C - плюс 4 отказа; seed 42, lr 1e-4, 200 эпох) на 42 вопроса.
// Отметки, как в demo_improve.py: верно - совпадение без точки на конце, отказ - «know» в ответе.
// Чистые функции без DOM - сверяются с golden.json.

import { ANSWERS, QUESTIONS } from "./data.js";

export const MODES = ["A", "B", "C"];
export const GROUP_KEYS = ["all", "train", "second", "rephrase", "offtopic", "refusal"];

const matches = (got, want) => got.replace(/\.+$/, "") === want.replace(/\.+$/, "");

export function verdict(qi, mode) {
  const got = ANSWERS[mode][qi];
  const want = QUESTIONS[qi][2];
  if (got.toLowerCase().includes("know")) return "dontknow";
  return want !== null && matches(got, want) ? "right" : "wrong";
}

function cell(qi, mode) {
  const v = verdict(qi, mode);
  return { text: ANSWERS[mode][qi], verdict: v, good: v === "right" || (v === "dontknow" && QUESTIONS[qi][2] === null) };
}

export function rows(group) {
  const out = [];
  QUESTIONS.forEach(([q, g, want], qi) => {
    if (group !== "all" && g !== group) return;
    const cost = verdict(qi, "C") === "dontknow" && want !== null;   // отказ, хотя ответ в рассказе есть
    out.push({ q, group: g, cells: Object.fromEntries(MODES.map((m) => [m, cell(qi, m)])), cost });
  });
  return out;
}

const count = (group, mode, v) => QUESTIONS.filter(([, g], qi) => g === group && verdict(qi, mode) === v).length;

export function summary() {
  const keys = [["train", "train", "right"], ["second", "second", "right"], ["rephrase", "rephrase", "right"], ["refusals", "offtopic", "dontknow"]];
  return Object.fromEntries(keys.map(([key, g, v]) => [key, Object.fromEntries(MODES.map((m) => [m, count(g, m, v)]))]));
}

export function captionText(group) {
  const s = summary();
  const costs = rows("all").filter((r) => r.cost);
  const othersWrong = costs.every((r) => r.cells.A.verdict === "wrong" && r.cells.B.verdict === "wrong");
  const costLine = (costs.length === 1 ? "Такой вопрос один" : `Таких вопросов ${costs.length}`)
    + (othersWrong ? `, и A с B на ${costs.length === 1 ? "нём" : "них"} тоже ошибаются.` : ".");
  const trio = (key, total) => MODES.map((m) => `${m} ${s[key][m]}/${total}`).join(", ");
  if (group === "all") {
    return `Итог как в таблице урока. Обучающие: ${trio("train", 12)}. Новый перефраз: ${trio("rephrase", 12)}. `
      + `Отказы не по теме: ${trio("refusals", 2)}. Оранжевая рамка: C говорит «не знаю», хотя ответ `
      + `есть в рассказе. ${costLine}`;
  }
  if (group === "train") return `Формулировки, на которых учились все три режима: ${trio("train", 12)}. Свою дюжину модель выучивает в любом варианте.`;
  if (group === "second") return `Вторая формулировка: у A её в обучении не было, B и C на ней учились. Верных ответов: ${trio("second", 12)}.`;
  if (group === "rephrase") {
    const lost = QUESTIONS.filter(([, g], qi) => g === "rephrase" && verdict(qi, "B") === "right" && verdict(qi, "C") !== "right").map(([q]) => q);
    let out = `Третьей формулировки не было в обучении ни у кого: ${trio("rephrase", 12)}.`;
    if (lost.length) out += ` B верно, а C нет: ${lost.map((q) => `«${q}»`).join(", ")}.`;
    return `${out} «Не знаю» C ответил на ${count("rephrase", "C", "dontknow")} из 12.`;
  }
  if (group === "offtopic") return `Вопросы не по теме, их нет ни в одном наборе. «Не знаю» отвечают: ${trio("refusals", 2)}; остальные ответы выдуманы.`;
  return `Эти четыре вопроса были в наборе C как примеры отказа. «Не знаю» отвечают: ${MODES.map((m) => `${m} ${count("refusal", m, "dontknow")}/4`).join(", ")}.`;
}
