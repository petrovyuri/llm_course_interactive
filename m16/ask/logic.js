// Логика демо «Спросите обе модели» (модуль 16, уроки 16.1, 16.3 и 16.6). Записанные ответы моделей модуля 16
// (data.js): база и дообученная модель, greedy как answer() в demo_finetune.py. Для каждой тройки «вопрос, шаблон,
// модель» - один прогон без стопа на 20 токенов; ответ со стопом - его начало до первого EOS.
// Чистые функции без DOM - сверяются с golden.json.

import { pText, tokenLabel } from "../../kit/format.js";
// EXPECTED - правильные ответы на три формулировки каждого факта (FACTS из demo_improve.py)
import { EXPECTED, QUESTIONS, RUNS, TEMPLATES, VOCAB } from "./data.js";

const EOS = "<|endoftext|>";
const TEMPLATE_MAP = Object.fromEntries(TEMPLATES);
export const TEMPLATE_KEYS = TEMPLATES.map(([key]) => key);
const TEMPLATE_ON = { nonl: "Без переноса строки", qa: "С «Q:» и «A:»", lower: "Строчными буквами", bare: "Один вопрос без шаблона" };
const TEMPLATE_WITH = { nonl: "без переноса строки", qa: "с «Q:»/«A:»", lower: "со строчными буквами", bare: "с одним вопросом" };

export const parseView = (text) => (text === "compare" || text === "template" ? text : "full");

export const questionList = (view) => QUESTIONS.map(([q, group], index) => ({ index, q, group }))
  .filter((x) => view !== "template" || x.group === "train");

export function answerView(qi, template, model, stop) {
  const r = RUNS[`${qi}|${template}|${model}`];
  const strings = r.t.map((t) => VOCAB[t]);
  const first = strings.indexOf(EOS);
  const cut = first === -1 ? strings.length : first;
  const n = stop ? cut : strings.length;
  const tokens = strings.slice(0, n).map((s, i) => ({ label: s === EOS ? "EOS" : tokenLabel(s), p: r.p[i], eos: s === EOS }));
  return {
    tokens, text: strings.slice(0, n).join("").trim(), empty: stop && cut === 0,
    eosP: stop && cut < strings.length ? r.p[cut] : null,
  };
}

export function templateRows(qi) {
  const q = QUESTIONS[qi][0];
  return TEMPLATES.map(([key, t]) => {
    const a = answerView(qi, key, "chat", true);
    return { template: key, prompt: t.replace("{q}", q).replaceAll("\n", "↵"), text: a.text, empty: a.empty };
  });
}

const matches = (got, want) => got.replace(/\.+$/, "") === want.replace(/\.+$/, "");

function quoteStart(text, n = 6) {
  const words = text.replaceAll("\n", " ").split(/\s+/).filter(Boolean);
  return `«${words.slice(0, n).join(" ")}${words.length > n ? "…" : ""}»`;
}

function chatLine(a) {
  if (a.empty) return "Дообученная модель сразу выдала EOS: ответ пустой.";
  if (a.eosP === null) return `Дообученная модель ответила «${a.text}» и за 20 токенов не остановилась.`;
  return `Дообученная модель ответила «${a.text}» и остановилась: EOS с p = ${pText(a.eosP)}.`;
}

function groupLine(qi, a) {
  const [q, g] = QUESTIONS[qi];
  if (g === "train") return "Этот вопрос был среди 12 обучающих.";
  if (g === "offtopic") {
    if (a.empty || a.text.toLowerCase().includes("know")) return "";
    return "Ответа в рассказе нет, а говорить «не знаю» модель не умеет: её этому не учили.";
  }
  const want = EXPECTED[q];
  if (want === undefined) return "Такого вопроса в обучении не было.";
  if (matches(a.text, want)) return "Вопрос задан другими словами, и ответ верный.";
  if (Object.values(EXPECTED).includes(a.text)) {
    return `Вопрос задан другими словами, и модель взяла ответ от другого вопроса, правильный ответ «${want}».`;
  }
  return `Вопрос задан другими словами, правильный ответ «${want}».`;
}

export function captionText(view, qi, template, stop) {
  if (view === "template") {
    const rows = templateRows(qi);
    const same = rows.slice(1).filter((r) => r.text === rows[0].text).map((r) => TEMPLATE_WITH[r.template]);
    let s = `С шаблоном урока ответ «${rows[0].text}».`;
    for (const r of rows.slice(1)) {
      if (r.text !== rows[0].text) s += ` ${TEMPLATE_ON[r.template]}: ${r.empty ? "пустой ответ." : `«${r.text}».`}`;
    }
    if (same.length) s += ` Тот же ответ ${same.length === 1 ? same[0] : `${same.slice(0, -1).join(", ")} и ${same.at(-1)}`}.`;
    return s;
  }
  const t = view === "compare" ? "trained" : template;
  const base = answerView(qi, t, "base", true);
  const parts = [`База не отвечает, а продолжает рассказ: ${quoteStart(base.text)}.`];
  if (view === "full" && !stop) {
    const first = answerView(qi, t, "chat", true);
    const full = answerView(qi, t, "chat", false).text;
    parts.push(chatLine(first));
    if (full.includes(EOS)) {
      const after = full.slice(full.indexOf(EOS) + EOS.length).replaceAll(EOS, " ");
      if (after.trim()) parts.push(`Без стопа генерация идёт дальше EOS: ${quoteStart(after)}.`);
    }
  } else {
    parts.push(chatLine(answerView(qi, t, "chat", true)));
  }
  if (view === "full" && template !== "trained") {
    const same = answerView(qi, template, "chat", true).text === answerView(qi, "trained", "chat", true).text;
    parts.push(`Шаблон не тот, на котором учили, ${same ? "а ответ тот же." : "и ответ изменился."}`);
  }
  const line = groupLine(qi, answerView(qi, t, "chat", true));
  if (line) parts.push(line);
  return parts.join(" ");
}

export const promptText = (qi, template) => TEMPLATE_MAP[template].replace("{q}", QUESTIONS[qi][0]);
