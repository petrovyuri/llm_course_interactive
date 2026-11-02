// Логика демо «Как блоки уточняют прогноз» (модуль 12, урок 12.2). Данные - GPT-2 на фразе
// «The cat loved sour» (data.js, считает golden_logit_lens.py): после каждого блока к его выходу
// применяются финальная LayerNorm и выходной слой (logit lens), блок 0 - эмбеддинги, блок 12 - выход модели.
// Чистые функции без DOM - сверяются с golden.json.

import { CREAM, LAYERS, TOP } from "./data.js";

export { LAYERS };
const CREAM_TOKEN = " cream";

/** Пробел в начале токена - «·»: « sour» → «·sour», «d» → «d». */
export function tokenLabel(t) {
  const s = t.replaceAll("\n", "↵");
  return s.startsWith(" ") ? `·${s.slice(1)}` : s;
}

function fmtP(p) {
  if (p < 0.0001) return "< 0.0001";
  return p.toFixed(p < 0.01 ? 4 : 3);
}

/** Пять самых вероятных токенов после блока l. */
export function topRows(l) {
  return TOP[l].map((r) => ({ label: tokenLabel(r.t), p: r.p, text: fmtP(r.p) }));
}

export function creamText(l) {
  return `«${tokenLabel(CREAM_TOKEN)}»: место ${CREAM[l].rank}, ${fmtP(CREAM[l].p)}`;
}

const upFirst = (s) => s[0].toUpperCase() + s.slice(1);

export function captionText(l) {
  const first = TOP[l][0];
  const who = tokenLabel(first.t);
  const p = fmtP(first.p);
  const c = CREAM[l];
  const cream = `«${tokenLabel(CREAM_TOKEN)}» на ${c.rank}-м месте (${fmtP(c.p)})`;
  if (l === 0) return `До первого блока, на одних эмбеддингах, модель повторяет последнее слово: «${who}» (${p}).`;
  const where = l === LAYERS - 1 ? "После блока 12, на выходе модели" : `После блока ${l}`;
  if (first.t === " sour") return `${where}: первое место всё ещё у «${who}» (${p}), модель повторяет последнее слово.`;
  if (first.t === " grapes") {
    const tail = c.rank <= 3 ? ` ${upFirst(cream)}.` : "";
    return `${where}: первое место у «${who}» (${p}), модель вспомнила выражение sour grapes.${tail}`;
  }
  if (first.t === "d") return `${where}: первое место у «d» (${p}), с него начинается слово sourdough. ${upFirst(cream)}.`;
  return `${where}: первое место у «${who}» (${p}).`;
}
