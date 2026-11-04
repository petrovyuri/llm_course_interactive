// Логика демо «Как модель учится говорить» (модуль 14, урок 14.2). Записанный прогон train.py урока (data.js):
// loss после каждой эпохи, отметка «сохранена лучшая модель» и greedy-генерация 20 токенов для трёх промтов весами
// этой эпохи. Подпись строится по признакам генерации: случайные токены, одни точки, повтор окна из 4 токенов,
// фразы рассказа. Чистые функции без DOM - сверяются с golden.json.

import { fixed } from "../../kit/format.js";
import { BEST_EPOCH, EPOCHS } from "./data.js";

export { BEST_EPOCH };
export const LAST = EPOCHS.length - 1;
const MAX_LEN = 4;

const plain = (tokens) => tokens.join("").replaceAll(String.fromCharCode(10), " ↵ ").trim();

function cycle(full, n) {
  const seen = new Set();
  for (let k = 0; k < full.length - n; k++) {
    const win = full.slice(0, n + k).slice(-MAX_LEN).join(",");
    if (seen.has(win)) return true;
    seen.add(win);
  }
  return false;
}

/** Лучший loss до эпохи e включительно: [эпоха, loss]. */
function bestBefore(e) {
  let best = null;
  for (const x of EPOCHS.slice(1, e + 1)) if (best === null || x.loss < best[1]) best = [x.epoch, x.loss];
  return best;
}

export function lossText(e) {
  if (e === 0) return "эпоха 0: обучения ещё не было, веса случайные";
  return `Epoch ${e}: loss = ${fixed(EPOCHS[e].loss, 4)}`;
}

export function savedText(e) {
  if (e === 0) return "";
  const x = EPOCHS[e];
  if (x.saved) return `  -> Сохранена лучшая модель с потерей: ${fixed(x.loss, 4)}`;
  const [b, lb] = bestBefore(e - 1);
  return `  (train.py ничего не печатает: лучший loss ${fixed(lb, 4)} был на эпохе ${b})`;
}

function kindOf(e, i) {
  const g = EPOCHS[e].gens[i];
  const gen = g.tokens.slice(g.n_prompt);
  if (e === 0) return "random";
  if (gen.filter((t) => t === ".").length > gen.length / 2) return "dots";
  if (cycle(g.ids, g.n_prompt)) return "loop";
  return "story";
}

// Первая эпоха, где все промты продолжаются фразами рассказа: до неё повтор - «ещё не выучила», после - «снова».
const FIRST_STORY = EPOCHS.findIndex((x, e) => e > 0 && x.gens.every((_, i) => kindOf(e, i) === "story"));

export const genRows = (e) => EPOCHS[e].gens.map((g, i) => ({
  // продолжение - как есть, со своим пробелом в начале: «bed.» не должно стать «bed .»
  prompt: plain(g.tokens.slice(0, g.n_prompt)), text: g.tokens.slice(g.n_prompt).join("").replaceAll(String.fromCharCode(10), " ↵ "),
  kind: kindOf(e, i),
}));

export function captionText(e) {
  if (e === 0) return "Эпоха 0: веса случайные, и модель выдаёт случайные токены из всего словаря GPT-2.";
  const kinds = EPOCHS[e].gens.map((_, i) => kindOf(e, i));
  const loss = fixed(EPOCHS[e].loss, 4);
  let t;
  if (kinds.every((k) => k === "dots")) {
    t = `Эпоха ${e}, loss ${loss}. Точка - самый частый токен рассказа, и пока модель ставит её на каждом шаге.`;
  } else if (kinds.some((k) => k === "dots" || k === "loop")) {
    const m = kinds.filter((k) => k === "dots" || k === "loop").length;
    const who = m === kinds.length ? "окно из 4 токенов быстро повторяется" : `у ${m} из ${kinds.length} промтов окно из 4 токенов повторяется`;
    t = e < FIRST_STORY
      ? `Эпоха ${e}, loss ${loss}. Модель уже собирает слова из рассказа, но ${who}, и текст идёт по кругу.`
      : `Эпоха ${e}, loss ${loss}. Модель продолжает фразами из рассказа, но ${who}, и текст снова идёт по кругу.`;
  } else {
    t = `Эпоха ${e}, loss ${loss}. Модель продолжает фразами из рассказа: за 20 токенов окно ни разу не повторилось.`;
  }
  if (!EPOCHS[e].saved) {
    const [b, lb] = bestBefore(e - 1);
    t += ` Loss выше лучшего (${fixed(lb, 4)}, эпоха ${b}), поэтому best_model.pth не перезаписан.`;
  }
  return t;
}

export const chartPoints = () => EPOCHS.slice(1).map((x) => [x.epoch, x.loss, x.saved]);
