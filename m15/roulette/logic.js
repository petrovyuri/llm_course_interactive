// Логика демо «Рулетка по вероятностям» (модуль 15, урок 15.2). Сектор слова равен его вероятности из урока,
// генератор mulberry32 с фиксированным seed: u ∈ [0, 1) попадает в сектор по накопленной сумме (как torch.multinomial
// по смыслу). Серия из n прокрутов всегда считается с начала, поэтому «сначала» повторяет ту же серию.
// Чистые функции без DOM - сверяются с golden.json.

import { fixed } from "../../kit/format.js";

export const WORDS = ["кот", "любил", "сметану", "он", "был"];
export const PROBS = [0.153, 0.171, 0.360, 0.196, 0.120];   // урок 15.2, сумма ровно 1.000
export const SEED = 42;
export const MAX = 10000;

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(k) {
  const rng = mulberry32(SEED);
  return Array.from({ length: k }, () => rng());
}

export function sector(u) {
  let cum = 0;
  for (let i = 0; i < PROBS.length; i++) {
    cum += PROBS[i];
    if (u < cum) return i;
  }
  return PROBS.length - 1;
}

export function run(n) {
  const rng = mulberry32(SEED);
  const counts = [0, 0, 0, 0, 0];
  let last = null;
  let u = null;
  for (let s = 0; s < Math.min(n, MAX); s++) {
    u = rng();
    last = sector(u);
    counts[last] += 1;
  }
  return { counts, last, u };
}

export const tableRows = (n) => run(n).counts.map((c, i) => ({
  word: WORDS[i], p: fixed(PROBS[i], 3), count: c, share: n ? fixed(c / n, 3) : "-", w: n ? c / n : 0,
}));

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}

/** 10000 → «10 000». */
export const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function maxDev(n) {
  const devs = run(n).counts.map((c, i) => Math.abs(c / n - PROBS[i]));
  let w = 0;
  for (let j = 1; j < 5; j++) if (devs[j] > devs[w]) w = j;
  return [w, devs[w]];
}

export function captionText(n) {
  if (n === 0) {
    return "Рулетка ещё не крутилась. Сектор каждого слова равен его вероятности: у «сметану» 36% круга. Нажмите «крутить».";
  }
  const r = run(n);
  if (n === 1) {
    return `Выпало «${WORDS[r.last]}». Один прокрут о частотах ничего не говорит: `
      + "выпасть могло любое слово, просто «сметану» - чаще других.";
  }
  const c = r.counts[2];
  let s = `После ${group(n)} ${plural(n, "прокрута", "прокрутов", "прокрутов")} «сметану» выпала `
    + `${group(c)} ${plural(c, "раз", "раза", "раз")}: доля ${fixed(c / n, 3)}, вероятность 0.360.`;
  if (n === 100) s += " Урок пишет «примерно 36 раз из 100» - вот оно, это «примерно».";
  const [w] = maxDev(n);
  if (w === 2) s += " Среди всех слов это самая большая разница между долей и вероятностью.";
  else s += ` Самая большая разница между долей и вероятностью - у слова «${WORDS[w]}»: ${fixed(r.counts[w] / n, 3)} против ${fixed(PROBS[w], 3)}.`;
  if (n < MAX) s += ` Для сравнения: после ${group(MAX)} прокрутов самая большая разница - ${fixed(maxDev(MAX)[1], 3)}.`;
  else s += ` Это предел демо. Для сравнения: после 100 прокрутов разница была ${fixed(maxDev(100)[1], 3)}.`;
  return s;
}
