// Логика демо «Один шаг обучения на числах» (модуль 13, уроки 13.5 и 13.7). Восемь этапов урока 13.7
// на примере модуля 12 с каузальной маской: вход [кот, любил], цель [любил, сметану], y блока - data.js.
// Шаг SGD только по W_vocab: ∂L/∂logits = (p - one-hot) / 2, ∂L/∂W_vocab = yᵀ · ∂L/∂logits, W - η · G.
// Градиенты остальных параметров - якобиан блока (data.js) × ∂L/∂y. Показ - 3 знака, строки клеток -
// из показанных чисел, «=» только при точном равенстве. Чистые функции без DOM - сверяются с golden.json.

import { fixed, pForLoss } from "../../kit/format.js";
import { JAC, WV, Y } from "./data.js";

export { WV, Y };
export const STAGES = ["batch", "split", "forward", "loss", "zero", "backward", "step", "next"];
export const TOKENS = ["кот", "любил", "сметану"];
const CONTEXTS = ["кот", "кот любил"];
export const VOCAB = ["кот", "любил", "сметану", "он", "был"];
const TARGET = [1, 2];
const DIMS = ["1", "2", "3", "4"];
const SUB = "₁₂";
export const MAX_ROUNDS = 100;
const MARKS = [[0, 1], [1, 2]];

const f3 = (v) => fixed(v, 3);
const par = (s) => (s.startsWith("-") ? `(${s})` : s);
const th = (s) => Math.round(Number(s) * 1000);

function plural(n, one, few, many) {
  if (n % 10 === 1 && n % 100 !== 11) return one;
  if (n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14)) return few;
  return many;
}
/** «после 1 шага», «после 5 шагов». */
const stepsWord = (n) => `${n} ${plural(n, "шага", "шагов", "шагов")}`;

// ---------- счёт ----------

export function forward(wv) {
  const logits = Y.map((y) => VOCAB.map((_, v) => y.reduce((a, yd, d) => a + yd * wv[d][v], 0)));
  const p = logits.map((row) => {
    const e = row.map((z) => Math.exp(z));
    const s = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / s);
  });
  return { logits, p };
}

export function lossOf(p) {
  const each = [0, 1].map((t) => -Math.log(p[t][TARGET[t]]));
  return { each, mean: (each[0] + each[1]) / 2 };
}

export const gradLogits = (p) => [0, 1].map((t) => VOCAB.map((_, v) => (p[t][v] - (v === TARGET[t] ? 1 : 0)) / 2));
export const gradW = (gl) => DIMS.map((_, d) => VOCAB.map((_, v) => Y[0][d] * gl[0][v] + Y[1][d] * gl[1][v]));
export const sgdStep = (wv, g, eta) => wv.map((row, d) => row.map((w, v) => w - eta * g[d][v]));

export function run(eta, n) {
  let wv = WV.map((r) => [...r]);
  const out = [];
  for (let k = 0; k <= n; k++) {
    const f = forward(wv);
    out.push({ wv, loss: lossOf(f.p).mean });
    if (k < n) wv = sgdStep(wv, gradW(gradLogits(f.p)), eta);
  }
  return out;
}

export function otherGrads(wv) {
  const gl = gradLogits(forward(wv).p);
  const dldy = [0, 1].flatMap((t) => DIMS.map((_, d) => gl[t].reduce((a, g, v) => a + g * wv[d][v], 0)));
  return JAC.map((g) => {
    let best = 0;
    for (let j = 0; j < g.J[0].length; j++) {
      let s = 0;
      for (let i = 0; i < 8; i++) s += g.J[i][j] * dldy[i];
      best = Math.max(best, Math.abs(s));
    }
    return { name: g.name, shape: g.shape, max: best };
  });
}

const gtext = (v) => (v >= 1e-6 ? v.toPrecision(3) : "< 0.000001");
export const gradsList = (s) => otherGrads(s.wv).map((g) => `${g.name} ${gtext(g.max)}`);

// ---------- вид этапов ----------

function derived(s) {
  const f = forward(s.wv);
  const gl = gradLogits(f.p);
  const g = gradW(gl);
  const wv2 = sgdStep(s.wv, g, s.eta);
  const f2 = forward(wv2);
  return { f, L: lossOf(f.p), gl, g, wv2, f2, L2: lossOf(f2.p) };
}

const mat = (m) => m.map((r) => r.map(f3));
const tbl = (key, title, rows, cols, cells, pick = false, marks = []) => ({ key, title, rows, cols, cells, pick, marks });

export function tables(stage, s) {
  const d = derived(s);
  const k = s.round;
  const two = TOKENS.slice(0, 2);
  switch (stage) {
    case "batch": return [tbl("batch", "batch [1, 3]", ["0"], ["0", "1", "2"], [["кот (0)", "любил (1)", "сметану (2)"]])];
    case "split": return [tbl("input", "input_ids = batch[:, :-1]", ["0"], ["0", "1"], [["кот (0)", "любил (1)"]]),
      tbl("target", "target_ids = batch[:, 1:]", ["0"], ["0", "1"], [["любил (1)", "сметану (2)"]])];
    case "forward": return [tbl("y", "y - выход блока [2×4]", two, DIMS, mat(Y)),
      tbl("logits", "logits = y · W_vocab [2×5]", two, VOCAB, mat(d.f.logits), true, MARKS)];
    case "loss": return [tbl("p", "вероятности [2×5]", two, VOCAB, mat(d.f.p), false, MARKS)];
    case "zero": return s.prevGrad === null ? []
      : [tbl("grad", `.grad до zero_grad (круг ${k - 1}) [4×5]`, DIMS, VOCAB, mat(s.prevGrad))];
    case "backward": return [tbl("gl", "∂L/∂logits [2×5]", two, VOCAB, mat(d.gl), true, MARKS),
      tbl("gw", "∂L/∂W_vocab [4×5]", DIMS, VOCAB, mat(d.g), true)];
    case "step": return [tbl("wv", `W_vocab круга ${k} [4×5]`, DIMS, VOCAB, mat(s.wv)),
      tbl("wv2", "W_vocab после шага [4×5]", DIMS, VOCAB, mat(d.wv2), true)];
    default: return [tbl("p2", "вероятности после шага [2×5]", two, VOCAB, mat(d.f2.p), false, MARKS)];
  }
}

export const defaultCell = (stage) => ({ forward: ["logits", 0, 1], backward: ["gl", 0, 1], step: ["wv2", 0, 1] }[stage] ?? null);

/** «=» или «≈»: сумма произведений показанных чисел (миллионные) против показанного итога. */
function relProducts(pairs, result) {
  const total = pairs.reduce((a, [x, y]) => a + th(x) * th(y), 0);
  return total === th(result) * 1000 ? "=" : "≈";
}

export function cellLine(stage, s, key, i, j) {
  const d = derived(s);
  if (key === "logits") {
    const pairs = [0, 1, 2, 3].map((k) => [f3(Y[i][k]), f3(s.wv[k][j])]);
    const res = f3(d.f.logits[i][j]);
    const terms = pairs.map(([a, b]) => `${par(a)}·${par(b)}`).join(" + ");
    return `«${TOKENS[i]}» → «${VOCAB[j]}»: ${terms} ${relProducts(pairs, res)} ${res}`;
  }
  if (key === "gl") {
    const p = f3(d.f.p[i][j]);
    const oh = j === TARGET[i] ? 1 : 0;
    const res = f3(d.gl[i][j]);
    const rel = th(p) - 1000 * oh === 2 * th(res) ? "=" : "≈";
    return `«${TOKENS[i]}» → «${VOCAB[j]}»: (${p} - ${oh}) / 2 ${rel} ${res}`;
  }
  if (key === "gw") {
    const pairs = [0, 1].map((t) => [f3(Y[t][i]), f3(d.gl[t][j])]);
    const res = f3(d.g[i][j]);
    const terms = pairs.map(([a, b]) => `${par(a)}·${par(b)}`).join(" + ");
    return `∂L/∂W[${i + 1}][«${VOCAB[j]}»]: y₁[${i + 1}]·g₁ + y₂[${i + 1}]·g₂ = ${terms} ${relProducts(pairs, res)} ${res}`;
  }
  const w = f3(s.wv[i][j]), g = f3(d.g[i][j]), res = f3(d.wv2[i][j]);
  const rel = th(w) * 100 - Math.round(s.eta * 100) * th(g) === th(res) * 100 ? "=" : "≈";
  return `W[${i + 1}][«${VOCAB[j]}»]: ${w} - ${fixed(s.eta, 2)}·${par(g)} ${rel} ${res}`;
}

export function stageLines(stage, s) {
  const d = derived(s);
  const k = s.round;
  switch (stage) {
    case "batch": return ["batch = [[0, 1, 2]]: «кот», «любил», «сметану»", "форма [batch_size, seq_length + 1] = [1, 3]",
      ...(k > 1 ? ["в демо батч на каждом круге один и тот же"] : [])];
    case "split": return ["input_ids = batch[:, :-1] = [[0, 1]]", "target_ids = batch[:, 1:] = [[1, 2]]"];
    case "forward": return ["y - выход блока модуля 12 с каузальной маской",
      `W_vocab круга ${k}: ${k === 1 ? "веса урока 12.18" : `после ${stepsWord(k - 1)}`}`];
    case "loss": {
      const { L } = d;
      const out = [0, 1].map((t) => `после «${CONTEXTS[t]}» правильное «${VOCAB[TARGET[t]]}»: L${SUB[t]} = -ln `
        + `${pForLoss(d.f.p[t][TARGET[t]])} ≈ ${f3(L.each[t])}`);
      const [a, b, m] = [f3(L.each[0]), f3(L.each[1]), f3(L.mean)];
      return [...out, `L = (${a} + ${b}) / 2 ${th(a) + th(b) === 2 * th(m) ? "=" : "≈"} ${m}`];
    }
    case "zero": {
      if (s.prevGrad === null) return [".grad у W_vocab пока None: backward ещё не запускали", "optimizer.zero_grad() → .grad остаётся None"];
      const top = Math.max(...s.prevGrad.flat().map(Math.abs));
      return [`.grad хранит градиент круга ${k - 1}: max |g| = ${f3(top)}`,
        "optimizer.zero_grad() → .grad = None: старый градиент стёрт (в torch 2.x вместо нулей)"];
    }
    case "backward": return ["∂L/∂logits = (p - one-hot) / 2: делим на 2, потому что loss - среднее по двум позициям",
      "∂L/∂W_vocab = yᵀ · ∂L/∂logits"];
    case "step": return [`W_vocab ← W_vocab - η · ∂L/∂W_vocab, η = ${fixed(s.eta, 2)}`, "остальные веса в этом демо стоят на месте"];
    default: {
      const p = d.f.p, p2 = d.f2.p;
      return [`loss: ${f3(d.L.mean)} → ${f3(d.L2.mean)}`, `p(«любил» после «кот»): ${f3(p[0][1])} → ${f3(p2[0][1])}`,
        `p(«сметану» после «кот любил»): ${f3(p[1][2])} → ${f3(p2[1][2])}`];
    }
  }
}

export function captionText(stage, s) {
  const d = derived(s);
  const k = s.round;
  switch (stage) {
    case "batch":
      return k === 1 ? "Из DataLoader пришёл батч: одна последовательность из трёх токенов. Третий токен нужен только как цель для второго."
        : `Круг ${k}: батч тот же, а W_vocab уже другая - после ${stepsWord(k - 1)}.`;
    case "split":
      return "Сдвиг на один токен: на каждой позиции модель должна угадать следующий. Маска не даёт позиции "
        + "заглянуть вперёд, поэтому ответа она не видит.";
    case "forward": {
      const p = d.f.p[0];
      let top = 0;
      p.forEach((v, i) => { if (v > p[top]) top = i; });
      if (top === 1) return `После «кот» модель уже ставит «любил» первым: ${f3(p[1])}.`;
      return `После «кот» модель ставит первым «${VOCAB[top]}» (${f3(p[top])}), а правильное «любил» получает ${f3(p[1])}.`;
    }
    case "loss": return `Средний loss круга ${k}: ${f3(d.L.mean)}. Это одно число на весь батч, его и будем уменьшать.`;
    case "zero": {
      const tail = "Во что обходится пропуск zero_grad настоящей модели, видно в демо урока 13.9.";
      if (s.prevGrad === null) {
        return "Перед первым backward градиентов ещё нет, и стирать нечего. Но zero_grad вызывают на каждом круге: "
          + `без него со второго круга новый градиент сложился бы со старым. ${tail}`;
      }
      return `В .grad лежит градиент прошлого круга, zero_grad его стирает, иначе новый градиент сложился бы со старым. ${tail}`;
    }
    case "backward":
      return "У правильных слов градиент по logits отрицательный: шаг против градиента поднимет их logits. "
        + "До остальных весов блока ошибка тоже дошла, но шаг в этом демо делаем только по W_vocab.";
    case "step":
      return `Шаг против градиента: каждая клетка W_vocab сдвинулась на η · градиент, η = ${fixed(s.eta, 2)}. `
        + "Чем больше η, тем длиннее шаг.";
    default: {
      const L = d.L.mean, L2 = d.L2.mean, p = d.f.p[0][1], p2 = d.f2.p[0][1];
      if (f3(L) === f3(L2)) {
        return `После шага loss почти не изменился: ${f3(L2)}. Модель уже почти не ошибается, градиент стал маленьким, `
          + "и шаг тоже.";
      }
      return `После шага loss ${L2 < L ? "упал" : "вырос"} с ${f3(L)} до ${f3(L2)}, а вероятность «любил» после «кот» `
        + `${p2 > p ? "выросла" : "упала"} с ${f3(p)} до ${f3(p2)}.`;
    }
  }
}

export const parseStage = (text) => (STAGES.includes(text) ? text : "batch");
