// Разбор и форматирование чисел так, как они записаны в уроках. Чистые функции
// без DOM. Python-зеркало - out/review/_guides/golden_io.py в проекте курса: по
// нему считаются строки эталонов, поэтому правила здесь и там должны совпадать.

/** Как Number.prototype.toFixed, но без «-0.00»: отрицательный ноль пишем без минуса. */
export function fixed(x, digits) {
  const s = x.toFixed(digits);
  return /^-[0.]+$/.test(s) ? s.slice(1) : s;
}

/** Убирает хвостовые нули дробной части: «4.60» → «4.6», «3.00» → «3». */
export function trimZeros(s) {
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

/** Число из урока: целое - как есть, дробное - до digits знаков без хвостовых нулей. */
export function num(x, digits = 2) {
  if (Number.isInteger(x)) return String(x);
  return trimZeros(fixed(x, digits));
}

/** Доля → процент по правилам урока 2 модуля 2: «73.1%», «0.0245%», «≈0%». */
export function percent(p) {
  const pct = p * 100;
  if (pct >= 0.1) return fixed(pct, 1) + "%";
  if (pct >= 0.001) return trimZeros(pct.toPrecision(3)) + "%";
  return "≈0%";
}

/** Экспонента для таблиц softmax: «7.39», «162754.79»; огромные и крошечные - «7.23e+86». */
export function expo(v) {
  if (v === 0) return "0";
  return v >= 1e6 || v < 0.005 ? v.toExponential(2) : fixed(v, 2);
}

/** Сумма произведений, как в уроке: «(1×3) + (2×4) = 3 + 8 = 11». */
export function sumOfProducts(xs, ys) {
  const products = xs.map((x, i) => x * ys[i]);
  let total = 0;
  for (const p of products) total += p;
  if (xs.length === 1) return `${num(xs[0])}×${num(ys[0])} = ${num(total)}`;
  const left = xs.map((x, i) => `(${num(x)}×${num(ys[i])})`).join(" + ");
  const middle = products.map((p, i) => (i > 0 && p < 0 ? `(${num(p)})` : num(p))).join(" + ");
  return `${left} = ${middle} = ${num(total)}`;
}

/** Число или вложенный список в записи Python: «[[19, 22], [43, 50]]». */
export function listStr(x) {
  return Array.isArray(x) ? "[" + x.map(listStr).join(", ") + "]" : num(x);
}

/**
 * Вероятность для строки «L = -ln p ≈ …» (модуль 13): наименьшая точность от 3 значащих после нулей,
 * при которой -ln от показанного числа округляется к тому же L, что и от точного: 0.2347 → «0.2347», а не «0.235».
 */
export function pForLoss(p) {
  const want = fixed(-Math.log(p), 3);
  const start = p >= 0.001 ? 3 : Math.floor(-Math.log10(p)) + 3;
  for (let k = start; k < start + 6; k++) {
    const s = fixed(p, k);
    if (Number(s) > 0 && fixed(-Math.log(Number(s)), 3) === want) return s;
  }
  return fixed(p, start + 5);
}

/** Текст поля ввода → число (запятая допустима) или null, если это не число или |x| > limit. */
export function parseNumber(text, limit) {
  const t = String(text).trim().replace(",", ".");
  if (!/^-?(\d+(\.\d*)?|\.\d+)$/.test(t)) return null;
  const v = Number(t);
  return Math.abs(v) <= limit ? v : null;
}
