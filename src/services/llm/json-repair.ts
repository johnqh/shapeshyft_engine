/**
 * @fileoverview Read model JSON that is nearly, but not quite, JSON.
 * @description A model writing a long answer slips now and then: a trailing
 * comma, a missing one, a bracket left open, a number like `0480`, an unescaped
 * quote inside a lyric. Function calling guarantees the SHAPE of the payload,
 * not that the text is valid — and a strict `JSON.parse` turns any of those
 * into a failed, billed call that then has to be asked for again.
 *
 * Measured on one product's history through this server: 1,045 of 1,205 failed
 * calls were parse failures, every one starting with a well-formed `{` and
 * breaking somewhere after it, at 500-1,700 tokens against a ceiling ten times
 * that. Most were "Unable to parse JSON string", "Expected ']'", "Invalid
 * number" and "Property name must be a string literal". At temperature 0 the
 * same prompt gives the same slip, so asking again does not help; reading the
 * answer tolerantly does, and costs nothing.
 *
 * What a real DeepSeek answer actually got wrong, from the first patched run
 * (6 repairs): five were TRAILING TEXT — a complete document, then the model
 * closed the outer object one brace early and carried on writing its notes,
 * which is what "Unable to parse JSON string" (55% of the history) was — and one
 * was arithmetic in a number, `[1920 - 480, 480, "F3", 106]`. Failed calls in
 * that run's styles fell from 6, 3 and 2 per song to one across three.
 *
 * The rule that keeps this safe: **strict parsing is tried first and is what
 * valid input gets**, byte for byte. The tolerant reader only runs on text
 * `JSON.parse` has already refused, and it reports every repair it made, so the
 * caller can log it. It never guesses at meaning — it closes what is open,
 * inserts what is missing between two values, and drops what is stray. A repaired
 * answer is still only as good as the shape the caller validates it against.
 */

/** One thing the reader had to fix to make the text parse. */
export type JsonRepair = { kind: string; at: number };

/** A parsed answer and what it took to parse it. */
export type ParsedModelJson = { value: unknown; repairs: JsonRepair[] };

const MISSING = Symbol("missing");
type Parsed = unknown | typeof MISSING;

/** More repairs than this is not a slip, it is not JSON. */
const MAX_REPAIRS = 40;
/** The longest unquoted key or value it will accept. */
const MAX_BAREWORD = 64;
/** Deeper than any real answer; a guard against runaway recursion. */
const MAX_DEPTH = 300;

const STRICT_NUMBER = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?$/;
const NUMBER_TOKEN = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/;
const WHITESPACE = /\s/;

class Reader {
  i = 0;
  readonly repairs: JsonRepair[] = [];

  constructor(private readonly s: string) {}

  private note(kind: string, at: number = this.i): void {
    if (this.repairs.length > MAX_REPAIRS) throw new TooBroken();
    this.repairs.push({ kind, at });
  }

  private skipSpace(): void {
    for (;;) {
      const c = this.s[this.i];
      if (c === undefined) return;
      if (WHITESPACE.test(c) || c === "\uFEFF") {
        this.i += 1;
      } else if (c === "/" && this.s[this.i + 1] === "/") {
        this.note("comment");
        while (this.i < this.s.length && this.s[this.i] !== "\n") this.i += 1;
      } else if (c === "/" && this.s[this.i + 1] === "*") {
        this.note("comment");
        const end = this.s.indexOf("*/", this.i + 2);
        this.i = end === -1 ? this.s.length : end + 2;
      } else {
        return;
      }
    }
  }

  /** The whole document: one object or array, anything after it ignored. */
  document(): unknown {
    this.skipSpace();
    const first = this.s[this.i];
    if (first !== "{" && first !== "[") throw new TooBroken();
    const value = this.value(0);
    if (value === MISSING) throw new TooBroken();
    this.skipSpace();
    if (this.i < this.s.length) this.note("trailing text");
    return value;
  }

  private value(depth: number): Parsed {
    if (depth > MAX_DEPTH) throw new TooBroken();
    this.skipSpace();
    const c = this.s[this.i];
    if (c === undefined) return MISSING;
    if (c === "{") return this.object(depth);
    if (c === "[") return this.array(depth);
    if (c === '"' || c === "'") return this.string(c);
    if ((c >= "0" && c <= "9") || c === "-" || c === "+" || c === ".") {
      return this.number();
    }
    return this.word();
  }

  private object(depth: number): Record<string, unknown> {
    this.i += 1; // {
    const out: Record<string, unknown> = {};
    let sawComma = false;
    for (;;) {
      this.skipSpace();
      const c = this.s[this.i];
      if (c === undefined) {
        this.note("unclosed object");
        return out;
      }
      if (c === "}") {
        if (sawComma) this.note("trailing comma");
        this.i += 1;
        return out;
      }
      if (c === "]") {
        // The array this object sits in is closing; this object lost its `}`.
        this.note("missing }");
        return out;
      }
      if (c === ",") {
        if (sawComma || Object.keys(out).length === 0) this.note("extra comma");
        sawComma = true;
        this.i += 1;
        continue;
      }
      sawComma = false;

      let key: string;
      if (c === '"' || c === "'") {
        key = this.string(c);
      } else {
        const start = this.i;
        while (this.i < this.s.length) {
          const k = this.s[this.i]!;
          if (k === ":" || k === "}" || k === "," || WHITESPACE.test(k)) break;
          this.i += 1;
        }
        key = this.s.slice(start, this.i);
        if (key.length > MAX_BAREWORD) throw new TooBroken();
        if (key === "") {
          // Something that cannot begin a key: drop it and carry on.
          this.note("stray character");
          this.i += 1;
          continue;
        }
        this.note("unquoted key", start);
      }
      this.skipSpace();
      if (this.s[this.i] === ":") this.i += 1;
      else this.note("missing colon");
      const v = this.value(depth + 1);
      if (v === MISSING) {
        this.note("unclosed object");
        return out;
      }
      out[key] = v;

      this.skipSpace();
      const next = this.s[this.i];
      if (next === ",") {
        sawComma = true;
        this.i += 1;
      } else if (next !== "}" && next !== "]" && next !== undefined) {
        this.note("missing comma");
      }
    }
  }

  private array(depth: number): unknown[] {
    this.i += 1; // [
    const out: unknown[] = [];
    let sawComma = false;
    for (;;) {
      this.skipSpace();
      const c = this.s[this.i];
      if (c === undefined) {
        this.note("unclosed array");
        return out;
      }
      if (c === "]") {
        if (sawComma) this.note("trailing comma");
        this.i += 1;
        return out;
      }
      if (c === "}") {
        // The object this array sits in is closing; this array lost its `]`.
        this.note("missing ]");
        return out;
      }
      if (c === ",") {
        if (sawComma || out.length === 0) this.note("extra comma");
        sawComma = true;
        this.i += 1;
        continue;
      }
      sawComma = false;

      const v = this.value(depth + 1);
      if (v === MISSING) {
        this.note("unclosed array");
        return out;
      }
      out.push(v);

      this.skipSpace();
      const next = this.s[this.i];
      if (next === ",") {
        sawComma = true;
        this.i += 1;
      } else if (next !== "]" && next !== "}" && next !== undefined) {
        this.note("missing comma");
      }
    }
  }

  private string(quote: string): string {
    const start = this.i;
    if (quote === "'") this.note("single-quoted string");
    this.i += 1;
    let out = "";
    let flaggedControl = false;
    for (;;) {
      const c = this.s[this.i];
      if (c === undefined) {
        this.note("unterminated string", start);
        return out;
      }
      if (c === "\\") {
        const e = this.s[this.i + 1];
        if (e === undefined) {
          this.i += 1;
          continue;
        }
        const simple: Record<string, string> = {
          n: "\n",
          t: "\t",
          r: "\r",
          b: "\b",
          f: "\f",
          "/": "/",
          '"': '"',
          "'": "'",
          "\\": "\\",
        };
        if (e in simple) {
          if (e === "'") this.note("bad escape");
          out += simple[e];
          this.i += 2;
        } else if (
          e === "u" &&
          /^[0-9a-fA-F]{4}$/.test(this.s.slice(this.i + 2, this.i + 6))
        ) {
          out += String.fromCharCode(
            parseInt(this.s.slice(this.i + 2, this.i + 6), 16)
          );
          this.i += 6;
        } else if (
          e === " " &&
          this.s[this.i + 2] !== undefined &&
          this.s[this.i + 2]! in simple
        ) {
          // `\ n`: a space slipped between the backslash and the escape letter.
          this.note("bad escape");
          out += simple[this.s[this.i + 2]!];
          this.i += 3;
        } else {
          // `\x`, `\d`: the backslash is the mistake; keep the character.
          this.note("bad escape");
          out += e;
          this.i += 2;
        }
        continue;
      }
      if (c === quote) {
        // A quote ends the string only if what follows could follow a string.
        let j = this.i + 1;
        while (j < this.s.length && WHITESPACE.test(this.s[j]!)) j += 1;
        const after = this.s[j];
        if (
          after === undefined ||
          after === "," ||
          after === "]" ||
          after === "}" ||
          after === ":"
        ) {
          this.i += 1;
          return out;
        }
        this.note("unescaped quote");
        out += c;
        this.i += 1;
        continue;
      }
      if (c < " ") {
        if (!flaggedControl) this.note("raw control character");
        flaggedControl = true;
      }
      out += c;
      this.i += 1;
    }
  }

  private number(): number | null {
    const at = this.i;
    const rest = this.s.slice(this.i, this.i + 64);
    const match = NUMBER_TOKEN.exec(rest);
    if (!match) {
      this.note("invalid number", at);
      this.i += 1;
      return null;
    }
    const token = match[0];
    this.i += token.length;
    if (!STRICT_NUMBER.test(token)) this.note("malformed number", at);
    const folded = this.arithmetic(Number(token), at);
    if (folded !== undefined) return folded;
    // `480ms`, `1.2.3`: letters or dots stuck to a number are junk, not part of it.
    let junk = false;
    while (this.i < this.s.length) {
      const k = this.s[this.i]!;
      if (k === "," || k === "]" || k === "}" || WHITESPACE.test(k)) break;
      junk = true;
      this.i += 1;
    }
    if (junk) this.note("junk after number", at);
    return Number(token);
  }

  /**
   * A number the model computed instead of writing: `1920 - 480`.
   *
   * Seen in a real answer — `[1920 - 480, 480, "F3", 106]` — where a model
   * working out a tick offset put the working in the payload. Read as separate
   * tokens it shifts every slot of the tuple after it, so it is evaluated:
   * `*` and `/` before `+` and `-`, left to right, numbers only.
   *
   * Only where the operator is spaced the same way on both sides (`1920 - 480`
   * or `1920-480`), never `1920 -480`, which is a missing comma before a
   * negative number and is read that way. Returns undefined when the text is
   * not an expression, and leaves the position where it was.
   */
  private arithmetic(first: number, at: number): number | null | undefined {
    const terms: number[] = [first];
    const ops: string[] = [];
    let j = this.i;
    for (;;) {
      let k = j;
      while (this.s[k] === " " || this.s[k] === "\t") k += 1;
      const op = this.s[k];
      if (op === undefined || !"+-*/".includes(op)) break;
      const spacedBefore = k > j;
      let m = k + 1;
      while (this.s[m] === " " || this.s[m] === "\t") m += 1;
      const spacedAfter = m > k + 1;
      if (spacedBefore !== spacedAfter) break;
      const operand = /^(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/.exec(
        this.s.slice(m, m + 64)
      );
      if (!operand) break;
      terms.push(Number(operand[0]));
      ops.push(op);
      j = m + operand[0].length;
    }
    if (ops.length === 0) return undefined;
    // * and / first, then + and -, each left to right.
    const values: number[] = [terms[0]!];
    const adds: string[] = [];
    ops.forEach((op, index) => {
      const next = terms[index + 1]!;
      if (op === "*") values[values.length - 1]! *= next;
      else if (op === "/") values[values.length - 1]! /= next;
      else {
        adds.push(op);
        values.push(next);
      }
    });
    let result = values[0]!;
    adds.forEach((op, index) => {
      result =
        op === "+" ? result + values[index + 1]! : result - values[index + 1]!;
    });
    this.note("arithmetic expression", at);
    this.i = j;
    return Number.isFinite(result) ? result : null;
  }

  private word(): unknown {
    const start = this.i;
    while (this.i < this.s.length) {
      const k = this.s[this.i]!;
      if (k === "," || k === "]" || k === "}" || k === "\n") break;
      this.i += 1;
    }
    const raw = this.s.slice(start, this.i).trim();
    if (raw === "") {
      // A closer where a value belonged (`{"a": }`): an empty value, and the
      // closer is left for the caller, so no position is skipped.
      this.note("missing value", start);
      return null;
    }
    // A pitch or a keyword is a few characters; a long unquoted run is prose
    // or noise, and reading it as a value would make garbage look like data.
    if (raw.length > MAX_BAREWORD) throw new TooBroken();
    const lower = raw.toLowerCase();
    if (raw === "true" || raw === "false" || raw === "null") {
      return raw === "true" ? true : raw === "false" ? false : null;
    }
    if (
      lower === "true" ||
      lower === "false" ||
      lower === "none" ||
      lower === "null"
    ) {
      this.note("non-JSON literal", start);
      return lower === "true" ? true : lower === "false" ? false : null;
    }
    if (
      raw === "NaN" ||
      raw === "Infinity" ||
      raw === "-Infinity" ||
      raw === "undefined"
    ) {
      this.note("non-JSON literal", start);
      return null;
    }
    this.note("unquoted string", start);
    return raw;
  }
}

class TooBroken extends Error {}

/**
 * Read `text` tolerantly, or return null when it is not recoverable JSON.
 *
 * Exported for tests. Callers want `parseModelJson`, which is strict first.
 */
export function lenientParse(text: string): ParsedModelJson | null {
  const reader = new Reader(text);
  try {
    const value = reader.document();
    return { value, repairs: reader.repairs };
  } catch (error) {
    if (error instanceof TooBroken) return null;
    throw error;
  }
}

/**
 * `JSON.parse`, then — only if that refuses — the tolerant reader.
 *
 * Valid JSON is returned exactly as `JSON.parse` returns it, with no repairs,
 * so putting this where `JSON.parse` was cannot change what a good answer
 * means. Throws the strict parser's own error when the text cannot be repaired,
 * so a caller's failure handling is unchanged.
 */
export function parseModelJson(text: string): ParsedModelJson {
  try {
    return { value: JSON.parse(text), repairs: [] };
  } catch (strict) {
    const lenient = lenientParse(text);
    if (lenient) return lenient;
    throw strict;
  }
}

/** A short window of `text` around the first repair, for a log line. */
export function windowAroundFirstRepair(
  text: string,
  repairs: readonly JsonRepair[],
  radius = 60
): string {
  const first = repairs[0];
  if (!first) return "";
  const from = Math.max(0, first.at - radius);
  const to = Math.min(text.length, first.at + radius);
  return `${from > 0 ? "…" : ""}${text.slice(from, to).replace(/\n/g, "\\n")}${to < text.length ? "…" : ""}`;
}

/** "trailing comma x2, missing comma" — what was repaired, once each. */
export function summarizeRepairs(repairs: readonly JsonRepair[]): string {
  const counts = new Map<string, number>();
  for (const { kind } of repairs) counts.set(kind, (counts.get(kind) ?? 0) + 1);
  return [...counts.entries()]
    .map(([kind, n]) => (n > 1 ? `${kind} x${n}` : kind))
    .join(", ");
}

/**
 * `parseModelJson` for an adapter: same result, and a log line when a repair
 * was needed.
 *
 * The line names what was fixed and shows the text around the first fix, which
 * is the only record there will be of what a model actually got wrong — the
 * failed answers were never kept, and a repair that works leaves no error to
 * read. `source` says which provider produced it.
 */
export function readModelJson(text: string, source: string): unknown {
  const { value, repairs } = parseModelJson(text);
  if (repairs.length > 0) {
    console.warn(
      `[llm] ${source}: repaired ${repairs.length} JSON fault(s) in the model's answer ` +
        `(${summarizeRepairs(repairs)}); first at char ${repairs[0]!.at} of ${text.length}: ` +
        windowAroundFirstRepair(text, repairs)
    );
  }
  return value;
}
