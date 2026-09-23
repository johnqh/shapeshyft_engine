import { describe, expect, it } from "vitest";
import {
  lenientParse,
  parseModelJson,
  summarizeRepairs,
  windowAroundFirstRepair,
} from "../../src/services/llm/json-repair.js";

const kinds = (text: string): string[] =>
  (lenientParse(text)?.repairs ?? []).map(r => r.kind);
const read = (text: string): unknown => parseModelJson(text).value;

/*
 * The slips a model makes in a long answer, each of which used to be a failed,
 * billed call. Every input below is refused by JSON.parse.
 */
describe("parseModelJson: valid JSON is returned exactly as JSON.parse returns it", () => {
  const valid = [
    "{}",
    "[]",
    '{"a":1,"b":[1,2,3],"c":{"d":null,"e":true,"f":false}}',
    '[[0,480,"C4",90],[480,240,"D4",88,"a"],[720,240,"E4",90,"",{"d":"ff"}]]',
    '{"s":"line\\nbreak \\"quoted\\" \\u00e9 tab\\t back\\\\slash"}',
    '{"n":[0,-1,1.5,-0.25,1e3,2E-2,123456789012345]}',
    '  \n {"padded": [ 1 , 2 ] } \n',
    '{"plan":{"ks":"0M","ts":"4/4","tempo":120},"tracks":[{"measures":[{"i":0,"voices":[{"events":[]}]}]}],"warnings":[]}',
  ];

  it("never repairs and never alters a valid document", () => {
    for (const text of valid) {
      const parsed = parseModelJson(text);
      expect(parsed.repairs, text).toEqual([]);
      expect(parsed.value, text).toEqual(JSON.parse(text));
    }
  });

  it("would read valid JSON identically even if the tolerant reader ran on it", () => {
    for (const text of valid) {
      const lenient = lenientParse(text);
      expect(lenient?.repairs, text).toEqual([]);
      expect(lenient?.value, text).toEqual(JSON.parse(text));
    }
  });
});

describe("parseModelJson: commas", () => {
  it("drops a trailing comma in an array and in an object", () => {
    expect(read("[1,2,3,]")).toEqual([1, 2, 3]);
    expect(read('{"a":1,"b":2,}')).toEqual({ a: 1, b: 2 });
    expect(kinds("[1,2,]")).toContain("trailing comma");
  });

  it("inserts a comma missing between two values", () => {
    expect(read('[0, 480 "D3", 100]')).toEqual([0, 480, "D3", 100]);
    expect(read('{"a": 1 "b": 2}')).toEqual({ a: 1, b: 2 });
  });

  it("inserts the comma missing between two event tuples", () => {
    expect(read('{"events":[[0,480,"C4",90] [480,480,"D4",88]]}')).toEqual({
      events: [
        [0, 480, "C4", 90],
        [480, 480, "D4", 88],
      ],
    });
  });

  it("skips a doubled comma", () => {
    expect(read("[1,,2]")).toEqual([1, 2]);
    expect(kinds("[1,,2]")).toContain("extra comma");
  });
});

describe("parseModelJson: brackets", () => {
  it("closes what the text left open", () => {
    expect(read('{"a":[1,2')).toEqual({ a: [1, 2] });
    expect(read('{"a":{"b":1')).toEqual({ a: { b: 1 } });
    expect(kinds('{"a":[1,2')).toContain("unclosed array");
  });

  it("closes an array that lost its ] when the object around it ends", () => {
    expect(read('{"a":[1,2},"b":3}')).toEqual({ a: [1, 2] });
  });

  it("closes an object that lost its } when the array around it ends", () => {
    expect(read('[{"a":1]')).toEqual([{ a: 1 }]);
  });
});

describe("parseModelJson: numbers", () => {
  it("reads the malformed numbers a model writes", () => {
    expect(read("[0480,+3,.5,5.,-0.5]")).toEqual([480, 3, 0.5, 5, -0.5]);
    expect(kinds("[0480]")).toContain("malformed number");
  });

  it("drops junk stuck to a number", () => {
    expect(read("[480ms,1.2.3,7]")).toEqual([480, 1.2, 7]);
    expect(kinds("[480ms]")).toContain("junk after number");
  });

  it("reads a sign with no digits as an empty value, not a crash", () => {
    expect(read("[1,-,2]")).toEqual([1, null, 2]);
  });
});

describe("parseModelJson: arithmetic where a number belongs", () => {
  /*
    From a real DeepSeek answer: a model working out a tick offset put the sum in
    the payload, which shifted every later slot of the tuple.
  */
  it("evaluates the expression a real answer contained, verbatim", () => {
    const real =
      '{"events": [[0, 240, "F3", 102], [480, 240, "F3", 102], [1920 - 480, 480, "F3", 106]]}';
    expect(read(real)).toEqual({
      events: [
        [0, 240, "F3", 102],
        [480, 240, "F3", 102],
        [1440, 480, "F3", 106],
      ],
    });
    expect(kinds(real)).toContain("arithmetic expression");
  });

  it("does the multiplication before the addition", () => {
    expect(read("[240 + 120 * 2, 480*3, 960 / 2 - 60]")).toEqual([
      480, 1440, 420,
    ]);
  });

  it("reads a spaced operator on one side only as a missing comma, not a sum", () => {
    expect(read("[1 -2]")).toEqual([1, -2]);
    expect(read("[480 480]")).toEqual([480, 480]);
  });

  it("gives null for a division by zero rather than Infinity", () => {
    expect(read("[1 / 0]")).toEqual([null]);
  });
});

describe("parseModelJson: an answer that ends early and keeps going", () => {
  /*
    The most common real failure: a complete document, then more text — the
    model closed the outer object one brace too soon and carried on with its
    notes. Five of six repairs in a real run.
  */
  it("keeps the complete document and drops what follows it", () => {
    const real =
      '{"plan": {"tempo": 120}, "tracks": [{"measures": []}], "warnings": []}, "warnings": ["Emitting exactly one track"]}';
    expect(read(real)).toEqual({
      plan: { tempo: 120 },
      tracks: [{ measures: [] }],
      warnings: [],
    });
    expect(kinds(real)).toEqual(["trailing text"]);
  });
});

describe("parseModelJson: keys and strings", () => {
  it("reads unquoted and single-quoted keys and strings", () => {
    expect(read("{i: 0, 'voices': []}")).toEqual({ i: 0, voices: [] });
    expect(read("{\"a\": 'it\\'s'}")).toEqual({ a: "it's" });
    expect(kinds("{i: 0}")).toContain("unquoted key");
  });

  it("reads a bare word as a string, so a pitch written without quotes survives", () => {
    expect(read("[0, 480, D3, 100]")).toEqual([0, 480, "D3", 100]);
    expect(read("[0, 480, C#4, 100]")).toEqual([0, 480, "C#4", 100]);
  });

  it("escapes a raw newline or tab inside a string", () => {
    expect(read('{"l":"one\ntwo\tthree"}')).toEqual({ l: "one\ntwo\tthree" });
    expect(kinds('{"l":"a\nb"}')).toContain("raw control character");
  });

  it("keeps a quote the model forgot to escape inside a lyric", () => {
    expect(read('{"l":"she said "hello" to me","n":1}')).toEqual({
      l: 'she said "hello" to me',
      n: 1,
    });
    expect(kinds('{"l":"say "hi" now"}')).toContain("unescaped quote");
  });

  it("repairs an invalid escape, including a space between backslash and letter", () => {
    expect(read('{"a":"it\\\'s","b":"x\\ ny"}')).toEqual({
      a: "it's",
      b: "x\ny",
    });
  });
});

describe("parseModelJson: everything else a model writes", () => {
  it("reads Python constants and NaN", () => {
    expect(read("[True, False, None, NaN]")).toEqual([true, false, null, null]);
  });

  it("skips comments", () => {
    expect(read('{"a":1, // the tempo\n "b":2 /* bars */}')).toEqual({
      a: 1,
      b: 2,
    });
  });

  it("ignores text after the document", () => {
    expect(read('{"a":1} That is the answer.')).toEqual({ a: 1 });
  });

  it("reads an empty value as null", () => {
    expect(read('{"a": , "b": 2}')).toEqual({ a: null, b: 2 });
  });

  it("repairs a realistic compact score with several slips at once", () => {
    const text =
      '{"plan": {"ks": "0M", "ts": "4/4", "tempo": 120}, "tracks": [{"measures": [' +
      '{"i": 0, "voices": [{"events": [[0, 480, "C4", 90], [480, 480, "D4", 88,], [960 480 "E4", 90]]}]},' +
      '{"i": 1, "voices": [{"events": [[0, 1920, "G4", 0480]]}]},' +
      ']}], "warnings": []';
    const value = read(text) as {
      tracks: { measures: { voices: { events: unknown[][] }[] }[] }[];
    };
    const [m0, m1] = value.tracks[0].measures;
    expect(m0.voices[0].events).toEqual([
      [0, 480, "C4", 90],
      [480, 480, "D4", 88],
      [960, 480, "E4", 90],
    ]);
    expect(m1.voices[0].events).toEqual([[0, 1920, "G4", 480]]);
  });
});

describe("parseModelJson: what it refuses", () => {
  it("throws the strict parser's own error for text that is not a document", () => {
    expect(() => parseModelJson("here is your answer")).toThrow();
    expect(() => parseModelJson("")).toThrow();
  });

  it("gives up on something that is mostly garbage", () => {
    const garbage = "{" + "@#$ ".repeat(80) + "}";
    expect(lenientParse(garbage)).toBeNull();
    expect(() => parseModelJson(garbage)).toThrow();
  });

  it("does not recurse without bound", () => {
    expect(() => parseModelJson("[".repeat(5000))).toThrow();
  });
});

describe("repair reporting", () => {
  it("summarises repairs once each, with counts", () => {
    const repairs = lenientParse("[1,,2,,3,]")!.repairs;
    expect(summarizeRepairs(repairs)).toBe("extra comma x2, trailing comma");
  });

  it("shows a window around the first repair", () => {
    const text = '{"a":[1,2,3,],"b":"tail"}';
    const repairs = lenientParse(text)!.repairs;
    expect(windowAroundFirstRepair(text, repairs)).toContain("3,]");
    expect(windowAroundFirstRepair(text, [])).toBe("");
  });
});
