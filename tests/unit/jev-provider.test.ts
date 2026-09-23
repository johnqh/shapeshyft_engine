import { afterEach, describe, expect, it, vi } from "vitest";
import {
  JevProvider,
  checkJevCompatibility,
  classifyJevField,
} from "../../src/services/llm/jev.js";
import { getFailedInvocationUsage } from "../../src/services/llm/usage-error.js";
import type { LLMRequest } from "../../src/services/llm/types.js";

const outputSchema = {
  type: "object",
  properties: {
    is_spam: { type: "boolean", description: "Is this message spam?" },
    category: {
      type: "string",
      enum: ["billing", "technical", "other"],
      description: "What is this ticket about?",
    },
    severity: {
      type: "string",
      enum: ["low", "medium", "high"],
      "x-jev-kind": "score",
      description: "How severe is the issue?",
    },
  },
} as const;

const request: LLMRequest = {
  prompt: "My card was charged twice.",
  outputSchema: outputSchema as unknown as LLMRequest["outputSchema"],
};

function jevResponse(overrides: Record<string, unknown> = {}) {
  return {
    model: "jev-1.13.0",
    answers: {
      is_spam: { type: "noul", noul: 0.1 },
      category: {
        type: "choice",
        choice: "billing",
        confidence: 0.92,
        probabilities: { billing: 0.92, technical: 0.05, other: 0.03 },
      },
      severity: {
        type: "score",
        score: 1.6,
        confidence: 0.4,
        legend: { "0": "low", "1": "medium", "2": "high" },
        probabilities: { "0": 0.1, "1": 0.3, "2": 0.6 },
      },
    },
    usage: { input_tokens: 120, output_tokens: 9 },
    ...overrides,
  };
}

function stubFetch(
  body: unknown,
  init: { ok?: boolean; status?: number } = {}
) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fetchMock = vi.fn(async (url: string, reqInit: RequestInit) => {
    calls.push({ url, init: reqInit });
    return {
      ok: init.ok ?? true,
      status: init.status ?? 200,
      statusText: "",
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("classifyJevField", () => {
  it("maps a boolean field to Noul", () => {
    const result = classifyJevField("is_spam", { type: "boolean" });
    expect(result).toEqual({
      field: {
        key: "is_spam",
        question: { type: "noul", instructions: "is_spam" },
      },
    });
  });

  it("maps a string enum to Choice", () => {
    const result = classifyJevField("category", {
      type: "string",
      enum: ["a", "b"],
    });
    expect("field" in result && result.field.question).toEqual({
      type: "choice",
      instructions: "category",
      criteria: { a: null, b: null },
    });
  });

  it("maps a string enum with x-jev-kind: score to Score, in enum order", () => {
    const result = classifyJevField("severity", {
      type: "string",
      enum: ["low", "medium", "high"],
      "x-jev-kind": "score",
    });
    expect("field" in result && result.field.question).toEqual({
      type: "score",
      instructions: "severity",
      criteria: ["low", "medium", "high"],
    });
  });

  it("rejects a free-form string field", () => {
    const result = classifyJevField("summary", { type: "string" });
    expect("error" in result).toBe(true);
  });

  it("rejects a number field", () => {
    const result = classifyJevField("count", { type: "number" });
    expect("error" in result).toBe(true);
  });

  it("rejects a score rubric with only one level", () => {
    const result = classifyJevField("severity", {
      type: "string",
      enum: ["only-one"],
      "x-jev-kind": "score",
    });
    expect("error" in result).toBe(true);
  });
});

describe("checkJevCompatibility", () => {
  it("accepts an object schema of boolean/enum fields", () => {
    expect(checkJevCompatibility(outputSchema as never)).toEqual({
      compatible: true,
      errors: [],
    });
  });

  it("rejects a non-object schema", () => {
    const result = checkJevCompatibility({ type: "string" } as never);
    expect(result.compatible).toBe(false);
    expect(result.errors).toHaveLength(1);
  });

  it("names every incompatible field, not just the first", () => {
    const result = checkJevCompatibility({
      type: "object",
      properties: {
        summary: { type: "string" },
        tags: { type: "array" },
        ok: { type: "boolean" },
      },
    } as never);
    expect(result.compatible).toBe(false);
    expect(result.errors).toHaveLength(2);
    expect(result.errors.join(" ")).toContain("summary");
    expect(result.errors.join(" ")).toContain("tags");
  });
});

describe("JevProvider", () => {
  it("requires an API key", () => {
    expect(() => new JevProvider({})).toThrow(/API key/);
  });

  it("sends state and one question per field, decodes each answer type back to the schema's shape", async () => {
    const calls = stubFetch(jevResponse());
    const provider = new JevProvider({ apiKey: "test" });

    const result = await provider.generate(request);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://api.typesafe.ai/v1/systemone");
    expect(calls[0].init.headers).toMatchObject({
      Authorization: "Bearer test",
    });
    const sentBody = JSON.parse(calls[0].init.body as string);
    expect(sentBody.model).toBe("jev-latest");
    expect(sentBody.state).toBe("My card was charged twice.");
    expect(Object.keys(sentBody.questions)).toEqual([
      "is_spam",
      "category",
      "severity",
    ]);

    // Noul thresholds at 0.5, Choice passes the label through, Score rounds
    // the continuous position (1.6 -> index 2) back to the nearest enum level.
    expect(result.content).toEqual({
      is_spam: false,
      category: "billing",
      severity: "high",
    });
    expect(result.provider).toBe("jev");
    expect(result.model).toBe("jev-1.13.0");
    expect(result.finishReason).toBe("stop");
    expect(result.usage).toEqual({
      promptTokens: 120,
      completionTokens: 9,
      totalTokens: 129,
    });
  });

  it("puts the system prompt alongside the input, not into a question", async () => {
    stubFetch(jevResponse());
    const provider = new JevProvider({ apiKey: "test" });
    const payload = provider.buildApiPayload({
      ...request,
      systemPrompt: "Be strict.",
    });
    expect(payload.state).toEqual({
      instructions: "Be strict.",
      input: "My card was charged twice.",
    });
  });

  it("rejects media input without calling Jev", async () => {
    const calls = stubFetch(jevResponse());
    const provider = new JevProvider({ apiKey: "test" });
    await expect(
      provider.generate({
        ...request,
        media: [
          {
            type: "image",
            format: "url",
            data: "https://x/y.png",
            mimeType: "image/png",
          },
        ] as never,
      })
    ).rejects.toThrow(/text-only/);
    expect(calls).toHaveLength(0);
  });

  it("throws with every incompatible field named, without calling Jev", async () => {
    const calls = stubFetch(jevResponse());
    const provider = new JevProvider({ apiKey: "test" });
    await expect(
      provider.generate({
        ...request,
        outputSchema: {
          type: "object",
          properties: { summary: { type: "string" } },
        },
      })
    ).rejects.toThrow(/summary/);
    expect(calls).toHaveLength(0);
  });

  it("attaches usage to the error when Jev omits an answer", async () => {
    stubFetch(
      jevResponse({
        answers: {
          is_spam: { type: "noul", noul: 0.1 },
          category: {
            type: "choice",
            choice: "billing",
            confidence: 0.9,
            probabilities: {},
          },
          // severity missing
        },
      })
    );
    const provider = new JevProvider({ apiKey: "test" });
    try {
      await provider.generate(request);
      expect.unreachable("expected generate() to throw");
    } catch (error) {
      expect((error as Error).message).toContain("severity");
      expect(getFailedInvocationUsage(error)?.usage).toEqual({
        promptTokens: 120,
        completionTokens: 9,
        totalTokens: 129,
      });
    }
  });

  it("surfaces a non-2xx response as an error", async () => {
    stubFetch({ error: "bad key" }, { ok: false, status: 401 });
    const provider = new JevProvider({ apiKey: "test" });
    await expect(provider.generate(request)).rejects.toThrow(/401/);
  });
});
