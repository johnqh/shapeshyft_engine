import { describe, expect, it } from "vitest";
import {
  estimateResponseCost,
  estimateUsageCost,
} from "../../src/lib/cost-estimation.js";
import {
  DEFAULT_MODEL_PRICING,
  MODEL_PRICING,
  findModelPricing,
  getModelPricing,
} from "../../src/config/providers.js";
import {
  addUsage,
  compatibleUsage,
} from "../../src/services/llm/compatible-usage.js";
import { geminiUsage } from "../../src/services/llm/gemini.js";
import { compoundCostCents } from "../../src/services/llm/groq.js";
import {
  attachUsage,
  getFailedInvocationUsage,
} from "../../src/services/llm/usage-error.js";
import { OpenAIProvider } from "../../src/services/llm/openai.js";
import type { ModelPricing } from "../../src/types/index.js";

/** A weekday outside DeepSeek's peak windows, before any 2027 price change. */
const OFF_PEAK = new Date("2026-09-14T12:00:00Z"); // Monday 12:00 UTC

const usage = (promptTokens: number, completionTokens: number) => ({
  promptTokens,
  completionTokens,
  totalTokens: promptTokens + completionTokens,
});

describe("estimateUsageCost", () => {
  const base: ModelPricing = { input: 100, output: 400 };

  it("prices plain input and output per 1M tokens, at full precision", () => {
    // 400 * 100/1M + 150 * 400/1M = 0.04 + 0.06
    expect(estimateUsageCost(base, usage(400, 150), OFF_PEAK)).toBeCloseTo(
      0.1,
      12
    );
  });

  it("prices cached, cache-write and audio tokens as parts of the prompt", () => {
    const pricing: ModelPricing = {
      ...base,
      cachedInput: 10,
      cacheWriteInput: 125,
      audioTokenInput: 300,
    };
    const cost = estimateUsageCost(
      pricing,
      {
        ...usage(1_000_000, 0),
        cachedInputTokens: 500_000,
        cacheWriteInputTokens: 100_000,
        audioInputTokens: 100_000,
      },
      OFF_PEAK
    );
    // 300K plain * 100 + 500K * 10 + 100K * 125 + 100K * 300, per 1M
    expect(cost).toBeCloseTo(30 + 5 + 12.5 + 30, 10);
  });

  it("never counts a part twice when the parts exceed the prompt", () => {
    const pricing: ModelPricing = { ...base, cachedInput: 0 };
    const cost = estimateUsageCost(
      pricing,
      { ...usage(100, 0), cachedInputTokens: 500 },
      OFF_PEAK
    );
    expect(cost).toBe(0);
  });

  it("switches every token to the long-context tier past its threshold", () => {
    const pricing: ModelPricing = {
      ...base,
      longContext: { minPromptTokens: 200_000, input: 200, output: 800 },
    };
    expect(estimateUsageCost(pricing, usage(199_999, 0), OFF_PEAK)).toBeCloseTo(
      19.9999,
      8
    );
    expect(
      estimateUsageCost(pricing, usage(200_000, 1_000_000), OFF_PEAK)
    ).toBeCloseTo(40 + 800, 8);
  });

  it("uses the largest single request, not the summed prompt, for the tier", () => {
    const pricing: ModelPricing = {
      ...base,
      longContext: { minPromptTokens: 200_000, input: 200, output: 800 },
    };
    // Three calls of 100K each: 300K in total, but no single call is long.
    const cost = estimateUsageCost(
      pricing,
      { ...usage(300_000, 0), maxPromptTokens: 100_000 },
      OFF_PEAK
    );
    expect(cost).toBeCloseTo(30, 8);
  });

  it("multiplies token cost inside peak windows only", () => {
    const pricing = getModelPricing("deepseek-flash");
    const offPeak = estimateUsageCost(pricing, usage(1_000_000, 0), OFF_PEAK);
    const peak = estimateUsageCost(
      pricing,
      usage(1_000_000, 0),
      new Date("2026-09-14T02:30:00Z") // Monday 02:30 UTC
    );
    const weekend = estimateUsageCost(
      pricing,
      usage(1_000_000, 0),
      new Date("2026-09-13T02:30:00Z") // Sunday
    );
    expect(offPeak).toBeCloseTo(15, 10);
    expect(peak).toBeCloseTo(30, 10);
    expect(weekend).toBeCloseTo(15, 10);
  });

  it("applies an announced price change from its date", () => {
    const pricing = getModelPricing("gemini-3.7-flash");
    expect(estimateUsageCost(pricing, usage(1_000_000, 0), OFF_PEAK)).toBe(75);
    expect(
      estimateUsageCost(
        pricing,
        usage(1_000_000, 0),
        new Date("2027-01-01T00:00:00Z")
      )
    ).toBe(150);
  });

  it("adds request, search, citation and separate reasoning charges", () => {
    const pricing: ModelPricing = {
      ...base,
      requestFee: 0.5,
      searchCall: 1,
      citationTokens: 200,
      reasoningTokens: 300,
    };
    const cost = estimateUsageCost(
      pricing,
      {
        ...usage(0, 0),
        requests: 2,
        searchCalls: 3,
        citationTokens: 1_000_000,
        separateReasoningTokens: 1_000_000,
      },
      OFF_PEAK
    );
    expect(cost).toBeCloseTo(1 + 3 + 200 + 300, 10);
  });

  it("returns a billed cost as is, without consulting the rates", () => {
    expect(
      estimateUsageCost(
        base,
        { ...usage(9_999_999, 9_999_999), billedCostCents: 0.42 },
        OFF_PEAK
      )
    ).toBe(0.42);
  });
});

describe("model pricing lookup", () => {
  it("prices any model on a self-hosted server at zero", () => {
    expect(
      getModelPricing("my-own-finetune", { provider: "lm_studio" })
    ).toEqual({ input: 0, output: 0 });
  });

  it("strips OpenAI and Anthropic snapshot suffixes", () => {
    expect(getModelPricing("gpt-4.1-mini-2025-04-14")).toBe(
      MODEL_PRICING["gpt-4.1-mini"]
    );
    expect(getModelPricing("claude-haiku-4-5-20251001")).toBe(
      MODEL_PRICING["claude-haiku-4-5"]
    );
  });

  it("falls back to the configured model when the reported one is unknown", () => {
    expect(
      getModelPricing("some-provider-internal-name", {
        configuredModel: "grok-4.6",
      })
    ).toBe(MODEL_PRICING["grok-4.6"]);
  });

  it("reports an unknown model as unknown rather than guessing", () => {
    expect(findModelPricing("no-such-model")).toBeUndefined();
    expect(getModelPricing("no-such-model")).toBe(DEFAULT_MODEL_PRICING);
  });
});

describe("estimateResponseCost", () => {
  it("prices a response under the credential's provider", () => {
    const { costCents, pricingKnown } = estimateResponseCost(
      { model: "llama-3.2-3b", usage: usage(1_000_000, 1_000_000) },
      { provider: "lm_studio", configuredModel: "llama-3.2-3b", at: OFF_PEAK }
    );
    expect(costCents).toBe(0);
    expect(pricingKnown).toBe(true);
  });

  it("flags a cost that fell back to the default rate", () => {
    const { pricingKnown } = estimateResponseCost(
      { model: "mystery", usage: usage(10, 10) },
      { provider: "openai", configuredModel: "mystery-too", at: OFF_PEAK }
    );
    expect(pricingKnown).toBe(false);
  });

  it("adds Whisper transcription by the minute to the extraction model's tokens", () => {
    const { costCents } = estimateResponseCost(
      {
        model: "openai/gpt-oss-20b",
        usage: usage(1_000_000, 0),
        transcription: { model: "whisper-large-v3", billedSeconds: 120 },
      },
      { provider: "groq", at: OFF_PEAK }
    );
    // 7.5 cents of input + 2 minutes at $0.111/hour
    expect(costCents).toBeCloseTo(7.5 + 2 * 0.185, 10);
  });
});

describe("compatibleUsage", () => {
  it("reads OpenAI cached, cache-write and reasoning parts", () => {
    const u = compatibleUsage({
      prompt_tokens: 1000,
      completion_tokens: 500,
      total_tokens: 1500,
      prompt_tokens_details: { cached_tokens: 600, cache_write_tokens: 200 },
      completion_tokens_details: { reasoning_tokens: 400 },
    });
    expect(u).toMatchObject({
      promptTokens: 1000,
      completionTokens: 500,
      cachedInputTokens: 600,
      cacheWriteInputTokens: 200,
      reasoningTokens: 400,
    });
  });

  it("reads DeepSeek cache hits", () => {
    const u = compatibleUsage({
      prompt_tokens: 1000,
      completion_tokens: 10,
      prompt_cache_hit_tokens: 900,
      prompt_cache_miss_tokens: 100,
    });
    expect(u.cachedInputTokens).toBe(900);
  });

  it("adds xAI reasoning, which completion_tokens leaves out, and uses its billed cost", () => {
    const u = compatibleUsage({
      prompt_tokens: 32,
      completion_tokens: 9,
      total_tokens: 135,
      completion_tokens_details: { reasoning_tokens: 94 },
      cost_in_usd_ticks: 25_000_000, // $0.0025
    });
    expect(u.completionTokens).toBe(103);
    expect(u.billedCostCents).toBeCloseTo(0.25, 12);
  });

  it("reads Perplexity's separate lines and billed cost", () => {
    const u = compatibleUsage({
      prompt_tokens: 20,
      completion_tokens: 11_395,
      total_tokens: 11_415,
      citation_tokens: 7_000,
      num_search_queries: 12,
      reasoning_tokens: 193_947,
      cost: { input_tokens_cost: 0.00004, total_cost: 0.8231 },
    });
    expect(u).toMatchObject({
      completionTokens: 11_395,
      citationTokens: 7_000,
      searchCalls: 12,
      separateReasoningTokens: 193_947,
    });
    expect(u.billedCostCents).toBeCloseTo(82.31, 10);
  });

  it("adds calls together, keeping the largest single prompt", () => {
    const total = addUsage(
      { ...usage(100, 10), cachedInputTokens: 50 },
      { ...usage(300, 20), searchCalls: 2 }
    );
    expect(total).toMatchObject({
      promptTokens: 400,
      completionTokens: 30,
      cachedInputTokens: 50,
      searchCalls: 2,
      requests: 2,
      maxPromptTokens: 300,
    });
  });

  it("drops a billed cost that only one of the calls reported", () => {
    const total = addUsage({ ...usage(1, 1), billedCostCents: 1 }, usage(1, 1));
    expect(total.billedCostCents).toBeUndefined();
  });
});

describe("geminiUsage", () => {
  it("bills thinking tokens as output and splits out audio and cache", () => {
    const u = geminiUsage({
      promptTokenCount: 2000,
      candidatesTokenCount: 100,
      thoughtsTokenCount: 900,
      totalTokenCount: 3000,
      cachedContentTokenCount: 500,
      promptTokensDetails: [
        { modality: "TEXT", tokenCount: 1200 },
        { modality: "AUDIO", tokenCount: 800 },
      ],
      cacheTokensDetails: [{ modality: "AUDIO", tokenCount: 300 }],
    });
    expect(u).toMatchObject({
      promptTokens: 2000,
      completionTokens: 1000,
      cachedInputTokens: 500,
      audioInputTokens: 500,
      reasoningTokens: 900,
    });
  });
});

describe("compoundCostCents", () => {
  it("prices each model in the breakdown and each executed tool", () => {
    const cents = compoundCostCents(
      {
        usage_breakdown: {
          models: [
            {
              model: "openai/gpt-oss-120b",
              usage: { prompt_tokens: 1_000_000, completion_tokens: 0 },
            },
            {
              model: "unlisted-helper",
              usage: { prompt_tokens: 0, completion_tokens: 1_000_000 },
            },
          ],
        },
        choices: [
          {
            message: {
              executed_tools: [{ type: "search" }, { type: "visit" }],
            },
          },
        ],
      },
      "groq/compound",
      OFF_PEAK
    );
    // 15 (120b input) + 60 (unlisted helper at compound's own output rate)
    // + 0.8 search + 0.1 visit
    expect(cents).toBeCloseTo(15 + 60 + 0.8 + 0.1, 10);
  });

  it("returns undefined when there is no breakdown to price", () => {
    expect(compoundCostCents({ choices: [] }, "groq/compound")).toBeUndefined();
  });
});

describe("failed invocation usage", () => {
  it("carries usage on the error without changing its message", () => {
    const error = attachUsage(new Error("bad json"), usage(5, 6), "gpt-5.4");
    expect(error.message).toBe("bad json");
    expect(getFailedInvocationUsage(error)).toEqual({
      usage: usage(5, 6),
      model: "gpt-5.4",
    });
    expect(getFailedInvocationUsage(new Error("plain"))).toBeUndefined();
  });

  it("is attached by the OpenAI adapter when the model returns unparseable JSON", async () => {
    const provider = new OpenAIProvider({ apiKey: "test" });
    (
      provider as unknown as {
        client: { chat: { completions: { create: () => Promise<unknown> } } };
      }
    ).client = {
      chat: {
        completions: {
          create: async () => ({
            model: "gpt-5.4",
            usage: {
              prompt_tokens: 40,
              completion_tokens: 7,
              total_tokens: 47,
            },
            choices: [
              {
                finish_reason: "stop",
                message: {
                  tool_calls: [
                    {
                      function: {
                        name: "structured_response",
                        arguments: "not json",
                      },
                    },
                  ],
                },
              },
            ],
          }),
        },
      },
    };

    const failure = await provider
      .generate({
        prompt: "p",
        outputSchema: { type: "object" },
        model: "gpt-5.4",
      })
      .catch((e: unknown) => e);

    expect(getFailedInvocationUsage(failure)).toMatchObject({
      model: "gpt-5.4",
      usage: { promptTokens: 40, completionTokens: 7 },
    });
  });
});
