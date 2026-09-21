import { describe, expect, it } from "vitest";
import {
  AnthropicProvider,
  anthropicUsage,
} from "../../src/services/llm/anthropic.js";
import { estimateUsageCost } from "../../src/lib/cost-estimation.js";
import { getModelPricing } from "../../src/config/providers.js";

type Body = Record<string, unknown>;

/** A provider whose SDK client records the request body and answers with `usage`. */
function providerAnswering(usage: Record<string, unknown>) {
  const seen: Body[] = [];
  const provider = new AnthropicProvider({ apiKey: "test" });
  (provider as unknown as { client: unknown }).client = {
    messages: {
      create: async (body: Body) => {
        seen.push(body);
        return {
          model: "claude-sonnet-5",
          stop_reason: "tool_use",
          usage,
          content: [
            { type: "tool_use", name: "structured_response", input: { ok: 1 } },
          ],
        };
      },
    },
  };
  return { provider, seen };
}

const request = {
  prompt: "write it",
  systemPrompt: "You are a composer.",
  outputSchema: { type: "object" },
};

const cachedSystem = [
  {
    type: "text",
    text: "You are a composer.",
    cache_control: { type: "ephemeral" },
  },
];

describe("Anthropic prompt caching", () => {
  it("marks the system prompt as a cache breakpoint", async () => {
    const { provider, seen } = providerAnswering({
      input_tokens: 10,
      output_tokens: 5,
    });
    await provider.generate(request);
    expect(seen[0].system).toEqual(cachedSystem);
  });

  it("sends no system field, and no empty cached block, without a system prompt", async () => {
    const { provider, seen } = providerAnswering({
      input_tokens: 10,
      output_tokens: 5,
    });
    await provider.generate({ ...request, systemPrompt: undefined });
    await provider.generate({ ...request, systemPrompt: "" });
    expect(seen).toHaveLength(2);
    expect(seen.every(body => !("system" in body))).toBe(true);
  });

  it("shows the same cached block in the prompt-preview payload", () => {
    const { provider } = providerAnswering({});
    expect(provider.buildApiPayload(request).system).toEqual(cachedSystem);
  });

  it("counts cache reads and writes as parts of the prompt, not additions to it", async () => {
    const { provider } = providerAnswering({
      input_tokens: 100,
      cache_read_input_tokens: 2000,
      cache_creation_input_tokens: 300,
      output_tokens: 50,
    });
    const { usage } = await provider.generate(request);
    expect(usage).toEqual({
      promptTokens: 2400,
      completionTokens: 50,
      totalTokens: 2450,
      cachedInputTokens: 2000,
      cacheWriteInputTokens: 300,
    });
  });

  it("omits the cache fields when nothing was cached, as before", () => {
    expect(
      anthropicUsage({
        input_tokens: 12,
        output_tokens: 3,
        cache_read_input_tokens: null,
        cache_creation_input_tokens: null,
      } as never)
    ).toEqual({ promptTokens: 12, completionTokens: 3, totalTokens: 15 });
  });

  it("prices a cache read at a tenth of input and a write at 1.25x", () => {
    const pricing = getModelPricing("claude-sonnet-5");
    // 200 cents/M input: 1M plain = 200, 1M read = 20, 1M written = 250.
    const cost = (extra: Record<string, number>) =>
      estimateUsageCost(pricing, {
        promptTokens: 1_000_000,
        completionTokens: 0,
        totalTokens: 1_000_000,
        ...extra,
      });
    expect(cost({})).toBeCloseTo(200, 6);
    expect(cost({ cachedInputTokens: 1_000_000 })).toBeCloseTo(20, 6);
    expect(cost({ cacheWriteInputTokens: 1_000_000 })).toBeCloseTo(250, 6);
  });
});
