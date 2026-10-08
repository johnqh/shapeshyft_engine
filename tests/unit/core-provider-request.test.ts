import { describe, it, expect, vi } from "vitest";
import {
  buildProviderRequest,
  parseProviderResponse,
  ProviderResponseError,
} from "../../src/core/index.js";
import { ApiHelper } from "../../src/lib/api-helper.js";
import { OpenAIProvider } from "../../src/services/llm/openai.js";
import { AnthropicProvider } from "../../src/services/llm/anthropic.js";
import { createLLMProvider } from "../../src/services/llm/index.js";
import type { LLMRequest } from "../../src/services/llm/types.js";

const schema = {
  type: "object" as const,
  properties: { answer: { type: "string" as const } },
  required: ["answer"],
};

const base = {
  systemPrompt: "SYSTEM",
  prompt: "USER",
  outputSchema: schema,
};

describe("buildProviderRequest", () => {
  it("describes an OpenAI call with a Bearer auth slot and no key", () => {
    const req = buildProviderRequest({
      ...base,
      provider: "openai",
      model: "gpt-5.4",
      maxTokens: 500,
    });
    expect(req).toMatchObject({
      provider: "openai",
      model: "gpt-5.4",
      method: "POST",
      url: "https://api.openai.com/v1/chat/completions",
      headers: { "content-type": "application/json" },
      auth: { header: "Authorization", prefix: "Bearer " },
    });
    expect(req.body).toEqual({
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "SYSTEM" },
        { role: "user", content: [{ type: "text", text: "USER" }] },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "structured_response",
            description: "Generate structured response matching the schema",
            parameters: schema,
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "structured_response" },
      },
      temperature: 0,
      // OpenAI's reasoning-era models take the newer name
      max_completion_tokens: 500,
    });
    expect(JSON.stringify(req)).not.toMatch(/sk-|api_key/i);
  });

  it("uses the catalog default model when none is given", () => {
    expect(buildProviderRequest({ ...base, provider: "openai" }).model).toBe(
      "gpt-5.6-terra"
    );
    expect(buildProviderRequest({ ...base, provider: "anthropic" }).model).toBe(
      "claude-sonnet-5"
    );
  });

  it("puts no token cap on the wire when there is none", () => {
    const { body } = buildProviderRequest({ ...base, provider: "openai" });
    const wire = JSON.parse(JSON.stringify(body));
    expect(wire).not.toHaveProperty("max_tokens");
    expect(wire).not.toHaveProperty("max_completion_tokens");
  });

  it("turns DeepSeek's thinking off, as invoke does", () => {
    const req = buildProviderRequest({
      ...base,
      provider: "deepseek",
      maxTokens: 100,
    });
    expect(req.url).toBe("https://api.deepseek.com/v1/chat/completions");
    expect(req.model).toBe("deepseek-flash");
    expect(req.body.thinking).toEqual({ type: "disabled" });
    expect(req.body.max_tokens).toBe(100);
    expect(req.body.tool_choice).toBeDefined();
    expect(req.auth).toEqual({ header: "Authorization", prefix: "Bearer " });
  });

  it("sends OpenRouter to its own URL with any vendor/model id", () => {
    const req = buildProviderRequest({
      ...base,
      provider: "openrouter",
      model: "some-vendor/brand-new-model",
      maxTokens: 64,
    });
    expect(req.url).toBe("https://openrouter.ai/api/v1/chat/completions");
    expect(req.model).toBe("some-vendor/brand-new-model");
    expect(req.body.model).toBe("some-vendor/brand-new-model");
    expect(req.body.max_tokens).toBe(64);
    expect(req.body).not.toHaveProperty("thinking");
    expect(req.auth).toEqual({ header: "Authorization", prefix: "Bearer " });
    expect(
      buildProviderRequest({ ...base, provider: "openrouter" }).model
    ).toBe("openai/gpt-5.6-terra");
  });

  it("describes an Anthropic call with x-api-key and the version header", () => {
    const req = buildProviderRequest({
      ...base,
      provider: "anthropic",
      model: "claude-sonnet-5",
    });
    expect(req.url).toBe("https://api.anthropic.com/v1/messages");
    expect(req.headers).toEqual({
      "content-type": "application/json",
      "anthropic-version": "2023-06-01",
    });
    expect(req.auth).toEqual({ header: "x-api-key", prefix: "" });
    expect(req.body).toEqual({
      model: "claude-sonnet-5",
      max_tokens: 4096,
      system: [
        {
          type: "text",
          text: "SYSTEM",
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: [{ role: "user", content: [{ type: "text", text: "USER" }] }],
      tools: [
        {
          name: "structured_response",
          description: "Generate structured response matching the schema",
          input_schema: schema,
        },
      ],
      tool_choice: { type: "tool", name: "structured_response" },
    });
  });

  it("sends Anthropic a temperature only when one was set", () => {
    expect(
      buildProviderRequest({ ...base, provider: "anthropic", temperature: 0.2 })
        .body.temperature
    ).toBe(0.2);
  });

  it("uses Cohere's response_format instead of a tool call", () => {
    const { body } = buildProviderRequest({ ...base, provider: "cohere" });
    expect(body).not.toHaveProperty("tools");
    expect(body.response_format).toMatchObject({ type: "json_object" });
  });

  it("sends Groq a plain-string user turn", () => {
    const { body, url } = buildProviderRequest({ ...base, provider: "groq" });
    expect(url).toBe("https://api.groq.com/openai/v1/chat/completions");
    expect(body.messages).toEqual([
      { role: "system", content: "SYSTEM" },
      { role: "user", content: "USER" },
    ]);
  });

  it.each(["gemini", "jev", "lm_studio"] as const)(
    "refuses %s with a clear error",
    provider => {
      expect(() => buildProviderRequest({ ...base, provider })).toThrow(
        /does not support provider/
      );
    }
  );

  it("refuses Groq's transcription models", () => {
    expect(() =>
      buildProviderRequest({
        ...base,
        provider: "groq",
        model: "whisper-large-v3",
      })
    ).toThrow(/transcription model/);
  });
});

describe("one builder for invoke and /prompt", () => {
  const request: LLMRequest = {
    ...base,
    model: "deepseek-flash",
    maxTokens: 300,
    temperature: 0,
  };

  it.each([
    ["openai", "gpt-5.4"],
    ["deepseek", "deepseek-flash"],
    ["openrouter", "openai/gpt-5.6-terra"],
    ["mistral", "mistral-medium-latest"],
    ["anthropic", "claude-sonnet-5"],
    ["groq", "openai/gpt-oss-120b"],
  ] as const)(
    "%s: buildApiPayload equals buildProviderRequest().body",
    (provider, model) => {
      const adapter = createLLMProvider(provider, { apiKey: "test" });
      expect(adapter.buildApiPayload({ ...request, model })).toEqual(
        buildProviderRequest({ ...request, provider, model }).body
      );
    }
  );

  it("OpenAI: generate() sends that body plus stream:false", async () => {
    const adapter = new OpenAIProvider(
      { apiKey: "test", endpointUrl: "https://api.deepseek.com/v1" },
      { disableThinking: true }
    );
    const create = vi.fn(async () => ({
      model: "deepseek-flash",
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      choices: [
        {
          finish_reason: "stop",
          message: {
            tool_calls: [
              {
                function: {
                  name: "structured_response",
                  arguments: '{"answer":"x"}',
                },
              },
            ],
          },
        },
      ],
    }));
    (adapter as unknown as { client: unknown }).client = {
      chat: { completions: { create } },
    };
    await adapter.generate(request);
    expect(create.mock.calls[0]).toEqual([
      {
        ...buildProviderRequest({ ...request, provider: "deepseek" }).body,
        stream: false,
      },
    ]);
  });

  it("Anthropic: generate() sends that body", async () => {
    const adapter = new AnthropicProvider({ apiKey: "test" });
    const create = vi.fn(async () => ({
      model: "claude-sonnet-5",
      stop_reason: "tool_use",
      usage: { input_tokens: 3, output_tokens: 4 },
      content: [
        {
          type: "tool_use",
          name: "structured_response",
          input: { answer: "y" },
        },
      ],
    }));
    (adapter as unknown as { client: unknown }).client = {
      messages: { create },
    };
    const res = await adapter.generate({
      ...request,
      model: "claude-sonnet-5",
    });
    expect(res.content).toEqual({ answer: "y" });
    expect(create.mock.calls[0]).toEqual([
      buildProviderRequest({
        ...request,
        provider: "anthropic",
        model: "claude-sonnet-5",
      }).body,
    ]);
  });

  it("ApiHelper.providerRequest uses invoke's prompts", () => {
    const input = {
      inputData: { city: "Paris" },
      outputSchema: schema,
      instructions: "Describe the city",
      context: "Travel guide",
      provider: "openai" as const,
    };
    const prompts = ApiHelper.buildLegacyPrompts(input);
    const req = ApiHelper.providerRequest(input);
    expect(req.body.messages).toEqual([
      { role: "system", content: prompts.system },
      { role: "user", content: [{ type: "text", text: prompts.user }] },
    ]);
  });
});

describe("parseProviderResponse", () => {
  it("reads an OpenAI forced call, with usage and stop reason", () => {
    const parsed = parseProviderResponse("openai", {
      model: "gpt-5.4-2026-01-01",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              {
                type: "function",
                function: {
                  name: "structured_response",
                  arguments: '{"answer":"42"}',
                },
              },
            ],
          },
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    });
    expect(parsed).toEqual({
      content: { answer: "42" },
      finishReason: "tool_calls",
      model: "gpt-5.4-2026-01-01",
      usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
    });
  });

  it("repairs the slips a model makes", () => {
    const parsed = parseProviderResponse("deepseek", {
      choices: [
        {
          finish_reason: "stop",
          message: {
            tool_calls: [
              {
                function: {
                  name: "structured_response",
                  arguments: '{"a": [1, 2,],}',
                },
              },
            ],
          },
        },
      ],
    });
    expect(parsed.content).toEqual({ a: [1, 2] });
  });

  it("falls back to message content", () => {
    const parsed = parseProviderResponse("openrouter", {
      choices: [
        { finish_reason: "stop", message: { content: '{"answer":"c"}' } },
      ],
    });
    expect(parsed.content).toEqual({ answer: "c" });
    expect(parsed.usage).toBeUndefined();
  });

  it("returns the raw text of a truncated answer", () => {
    const parsed = parseProviderResponse("openai", {
      choices: [
        {
          finish_reason: "length",
          message: {
            tool_calls: [
              {
                function: { name: "structured_response", arguments: '{"x":[' },
              },
            ],
          },
        },
      ],
    });
    expect(parsed).toMatchObject({ content: '{"x":[', finishReason: "length" });
  });

  it("throws on an unparseable, untruncated answer", () => {
    expect(() =>
      parseProviderResponse("openai", {
        choices: [
          { finish_reason: "stop", message: { content: "not json at all" } },
        ],
      })
    ).toThrow(/unparseable JSON/);
  });

  it("throws the provider's own error message", () => {
    expect(() =>
      parseProviderResponse("openai", {
        error: { message: "Incorrect API key provided" },
      })
    ).toThrow("Incorrect API key provided");
    expect(() =>
      parseProviderResponse("anthropic", {
        type: "error",
        error: { type: "authentication_error", message: "invalid x-api-key" },
      })
    ).toThrow(ProviderResponseError);
  });

  it("reads Anthropic's structured_response tool_use block", () => {
    const parsed = parseProviderResponse("anthropic", {
      model: "claude-sonnet-5",
      stop_reason: "tool_use",
      content: [
        { type: "text", text: "Here you go" },
        {
          type: "tool_use",
          id: "toolu_1",
          name: "structured_response",
          input: { answer: "a" },
        },
      ],
      usage: {
        input_tokens: 100,
        output_tokens: 20,
        cache_read_input_tokens: 50,
        cache_creation_input_tokens: 0,
      },
    });
    expect(parsed).toEqual({
      content: { answer: "a" },
      finishReason: "tool_calls",
      model: "claude-sonnet-5",
      usage: {
        promptTokens: 150,
        completionTokens: 20,
        totalTokens: 170,
        cachedInputTokens: 50,
      },
    });
  });

  it("throws when Anthropic answered without the tool", () => {
    expect(() =>
      parseProviderResponse("anthropic", {
        stop_reason: "max_tokens",
        content: [{ type: "text", text: "..." }],
        usage: { input_tokens: 1, output_tokens: 1 },
      })
    ).toThrow(/structured_response tool_use/);
  });

  it("refuses a provider it cannot read", () => {
    expect(() => parseProviderResponse("gemini", {})).toThrow(
      /does not support/
    );
  });
});
