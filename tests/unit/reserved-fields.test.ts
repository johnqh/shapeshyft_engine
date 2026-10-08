import { describe, it, expect } from "vitest";
import {
  extractReservedFields,
  resolveProviderOverride,
} from "../../src/lib/reserved-fields.js";
import { findModelPricing } from "../../src/config/providers.js";

describe("extractReservedFields", () => {
  it("leaves ordinary input untouched", () => {
    const input = { text: "hello", count: 3 };
    const result = extractReservedFields(input);

    expect(result.cleanedInput).toEqual({ text: "hello", count: 3 });
    expect(result.context).toBeUndefined();
    expect(result.webSearch).toBeUndefined();
    expect(result.maxOutputTokens).toBeUndefined();
  });

  it("removes every reserved field from the input the model sees", () => {
    const input = {
      text: "hello",
      context: "be terse",
      web_search: false,
      max_output_tokens: 2000,
    };

    const result = extractReservedFields(input);

    expect(result.cleanedInput).toEqual({ text: "hello" });
  });

  it("does not mutate the caller's object", () => {
    const input = { text: "hello", max_output_tokens: 2000 };
    extractReservedFields(input);
    expect(input).toEqual({ text: "hello", max_output_tokens: 2000 });
  });

  describe("context", () => {
    it("extracts a non-empty string", () => {
      expect(extractReservedFields({ context: "be terse" }).context).toBe(
        "be terse"
      );
    });

    it("ignores a blank string, so it does not override the endpoint's", () => {
      expect(extractReservedFields({ context: "   " }).context).toBeUndefined();
    });

    it("ignores a non-string", () => {
      expect(extractReservedFields({ context: 42 }).context).toBeUndefined();
    });

    it("still strips a rejected context from the input", () => {
      expect(extractReservedFields({ a: 1, context: 42 }).cleanedInput).toEqual(
        {
          a: 1,
        }
      );
    });
  });

  describe("web_search", () => {
    it("is undefined when absent", () => {
      expect(extractReservedFields({ a: 1 }).webSearch).toBeUndefined();
    });

    it("is false only for an explicit false", () => {
      expect(extractReservedFields({ web_search: false }).webSearch).toBe(
        false
      );
    });

    it("is true for any other present value", () => {
      expect(extractReservedFields({ web_search: true }).webSearch).toBe(true);
      expect(extractReservedFields({ web_search: "yes" }).webSearch).toBe(true);
    });

    it('treats the string "false" as false, since GET params are strings', () => {
      expect(extractReservedFields({ web_search: "false" }).webSearch).toBe(
        false
      );
      expect(extractReservedFields({ web_search: "FALSE" }).webSearch).toBe(
        false
      );
    });
  });

  describe("max_output_tokens", () => {
    it("passes the raw value through for validation elsewhere", () => {
      expect(
        extractReservedFields({ max_output_tokens: 2000 }).maxOutputTokens
      ).toBe(2000);
      expect(
        extractReservedFields({ max_output_tokens: "bad" }).maxOutputTokens
      ).toBe("bad");
    });

    it("is undefined when absent", () => {
      expect(extractReservedFields({ a: 1 }).maxOutputTokens).toBeUndefined();
    });
  });

  describe("non-object input", () => {
    it.each([
      ["a string", "just text"],
      ["an array", [1, 2, 3]],
      ["null", null],
      ["a number", 7],
    ])(
      "passes %s through unchanged with no reserved fields",
      (_label, input) => {
        const result = extractReservedFields(input);
        expect(result.cleanedInput).toEqual(input);
        expect(result.context).toBeUndefined();
        expect(result.webSearch).toBeUndefined();
        expect(result.maxOutputTokens).toBeUndefined();
      }
    );
  });
});

describe("llm_provider / llm_model", () => {
  it("are stripped from the input and returned raw", () => {
    const result = extractReservedFields({
      text: "hi",
      llm_provider: "anthropic",
      llm_model: "claude-sonnet-5",
    });
    expect(result.cleanedInput).toEqual({ text: "hi" });
    expect(result.llmProvider).toBe("anthropic");
    expect(result.llmModel).toBe("claude-sonnet-5");
  });

  it("mean no override when llm_provider is absent, even with llm_model", () => {
    expect(resolveProviderOverride(undefined, "gpt-5.4")).toEqual({
      ok: true,
      value: null,
    });
    expect(resolveProviderOverride("", undefined)).toEqual({
      ok: true,
      value: null,
    });
  });

  it("default the model from the provider catalog", () => {
    expect(resolveProviderOverride("deepseek", undefined)).toEqual({
      ok: true,
      value: { provider: "deepseek", model: "deepseek-flash" },
    });
    expect(resolveProviderOverride("openrouter", "")).toEqual({
      ok: true,
      value: { provider: "openrouter", model: "openai/gpt-5.6-terra" },
    });
  });

  it("keep an explicit model", () => {
    expect(resolveProviderOverride("openai", "gpt-5.4")).toEqual({
      ok: true,
      value: { provider: "openai", model: "gpt-5.4" },
    });
  });

  it("reject an unknown provider id or a non-string model", () => {
    const bad = resolveProviderOverride("chatgpt", undefined);
    expect(bad.ok).toBe(false);
    expect(!bad.ok && bad.error).toMatch(/Invalid llm_provider "chatgpt"/);
    expect(resolveProviderOverride(42, undefined).ok).toBe(false);
    expect(resolveProviderOverride("openai", 7).ok).toBe(false);
  });
});

describe("OpenRouter pricing", () => {
  it("falls back to the vendor's own id after the slash", () => {
    expect(
      findModelPricing("anthropic/claude-sonnet-5", { provider: "openrouter" })
    ).toEqual(findModelPricing("claude-sonnet-5"));
  });

  it("does not strip the prefix for other providers", () => {
    expect(
      findModelPricing("anthropic/claude-sonnet-5", { provider: "groq" })
    ).toBeUndefined();
  });
});
