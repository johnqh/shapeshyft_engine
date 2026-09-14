import { describe, it, expect } from "vitest";
import { CustomLLMProvider } from "../../src/services/llm/custom.js";

/** `timeout` is private; read it the way the provider's fetch does. */
function timeoutOf(provider: CustomLLMProvider): number {
  return (provider as unknown as { timeout: number }).timeout;
}

describe("CustomLLMProvider timeout", () => {
  it("defaults to ten minutes", () => {
    const provider = new CustomLLMProvider({
      endpointUrl: "http://localhost:1234/v1",
    });
    expect(timeoutOf(provider)).toBe(600_000);
  });

  it("uses timeoutMs from the provider config", () => {
    const provider = new CustomLLMProvider({
      endpointUrl: "http://localhost:1234/v1",
      timeoutMs: 1_800_000,
    });
    expect(timeoutOf(provider)).toBe(1_800_000);
  });
});
