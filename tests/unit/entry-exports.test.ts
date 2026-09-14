import { describe, it, expect } from "vitest";
import * as engine from "../../src/index.js";

describe("engine root exports", () => {
  it.each([
    "createLLMProvider",
    "PROVIDER_ENDPOINTS",
    "estimateCost",
    "estimateUsageCost",
    "estimateResponseCost",
    "getModelPricing",
    "findModelPricing",
    "getFailedInvocationUsage",
    "ApiHelper",
    "extractReservedFields",
    "resolveMaxOutputTokens",
    "extractMediaFromInput",
    "convertAllMediaIfNeeded",
    "validateMediaCapabilities",
    "validateWhisperRequest",
    "isTranscriptionModel",
    "normalizeFinishReason",
    "extractJson",
    "PROVIDERS",
    "MODEL_CAPABILITIES",
    "MODEL_PRICING",
    "getProviderById",
    "OpenAIProvider",
    "CustomLLMProvider",
  ])("exports %s", name => {
    expect(engine).toHaveProperty(name);
  });
});
