/**
 * @fileoverview @sudobility/shapeshyft_engine
 * @description Stateless LLM structured-output engine. No database, no HTTP
 * framework, no environment reads -- callers pass configuration in.
 *
 * Domain types live at `@sudobility/shapeshyft_engine/types`, not here, so the
 * two `ProviderConfig` and `PROVIDER_MODELS` declarations (LLM call config vs.
 * provider catalog entry) never collide in one namespace.
 */

export * from "./services/llm/index.js";
export { OpenAIProvider } from "./services/llm/openai.js";
export { AnthropicProvider } from "./services/llm/anthropic.js";
export { GeminiProvider } from "./services/llm/gemini.js";
export { GroqProvider } from "./services/llm/groq.js";
export { CustomLLMProvider } from "./services/llm/custom.js";
export * from "./services/llm/finish-reason.js";
export * from "./services/llm/extract-json.js";
export * from "./config/providers.js";
export * from "./lib/prompt-builder.js";
export * from "./lib/api-helper.js";
export * from "./lib/media-constants.js";
export * from "./lib/media-utils.js";
export * from "./lib/media-conversion.js";
export * from "./lib/capability-validator.js";
export * from "./lib/reserved-fields.js";
export * from "./lib/output-limit.js";
