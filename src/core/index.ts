/**
 * @fileoverview @sudobility/shapeshyft_engine/core
 * @description The pure half of the engine: build a provider request without
 * sending it, and read the provider's reply. Safe for React Native and the
 * browser -- no provider SDK, no `sharp`, no `Buffer`, no Node built-ins.
 * `tests/unit/core-no-runtime-deps.test.ts` holds it to that.
 */

export type {
  AiPromptResponse,
  AiProviderRequest,
  FinishReason,
  JsonSchema,
  LlmProvider,
  MediaContent,
} from "../types/index.js";
export type { LLMUsage } from "../services/llm/types.js";

export {
  buildProviderRequest,
  defaultModelFor,
  PROVIDER_REQUEST_SUPPORTED,
  type ProviderRequestInput,
} from "./request.js";
export {
  parseProviderResponse,
  ProviderResponseError,
  anthropicUsage,
  findAnthropicToolUse,
  openAIStructuredText,
  type ParsedProviderResponse,
  type AnthropicUsageLike,
} from "./response.js";
export {
  buildOpenAIChatBody,
  buildAnthropicMessagesBody,
  openAIChatDialect,
  tokenLimitParamFor,
  STRUCTURED_RESPONSE_TOOL,
  ANTHROPIC_DEFAULT_MAX_TOKENS,
  type ChatBodyInput,
  type OpenAIChatDialect,
  type StructuredOutputMode,
  type TokenLimitParam,
} from "./payload.js";
export {
  PROVIDER_ENDPOINTS,
  OPENAI_COMPATIBLE_BASE_URLS,
  ANTHROPIC_VERSION,
} from "./endpoints.js";
export {
  parseModelJson,
  type ParsedModelJson,
  type JsonRepair,
} from "../services/llm/json-repair.js";
