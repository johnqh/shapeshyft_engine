/**
 * @fileoverview LLM provider type definitions
 * @description Core types for the LLM provider system including request/response
 * types, the provider interface, and cost estimation re-exports.
 */

import type {
  FinishReason,
  JsonSchema,
  LlmProvider,
  MediaContent,
  GeneratedMedia,
} from "../../types/index.js";

// Re-export types for convenience
export type { MediaContent, GeneratedMedia, LlmProvider };

// Re-export cost estimation functions from types package
export {
  estimateCost,
  formatCost,
  formatCostPerMillion,
} from "../../types/index.js";
export type { ModelPricing } from "../../types/index.js";

// Re-export getModelPricing from local providers config (data is now server-side)
export { getModelPricing } from "../../config/providers.js";

// =============================================================================
// LLM Request Types (Discriminated Union)
// =============================================================================

/**
 * Base request fields shared by all variants
 */
interface LLMRequestBase {
  prompt: string;
  systemPrompt?: string;
  outputSchema: JsonSchema;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  /** Input media (images, audio, video) */
  media?: MediaContent[];
  /** What media types this endpoint generates */
  expectsMediaOutput?: {
    image?: boolean;
    audio?: boolean;
    video?: boolean;
  };
  /** Enable web search (OpenAI Responses API) */
  webSearch?: boolean;
  /** For Whisper: model to use for structured extraction from transcription */
  extractionModel?: string;
  /** For Whisper: API key for the extraction model's provider */
  extractionApiKey?: string;
}

/**
 * Request variant when outputMediaFormat is "url" - entityId is REQUIRED
 */
interface LLMRequestWithUrlOutput extends LLMRequestBase {
  outputMediaFormat: "url";
  entityId: string; // Required for storage lookup
}

/**
 * Request variant when outputMediaFormat is "base64" or undefined - entityId is optional
 */
interface LLMRequestWithBase64Output extends LLMRequestBase {
  outputMediaFormat?: "base64";
  entityId?: string;
}

/**
 * Discriminated union ensures entityId is required when outputMediaFormat === "url"
 */
export type LLMRequest = LLMRequestWithUrlOutput | LLMRequestWithBase64Output;

/**
 * Type guard for URL output requests
 */
export function requiresEntityId(
  request: LLMRequest
): request is LLMRequestWithUrlOutput {
  return request.outputMediaFormat === "url";
}

/**
 * Token accounting for one invocation, which may span several provider calls.
 *
 * The first three fields are what every provider reports. The rest are the
 * parts of the bill that plain token counts misprice; an adapter sets only
 * what its provider reports, and `estimateUsageCost` treats an absent field as
 * zero or as "billed at the plain rate".
 */
export interface LLMUsage {
  promptTokens: number;
  /** Everything billed at the output rate, including thinking tokens. */
  completionTokens: number;
  totalTokens: number;
  /** Part of `promptTokens` served from the provider's prompt cache. */
  cachedInputTokens?: number;
  /** Part of `promptTokens` written to the prompt cache, where writes cost extra. */
  cacheWriteInputTokens?: number;
  /** Part of `promptTokens` that was audio, not cached, for providers billing audio by the token. */
  audioInputTokens?: number;
  /** Part of `completionTokens` spent reasoning. Informational; already priced as output. */
  reasoningTokens?: number;
  /** Reasoning tokens billed apart from `completionTokens` (Perplexity). */
  separateReasoningTokens?: number;
  /** Citation tokens billed apart from `completionTokens` (Perplexity). */
  citationTokens?: number;
  /** Web search calls or search queries the provider bills per call. */
  searchCalls?: number;
  /** Billable provider requests. Absent means one. */
  requests?: number;
  /** Largest prompt sent in any single request, for long-context tiers. Absent: `promptTokens`. */
  maxPromptTokens?: number;
  /**
   * The complete cost in cents, when it is known without the catalog: billed
   * by the provider itself (xAI, Perplexity) or priced by the adapter from a
   * per-model and per-tool breakdown (Groq Compound). Replaces the token-based
   * estimate entirely.
   */
  billedCostCents?: number;
}

/**
 * Response from an LLM provider
 */
export interface LLMResponse {
  content: unknown;
  rawResponse: string;
  usage: LLMUsage;
  /**
   * Audio transcribed before generation (Groq Whisper). Priced against the
   * transcription model, which is not `model` when an extraction model
   * produced the structured output.
   */
  transcription?: { model: string; billedSeconds: number };
  model: string;
  provider: LlmProvider;
  latencyMs: number;
  /**
   * Why the model stopped, normalized across providers. `"length"` means the
   * output ceiling was hit and `content` is truncated. Undefined when the
   * provider reported nothing usable.
   */
  finishReason?: FinishReason;
  /** Generated media (images, audio, video) from generative models */
  generatedMedia?: GeneratedMedia[];
}

/**
 * Provider configuration
 */
export interface ProviderConfig {
  apiKey?: string;
  endpointUrl?: string;
  model?: string;
  /**
   * Wall-clock ceiling for one provider call, in milliseconds. Only the custom
   * (LM Studio) provider honours it today; its default is ten minutes. The
   * caller supplies it -- in the API shells, from LM_STUDIO_TIMEOUT_MS.
   */
  timeoutMs?: number;
}

/**
 * LLM Provider interface
 */
export interface ILLMProvider {
  readonly providerName: LlmProvider;

  /**
   * Generate a structured response from the LLM
   */
  generate(request: LLMRequest): Promise<LLMResponse>;

  /**
   * Build the API payload without calling the LLM
   * Used for Type 3 and Type 4 endpoints
   */
  buildApiPayload(request: LLMRequest): Record<string, unknown>;
}
