/**
 * @fileoverview Reserved input field extraction for AI invocations
 * @description A handful of input keys are consumed by ShapeShyft rather than
 * passed to the model: `context`, `web_search`, `max_output_tokens`, and the
 * call-time provider override `llm_provider` / `llm_model`.
 *
 * They are pulled out in one pass, before the prompt is built, so a reserved key
 * can never leak into the text the model sees. Doing this per-field at different
 * points in the request is what previously let `web_search` reach the prompt.
 */

import { LLM_PROVIDERS, type LlmProvider } from "../types/index.js";
import { getProviderById } from "../config/providers.js";

/** Input keys ShapeShyft consumes instead of forwarding to the model. */
export const RESERVED_INPUT_FIELDS = [
  "context",
  "web_search",
  "max_output_tokens",
  "llm_provider",
  "llm_model",
] as const;

/** Reserved values pulled from an invocation's input. */
export interface ReservedFields {
  /**
   * Per-call override of the endpoint's configured context. Only a non-empty
   * string counts; anything else is discarded so a malformed value cannot blank
   * out the endpoint's own context.
   */
  context?: string;
  /**
   * The caller's `web_search` preference: `false` only for an explicit `false`,
   * `true` for any other present value, `undefined` when absent. Whether search
   * actually runs is still gated by the endpoint -- this can only turn it off.
   */
  webSearch?: boolean;
  /**
   * Raw `max_output_tokens` as supplied. Left unvalidated here so the caller
   * sees one validation error from `resolveMaxOutputTokens`, which owns the rule.
   */
  maxOutputTokens?: unknown;
  /**
   * Raw `llm_provider` as supplied: a call-time provider override. Validated by
   * `resolveProviderOverride`, which owns the rule.
   */
  llmProvider?: unknown;
  /** Raw `llm_model` as supplied. Read only alongside `llm_provider`. */
  llmModel?: unknown;
  /** The input with every reserved key removed. */
  cleanedInput: unknown;
}

/**
 * Split an invocation's input into reserved values and the payload for the model.
 *
 * @param inputData - Raw input: a JSON body, or parsed query parameters
 * @returns The reserved values plus a copy of the input without them
 */
export function extractReservedFields(inputData: unknown): ReservedFields {
  if (
    typeof inputData !== "object" ||
    inputData === null ||
    Array.isArray(inputData)
  ) {
    return { cleanedInput: inputData };
  }

  const {
    context,
    web_search: webSearchRaw,
    max_output_tokens: maxOutputTokens,
    llm_provider: llmProvider,
    llm_model: llmModel,
    ...cleanedInput
  } = inputData as Record<string, unknown>;

  const result: ReservedFields = { cleanedInput };

  if (typeof context === "string" && context.trim().length > 0) {
    result.context = context;
  }
  if (webSearchRaw !== undefined) {
    // A GET invocation's params are strings, so an explicit `web_search=false`
    // arrives as the string "false". Treating that as truthy would silently
    // ignore the caller's only way to turn search off on that path.
    const isFalse =
      webSearchRaw === false ||
      (typeof webSearchRaw === "string" &&
        webSearchRaw.toLowerCase() === "false");
    result.webSearch = !isFalse;
  }
  if (maxOutputTokens !== undefined) {
    result.maxOutputTokens = maxOutputTokens;
  }
  if (llmProvider !== undefined) {
    result.llmProvider = llmProvider;
  }
  if (llmModel !== undefined) {
    result.llmModel = llmModel;
  }

  return result;
}

/** A validated call-time provider override. */
export interface ProviderOverride {
  provider: LlmProvider;
  /** `llm_model`, or the provider catalog's `defaultModel` when omitted. */
  model: string;
}

/**
 * Validate `llm_provider` / `llm_model` from {@link extractReservedFields}.
 *
 * No `llm_provider` (undefined, null or "") means no override -- `llm_model`
 * alone is ignored, and the endpoint's bound key and model apply as before.
 *
 * @returns `value: null` for no override, the override, or an error message
 *   for a 400
 */
export function resolveProviderOverride(
  llmProvider: unknown,
  llmModel: unknown
): { ok: true; value: ProviderOverride | null } | { ok: false; error: string } {
  if (llmProvider === undefined || llmProvider === null || llmProvider === "") {
    return { ok: true, value: null };
  }
  if (
    typeof llmProvider !== "string" ||
    !(LLM_PROVIDERS as readonly string[]).includes(llmProvider)
  ) {
    return {
      ok: false,
      error: `Invalid llm_provider ${JSON.stringify(llmProvider)}. Expected one of: ${LLM_PROVIDERS.join(", ")}`,
    };
  }
  const provider = llmProvider as LlmProvider;
  if (llmModel !== undefined && llmModel !== null && llmModel !== "") {
    if (typeof llmModel !== "string" || llmModel.trim() === "") {
      return { ok: false, error: "llm_model must be a non-empty string" };
    }
    return { ok: true, value: { provider, model: llmModel.trim() } };
  }
  const model = getProviderById(provider)?.defaultModel;
  if (!model) {
    return {
      ok: false,
      error: `No default model for llm_provider "${provider}"; pass llm_model`,
    };
  }
  return { ok: true, value: { provider, model } };
}
