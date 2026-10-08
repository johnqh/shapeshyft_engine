/**
 * @fileoverview Reading a provider's raw JSON reply
 * @description `parseProviderResponse` turns the JSON a provider returned for a
 * {@link buildProviderRequest} call into the structured answer, the stop reason
 * and the usage. The pieces it is made of are what the SDK adapters use to read
 * the same replies.
 */

import type { FinishReason, LlmProvider } from "../types/index.js";
import type { LLMUsage } from "../services/llm/types.js";
import { normalizeFinishReason } from "../services/llm/finish-reason.js";
import { compatibleUsage } from "../services/llm/compatible-usage.js";
import { parseModelJson } from "../services/llm/json-repair.js";
import { openAIChatDialect, STRUCTURED_RESPONSE_TOOL } from "./payload.js";

/** What {@link parseProviderResponse} reads from a reply. */
export interface ParsedProviderResponse {
  /**
   * The structured answer. On a truncated OpenAI-compatible reply
   * (`finishReason === "length"`), the raw text the model produced, unparsed.
   */
  content: unknown;
  /** Normalized stop reason, or null when the provider reported none. */
  finishReason: FinishReason | null;
  usage?: LLMUsage;
  /** The model the provider says answered, when it says. */
  model?: string;
}

/** A reply that holds no usable answer. `usage` is set when it was reported. */
export class ProviderResponseError extends Error {
  readonly usage?: LLMUsage;

  constructor(message: string, usage?: LLMUsage) {
    super(message);
    this.name = "ProviderResponseError";
    this.usage = usage;
  }
}

// =============================================================================
// Anthropic
// =============================================================================

/** The `usage` object of an Anthropic Messages reply. */
export interface AnthropicUsageLike {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}

/**
 * Anthropic reports `input_tokens` as only the tokens neither read from nor
 * written to the cache, so the whole prompt is the three added together. The
 * cached and written parts stay named, as `LLMUsage` defines them: parts of
 * `promptTokens`, priced apart.
 */
export function anthropicUsage(usage: AnthropicUsageLike): LLMUsage {
  const read = usage.cache_read_input_tokens ?? 0;
  const written = usage.cache_creation_input_tokens ?? 0;
  const promptTokens = usage.input_tokens + read + written;
  return {
    promptTokens,
    completionTokens: usage.output_tokens,
    totalTokens: promptTokens + usage.output_tokens,
    ...(read > 0 ? { cachedInputTokens: read } : {}),
    ...(written > 0 ? { cacheWriteInputTokens: written } : {}),
  };
}

/**
 * The first `tool_use` block of an Anthropic reply's `content`, if any.
 * Callers check its `name` is `structured_response`.
 */
export function findAnthropicToolUse(
  content: unknown
): { name: string; input: unknown } | undefined {
  if (!Array.isArray(content)) return undefined;
  const block = content.find(
    b =>
      b &&
      typeof b === "object" &&
      (b as { type?: unknown }).type === "tool_use"
  ) as { name?: unknown; input?: unknown } | undefined;
  return block && typeof block.name === "string"
    ? { name: block.name, input: block.input }
    : undefined;
}

// =============================================================================
// OpenAI-compatible
// =============================================================================

/**
 * The text carrying the structured answer in an OpenAI chat-completions
 * message: the `structured_response` call's arguments, or, in Cohere's
 * `response_format` mode, the message content. Undefined when absent.
 */
export function openAIStructuredText(
  message: unknown,
  mode: "tool_call" | "response_format" = "tool_call"
): string | undefined {
  if (!message || typeof message !== "object") return undefined;
  const m = message as {
    content?: unknown;
    tool_calls?: Array<{ function?: { name?: unknown; arguments?: unknown } }>;
  };
  if (mode === "response_format") {
    return typeof m.content === "string" && m.content.length > 0
      ? m.content
      : undefined;
  }
  const fn = m.tool_calls?.[0]?.function;
  return fn?.name === STRUCTURED_RESPONSE_TOOL &&
    typeof fn.arguments === "string"
    ? fn.arguments
    : undefined;
}

// =============================================================================
// parseProviderResponse
// =============================================================================

/** The provider's own error message, when the reply is an error body. */
function providerErrorMessage(
  raw: Record<string, unknown>
): string | undefined {
  const error = raw.error;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const message = (error as { message?: unknown }).message;
    return typeof message === "string" ? message : JSON.stringify(error);
  }
  return undefined;
}

/**
 * Read a provider's raw JSON reply to a {@link buildProviderRequest} call.
 *
 * - OpenAI-compatible: `choices[0].message.tool_calls[0].function.arguments`
 *   of the `structured_response` call, else `message.content` -- read with the
 *   tolerant JSON reader, which repairs the common slips a model makes.
 * - Anthropic: the `tool_use` block named `structured_response`; its `input`.
 *
 * @param provider - The provider the request went to
 * @param raw - The parsed JSON body of the reply
 * A truncated OpenAI-compatible reply (`finishReason: "length"`) returns its
 * raw text as `content`, unparsed.
 *
 * @throws ProviderResponseError when the reply is an error, carries no
 *   answer, or the answer is not JSON (and the reply was not truncated)
 */
export function parseProviderResponse(
  provider: LlmProvider,
  raw: unknown
): ParsedProviderResponse {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new ProviderResponseError("Provider reply is not a JSON object");
  }
  const body = raw as Record<string, unknown>;
  const model = typeof body.model === "string" ? body.model : undefined;

  if (provider === "anthropic") {
    const usage =
      body.usage && typeof body.usage === "object"
        ? anthropicUsage(body.usage as AnthropicUsageLike)
        : undefined;
    const errorMessage = providerErrorMessage(body);
    if (errorMessage) throw new ProviderResponseError(errorMessage, usage);
    const finishReason = normalizeFinishReason(body.stop_reason) ?? null;
    const toolUse = findAnthropicToolUse(body.content);
    if (!toolUse || toolUse.name !== STRUCTURED_RESPONSE_TOOL) {
      throw new ProviderResponseError(
        `Expected a ${STRUCTURED_RESPONSE_TOOL} tool_use block from Anthropic` +
          (finishReason ? ` (stop reason: ${finishReason})` : ""),
        usage
      );
    }
    return {
      content: toolUse.input,
      finishReason,
      ...(usage ? { usage } : {}),
      ...(model ? { model } : {}),
    };
  }

  const dialect = openAIChatDialect(provider);
  if (!dialect) {
    throw new ProviderResponseError(
      `parseProviderResponse does not support provider "${provider}"`
    );
  }
  const usage = body.usage ? compatibleUsage(body.usage) : undefined;
  const errorMessage = providerErrorMessage(body);
  if (errorMessage) throw new ProviderResponseError(errorMessage, usage);

  const choice = Array.isArray(body.choices)
    ? (body.choices[0] as Record<string, unknown> | undefined)
    : undefined;
  const finishReason = normalizeFinishReason(choice?.finish_reason) ?? null;
  // The forced call first; content as the fallback (and Cohere's only form).
  const text =
    openAIStructuredText(choice?.message, dialect.structuredOutput) ??
    openAIStructuredText(choice?.message, "response_format");
  if (text === undefined) {
    throw new ProviderResponseError(
      `Expected a ${STRUCTURED_RESPONSE_TOOL} function call or JSON content from ${provider}` +
        (finishReason ? ` (finish reason: ${finishReason})` : ""),
      usage
    );
  }

  // A truncated answer is cut off mid-value. Hand back the text unread, as
  // the adapters do, so the caller can tell "cut short" from "malformed" --
  // the tolerant reader would otherwise close the brackets and pass a partial
  // answer off as whole.
  let content: unknown = text;
  if (finishReason !== "length") {
    try {
      content = parseModelJson(text).value;
    } catch (error) {
      throw new ProviderResponseError(
        `Model returned unparseable JSON (${error instanceof Error ? error.message : String(error)}). First 300 chars: ${text.slice(0, 300)}`,
        usage
      );
    }
  }
  return {
    content,
    finishReason,
    ...(usage ? { usage } : {}),
    ...(model ? { model } : {}),
  };
}
