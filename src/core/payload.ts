/**
 * @fileoverview Provider request bodies, built once
 * @description The JSON body each provider family is sent. The SDK adapters
 * send these (with only transport fields such as `stream` added), and
 * `buildProviderRequest` returns them to callers that make the call
 * themselves -- so a body previewed by `/prompt` is the body invoke sends.
 *
 * Pure: no SDK, no `Buffer`, no Node built-ins.
 */

import type { JsonSchema, LlmProvider, MediaContent } from "../types/index.js";
import { getModelCapabilities } from "../config/providers.js";
import { getOpenAIAudioFormat } from "../lib/media-constants.js";
import { cohereResponseFormat } from "../services/llm/cohere-schema.js";

/** Name of the function/tool that carries the structured answer. */
export const STRUCTURED_RESPONSE_TOOL = "structured_response";

const STRUCTURED_RESPONSE_DESCRIPTION =
  "Generate structured response matching the schema";

/** The parts of an LLM call that decide its body. Same names as `LLMRequest`. */
export interface ChatBodyInput {
  /** Resolved model id. Callers apply their own default first. */
  model: string;
  /** User prompt. */
  prompt: string;
  systemPrompt?: string;
  outputSchema: JsonSchema | Record<string, unknown>;
  temperature?: number;
  maxTokens?: number;
  /** Input media (images, audio). Placeholders already in `prompt`. */
  media?: MediaContent[];
  /** What media the endpoint generates (only audio is honoured, OpenAI only). */
  expectsMediaOutput?: { image?: boolean; audio?: boolean; video?: boolean };
}

// =============================================================================
// OpenAI chat completions and compatible APIs
// =============================================================================

/**
 * How a provider is made to return the endpoint's schema.
 *
 * - `tool_call`: a `structured_response` function with a forced `tool_choice`.
 * - `response_format`: Cohere's `{ type: "json_object", schema }`, read from
 *   the message content. Cohere's compatibility API does not accept
 *   `tool_choice`, so a tool call there could not be forced.
 */
export type StructuredOutputMode = "tool_call" | "response_format";

/**
 * What a provider calls its cap on generated tokens.
 *
 * OpenAI's own newer models reject `max_tokens` outright, while the
 * OpenAI-*compatible* third parties know only `max_tokens`.
 */
export type TokenLimitParam = "max_tokens" | "max_completion_tokens";

/**
 * OpenAI models that reject `max_tokens`: the reasoning-era families, matched
 * on the family prefix so the next member is covered the day it ships.
 */
const REQUIRES_MAX_COMPLETION_TOKENS = /^(gpt-5|o[1-9])/i;

/** What to call the output cap for this provider and model. */
export function tokenLimitParamFor(
  isOpenAi: boolean,
  model: string
): TokenLimitParam {
  return isOpenAi && REQUIRES_MAX_COMPLETION_TOKENS.test(model)
    ? "max_completion_tokens"
    : "max_tokens";
}

/** Per-provider quirks of the OpenAI chat-completions dialect. */
export interface OpenAIChatDialect {
  /**
   * Send DeepSeek's `thinking: { type: "disabled" }`. DeepSeek V4 reasons by
   * default, and thinking mode rejects `tool_choice`.
   */
  disableThinking?: boolean;
  /** True for OpenAI itself: its newer models need `max_completion_tokens`. */
  isOpenAi?: boolean;
  /** How to request structured output. Default: `tool_call`. */
  structuredOutput?: StructuredOutputMode;
  /**
   * Send the user turn as a plain string rather than content parts, and no
   * media. Groq's chat models take text only.
   */
  plainUserContent?: boolean;
}

/**
 * The dialect each OpenAI-compatible provider speaks, or undefined for a
 * provider that does not use this body (Anthropic, Gemini, Jev, LM Studio).
 */
export function openAIChatDialect(
  provider: LlmProvider
): OpenAIChatDialect | undefined {
  switch (provider) {
    case "openai":
      return { isOpenAi: true };
    case "deepseek":
      return { disableThinking: true };
    case "cohere":
      return { structuredOutput: "response_format" };
    case "groq":
      return { plainUserContent: true };
    case "mistral":
    case "xai":
    case "perplexity":
    case "openrouter":
      return {};
    default:
      return undefined;
  }
}

/** Request fields that ask for structured output, in this dialect. */
function structuredOutputParams(
  outputSchema: ChatBodyInput["outputSchema"],
  mode: StructuredOutputMode
): Record<string, unknown> {
  if (mode === "response_format") {
    return { response_format: cohereResponseFormat(outputSchema) };
  }
  return {
    tools: [
      {
        type: "function",
        function: {
          name: STRUCTURED_RESPONSE_TOOL,
          description: STRUCTURED_RESPONSE_DESCRIPTION,
          parameters: outputSchema as Record<string, unknown>,
        },
      },
    ],
    tool_choice: {
      type: "function",
      function: { name: STRUCTURED_RESPONSE_TOOL },
    },
  };
}

/** User content parts: media first, then the prompt text. */
function openAIUserContent(
  input: ChatBodyInput
): Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = [];
  for (const m of input.media ?? []) {
    if (m.type === "image") {
      parts.push({
        type: "image_url",
        image_url: {
          url:
            m.format === "base64"
              ? `data:${m.mimeType};base64,${m.data}`
              : m.data,
        },
      });
    }
    if (m.type === "audio") {
      // Format already validated at the capability validation layer
      parts.push({
        type: "input_audio",
        input_audio: { data: m.data, format: getOpenAIAudioFormat(m.mimeType) },
      });
    }
  }
  parts.push({ type: "text", text: input.prompt });
  return parts;
}

/**
 * The chat-completions body for an OpenAI-compatible provider.
 *
 * `temperature` defaults to 0, as every OpenAI-compatible adapter always sent.
 */
export function buildOpenAIChatBody(
  input: ChatBodyInput,
  dialect: OpenAIChatDialect = {}
): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  if (input.systemPrompt) {
    messages.push({ role: "system", content: input.systemPrompt });
  }
  messages.push({
    role: "user",
    content: dialect.plainUserContent ? input.prompt : openAIUserContent(input),
  });

  // Audio output only when the endpoint expects it and the model makes it.
  // V1: voice and format are fixed.
  const audioOutput =
    !dialect.plainUserContent &&
    input.expectsMediaOutput?.audio === true &&
    getModelCapabilities(input.model).audioOutput === true;

  return {
    model: input.model,
    messages,
    ...(audioOutput
      ? {
          modalities: ["text", "audio"],
          audio: { voice: "alloy", format: "mp3" },
        }
      : {}),
    ...structuredOutputParams(
      input.outputSchema,
      dialect.structuredOutput ?? "tool_call"
    ),
    // Off where the model reasons by default: thinking mode rejects
    // `tool_choice` outright, so this is what keeps the forced call legal.
    ...(dialect.disableThinking ? { thinking: { type: "disabled" } } : {}),
    temperature: input.temperature ?? 0,
    // Always present, `undefined` when there is no cap: the SDKs and
    // JSON.stringify drop it, so nothing reaches the wire.
    [tokenLimitParamFor(dialect.isOpenAi ?? false, input.model)]:
      input.maxTokens,
  };
}

// =============================================================================
// Anthropic Messages API
// =============================================================================

/** Default `max_tokens` for Anthropic, which requires the field. */
export const ANTHROPIC_DEFAULT_MAX_TOKENS = 4096;

/**
 * The system prompt as one cacheable block.
 *
 * A breakpoint on the system block caches the tools and the system prompt --
 * the part that repeats from call to call. An empty prompt yields no block:
 * Anthropic rejects an empty text block.
 */
function anthropicSystemBlocks(
  systemPrompt: string | undefined
): Array<Record<string, unknown>> | undefined {
  if (!systemPrompt) return undefined;
  return [
    { type: "text", text: systemPrompt, cache_control: { type: "ephemeral" } },
  ];
}

/**
 * The Messages API body: a forced `structured_response` tool, images first in
 * the user turn (Claude takes no audio/video), and `temperature` only when the
 * caller set one -- Opus 4.7+, Sonnet 5 and Fable 5 reject it with a 400.
 */
export function buildAnthropicMessagesBody(
  input: ChatBodyInput
): Record<string, unknown> {
  const userContent: Array<Record<string, unknown>> = [];
  for (const m of input.media ?? []) {
    if (m.type !== "image") continue;
    if (m.format === "base64") {
      userContent.push({
        type: "image",
        source: { type: "base64", media_type: m.mimeType, data: m.data },
      });
    } else if (m.format === "url") {
      userContent.push({ type: "image", source: { type: "url", url: m.data } });
    }
  }
  userContent.push({ type: "text", text: input.prompt });

  const system = anthropicSystemBlocks(input.systemPrompt);
  return {
    model: input.model,
    max_tokens: input.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
    ...(system ? { system } : {}),
    messages: [{ role: "user", content: userContent }],
    tools: [
      {
        name: STRUCTURED_RESPONSE_TOOL,
        description: STRUCTURED_RESPONSE_DESCRIPTION,
        input_schema: input.outputSchema,
      },
    ],
    tool_choice: { type: "tool", name: STRUCTURED_RESPONSE_TOOL },
    ...(input.temperature !== undefined
      ? { temperature: input.temperature }
      : {}),
  };
}
