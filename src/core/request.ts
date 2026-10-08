/**
 * @fileoverview A provider call, described for someone else to make
 * @description `buildProviderRequest` returns the URL, headers, auth slot and
 * body of the exact call invoke makes, for a caller holding its own provider
 * key. The key is never part of it.
 */

import type { AiProviderRequest, LlmProvider } from "../types/index.js";
import { getProviderById } from "../config/providers.js";
import { isTranscriptionModel } from "../lib/capability-validator.js";
import { ANTHROPIC_VERSION, PROVIDER_ENDPOINTS } from "./endpoints.js";
import {
  buildAnthropicMessagesBody,
  buildOpenAIChatBody,
  openAIChatDialect,
  type ChatBodyInput,
} from "./payload.js";

/** Input for {@link buildProviderRequest}. */
export interface ProviderRequestInput extends Omit<ChatBodyInput, "model"> {
  provider: LlmProvider;
  /** Model id. Omitted: the provider catalog's `defaultModel`. */
  model?: string;
}

/** Providers `buildProviderRequest` can describe. */
export const PROVIDER_REQUEST_SUPPORTED: readonly LlmProvider[] = [
  "openai",
  "anthropic",
  "deepseek",
  "openrouter",
  "mistral",
  "xai",
  "perplexity",
  "cohere",
  "groq",
];

/** The catalog's default model for a provider, or undefined if it has none. */
export function defaultModelFor(provider: LlmProvider): string | undefined {
  return getProviderById(provider)?.defaultModel;
}

/**
 * Describe the provider call invoke would make for these prompts.
 *
 * @throws Error for a provider whose call cannot be described as one JSON
 *   POST: Gemini (SDK-only here), Jev, LM Studio (no fixed URL), and Groq's
 *   Whisper transcription models.
 */
export function buildProviderRequest(
  input: ProviderRequestInput
): AiProviderRequest {
  const { provider } = input;
  const model = input.model ?? defaultModelFor(provider);
  if (!model) {
    throw new Error(`No default model for provider "${provider}"`);
  }
  const bodyInput: ChatBodyInput = { ...input, model };

  if (provider === "anthropic") {
    return {
      provider,
      model,
      method: "POST",
      url: PROVIDER_ENDPOINTS.anthropic,
      headers: {
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
      },
      auth: { header: "x-api-key", prefix: "" },
      body: buildAnthropicMessagesBody(bodyInput),
    };
  }

  const dialect = openAIChatDialect(provider);
  if (dialect && !(provider === "groq" && isTranscriptionModel(model))) {
    return {
      provider,
      model,
      method: "POST",
      url: PROVIDER_ENDPOINTS[provider],
      headers: { "content-type": "application/json" },
      auth: { header: "Authorization", prefix: "Bearer " },
      body: buildOpenAIChatBody(bodyInput, dialect),
    };
  }

  throw new Error(
    `buildProviderRequest does not support provider "${provider}"` +
      (provider === "groq" ? ` with transcription model "${model}"` : "") +
      `. Supported: ${PROVIDER_REQUEST_SUPPORTED.join(", ")}`
  );
}
