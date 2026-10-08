/**
 * @fileoverview Anthropic LLM provider
 * @description Implements the ILLMProvider interface for Anthropic Claude models.
 * Uses tool_use for structured output extraction. Supports image input
 * via base64 and URL formats.
 */

import Anthropic from "@anthropic-ai/sdk";
import type {
  ILLMProvider,
  LLMRequest,
  LLMResponse,
  ProviderConfig,
} from "./types.js";
import { normalizeFinishReason } from "./finish-reason.js";
import { attachUsage } from "./usage-error.js";
import { buildAnthropicMessagesBody } from "../../core/payload.js";
import { anthropicUsage, findAnthropicToolUse } from "../../core/response.js";

// Fallback when an endpoint doesn't pin a model. claude-sonnet-4-20250514 is
// deprecated (retires 2026-06-15); use a current, non-deprecated Sonnet-tier id.
// (Bump to "claude-sonnet-5" or "claude-opus-4-8" for the latest tier.)
const DEFAULT_MODEL = "claude-sonnet-4-6";

export { anthropicUsage } from "../../core/response.js";

export class AnthropicProvider implements ILLMProvider {
  readonly providerName = "anthropic" as const;
  private client: Anthropic;
  private defaultModel: string;

  constructor(config: ProviderConfig) {
    if (!config.apiKey) {
      throw new Error("Anthropic API key is required");
    }
    this.client = new Anthropic({ apiKey: config.apiKey });
    this.defaultModel = config.model ?? DEFAULT_MODEL;
  }

  async generate(request: LLMRequest): Promise<LLMResponse> {
    const model = request.model ?? this.defaultModel;
    const startTime = Date.now();

    // The same body buildApiPayload() and /prompt describe.
    const response = (await this.client.messages.create(
      buildAnthropicMessagesBody({
        ...request,
        model,
      }) as unknown as Anthropic.MessageCreateParamsNonStreaming
    )) as Anthropic.Message;

    const latencyMs = Date.now() - startTime;

    const usage = anthropicUsage(response.usage);

    // Extract structured response from tool use
    const toolUseBlock = findAnthropicToolUse(response.content);

    if (!toolUseBlock || toolUseBlock.name !== "structured_response") {
      throw attachUsage(
        new Error("Expected tool_use response from Anthropic"),
        usage,
        response.model
      );
    }

    const content = toolUseBlock.input;
    const rawResponse = JSON.stringify(content);

    return {
      content,
      rawResponse,
      usage,
      model: response.model,
      provider: this.providerName,
      latencyMs,
      finishReason: normalizeFinishReason(response.stop_reason),
    };
  }

  buildApiPayload(request: LLMRequest): Record<string, unknown> {
    return buildAnthropicMessagesBody({
      ...request,
      model: request.model ?? this.defaultModel,
    });
  }
}
