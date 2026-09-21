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
  LLMUsage,
  ProviderConfig,
} from "./types.js";
import { normalizeFinishReason } from "./finish-reason.js";
import { attachUsage } from "./usage-error.js";

/**
 * The system prompt as one cacheable block.
 *
 * Anthropic caches nothing unless asked. A breakpoint on the system block
 * caches everything before it, which is the tools and the system prompt -- the
 * part of a request that stays identical from one call to the next while the
 * user message changes. A cache write bills at 1.25x the input rate and a read
 * at 0.1x, so the first call to repeat within the five-minute window already
 * comes out ahead; a prompt below the model's minimum (1024 tokens for most,
 * 4096 for Opus 4.5+ and Haiku 4.5) is simply not cached and costs nothing
 * extra.
 *
 * An empty prompt yields no block at all: Anthropic rejects an empty text
 * block, and `cache_control` on one is worse.
 */
function systemBlocks(
  systemPrompt: string | undefined
): Anthropic.TextBlockParam[] | undefined {
  if (!systemPrompt) return undefined;
  return [
    {
      type: "text",
      text: systemPrompt,
      cache_control: { type: "ephemeral" },
    },
  ];
}

// Fallback when an endpoint doesn't pin a model. claude-sonnet-4-20250514 is
// deprecated (retires 2026-06-15); use a current, non-deprecated Sonnet-tier id.
// (Bump to "claude-sonnet-5" or "claude-opus-4-8" for the latest tier.)
const DEFAULT_MODEL = "claude-sonnet-4-6";

/**
 * Anthropic reports `input_tokens` as only the tokens neither read from nor
 * written to the cache, so the whole prompt is the three added together. The
 * cached and written parts stay named, as `LLMUsage` defines them: parts of
 * `promptTokens`, priced apart.
 */
export function anthropicUsage(usage: Anthropic.Usage): LLMUsage {
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

    // Use tool_use for structured output
    const tools: Anthropic.Tool[] = [
      {
        name: "structured_response",
        description: "Generate structured response matching the schema",
        input_schema: request.outputSchema as Anthropic.Tool.InputSchema,
      },
    ];

    // Build multimodal content blocks
    const userContent: Anthropic.ContentBlockParam[] = [];

    // Add media blocks first (images only for Claude)
    if (request.media?.length) {
      for (const m of request.media) {
        if (m.type === "image") {
          if (m.format === "base64") {
            userContent.push({
              type: "image",
              source: {
                type: "base64",
                media_type: m.mimeType as
                  "image/jpeg" | "image/png" | "image/gif" | "image/webp",
                data: m.data,
              },
            });
          } else if (m.format === "url") {
            userContent.push({
              type: "image",
              source: {
                type: "url",
                url: m.data,
              },
            });
          }
        }
        // Claude doesn't support audio/video input currently
      }
    }

    // Add text prompt
    userContent.push({ type: "text", text: request.prompt });

    const response = await this.client.messages.create({
      model,
      max_tokens: request.maxTokens ?? 4096,
      ...(systemBlocks(request.systemPrompt)
        ? { system: systemBlocks(request.systemPrompt) }
        : {}),
      messages: [{ role: "user", content: userContent }],
      tools,
      tool_choice: { type: "tool", name: "structured_response" },
      // `temperature`/`top_p` are rejected (400) on Opus 4.7+, Sonnet 5, and
      // Fable 5. Only send it when a caller explicitly requests one; older
      // models still accept it. Omitting it lets current models work.
      ...(request.temperature !== undefined
        ? { temperature: request.temperature }
        : {}),
    });

    const latencyMs = Date.now() - startTime;

    const usage = anthropicUsage(response.usage);

    // Extract structured response from tool use
    const toolUseBlock = response.content.find(
      (block): block is Anthropic.ToolUseBlock => block.type === "tool_use"
    );

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
    const model = request.model ?? this.defaultModel;

    // Build multimodal content blocks
    const userContent: Anthropic.ContentBlockParam[] = [];

    // Add media blocks first (images only for Claude)
    if (request.media?.length) {
      for (const m of request.media) {
        if (m.type === "image") {
          if (m.format === "base64") {
            userContent.push({
              type: "image",
              source: {
                type: "base64",
                media_type: m.mimeType as
                  "image/jpeg" | "image/png" | "image/gif" | "image/webp",
                data: m.data,
              },
            });
          } else if (m.format === "url") {
            userContent.push({
              type: "image",
              source: {
                type: "url",
                url: m.data,
              },
            });
          }
        }
      }
    }

    // Add text prompt
    userContent.push({ type: "text", text: request.prompt });

    return {
      model,
      max_tokens: request.maxTokens ?? 4096,
      ...(systemBlocks(request.systemPrompt)
        ? { system: systemBlocks(request.systemPrompt) }
        : {}),
      messages: [{ role: "user", content: userContent }],
      tools: [
        {
          name: "structured_response",
          description: "Generate structured response matching the schema",
          input_schema: request.outputSchema,
        },
      ],
      tool_choice: { type: "tool", name: "structured_response" },
      // See generate(): only include temperature when explicitly requested.
      ...(request.temperature !== undefined
        ? { temperature: request.temperature }
        : {}),
    };
  }
}
