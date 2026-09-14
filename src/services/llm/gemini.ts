/**
 * @fileoverview Google Gemini LLM provider
 * @description Implements the ILLMProvider interface for Gemini models.
 * Uses native responseSchema for structured JSON output. Supports multimodal
 * input (images, audio, video) via base64 and GCS URLs. Includes stubs
 * for Imagen image generation and Veo video generation (requires Vertex AI SDK).
 */

import {
  GoogleGenerativeAI,
  type GenerationConfig,
  type Part,
} from "@google/generative-ai";
import type {
  ILLMProvider,
  LLMRequest,
  LLMResponse,
  LLMUsage,
  ProviderConfig,
} from "./types.js";
import { isGenerativeModel } from "../../lib/capability-validator.js";
import { normalizeFinishReason } from "./finish-reason.js";
import { attachUsage } from "./usage-error.js";

const DEFAULT_MODEL = "gemini-2.5-flash";

/**
 * `usageMetadata` as the REST API returns it. The SDK version in use types
 * only the first three counts, but the rest arrive in the same object.
 */
interface GeminiUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  thoughtsTokenCount?: number;
  cachedContentTokenCount?: number;
  promptTokensDetails?: { modality?: string; tokenCount?: number }[];
  cacheTokensDetails?: { modality?: string; tokenCount?: number }[];
}

function modalityTokens(
  details: GeminiUsageMetadata["promptTokensDetails"],
  modality: string
): number {
  return (details ?? [])
    .filter(d => d.modality === modality)
    .reduce((sum, d) => sum + (d.tokenCount ?? 0), 0);
}

/**
 * Token usage for a Gemini response.
 *
 * `candidatesTokenCount` excludes thinking, which Gemini reports separately as
 * `thoughtsTokenCount` and bills at the output rate. Gemini 3 models think by
 * default, so reading candidates alone under-prices every call.
 */
export function geminiUsage(metadata: unknown): LLMUsage {
  const m = (metadata ?? {}) as GeminiUsageMetadata;
  const promptTokens = m.promptTokenCount ?? 0;
  const thoughts = m.thoughtsTokenCount ?? 0;
  const completionTokens = (m.candidatesTokenCount ?? 0) + thoughts;
  const cachedAudio = modalityTokens(m.cacheTokensDetails, "AUDIO");
  const audio = Math.max(
    0,
    modalityTokens(m.promptTokensDetails, "AUDIO") - cachedAudio
  );
  return {
    promptTokens,
    completionTokens,
    totalTokens: m.totalTokenCount ?? promptTokens + completionTokens,
    ...(m.cachedContentTokenCount
      ? { cachedInputTokens: m.cachedContentTokenCount }
      : {}),
    ...(audio ? { audioInputTokens: audio } : {}),
    ...(thoughts ? { reasoningTokens: thoughts } : {}),
  };
}

export class GeminiProvider implements ILLMProvider {
  readonly providerName = "gemini" as const;
  private genAI: GoogleGenerativeAI;
  private defaultModel: string;

  constructor(config: ProviderConfig) {
    if (!config.apiKey) {
      throw new Error("Gemini API key is required");
    }
    this.genAI = new GoogleGenerativeAI(config.apiKey);
    this.defaultModel = config.model ?? DEFAULT_MODEL;
  }

  async generate(request: LLMRequest): Promise<LLMResponse> {
    const modelName = request.model ?? this.defaultModel;
    const startTime = Date.now();

    // Check if this is a generative model (Imagen, Veo)
    if (isGenerativeModel(modelName)) {
      return this.generateMedia(request, modelName, startTime);
    }

    // Build multimodal parts
    const parts: Part[] = [];

    if (request.media?.length) {
      for (const m of request.media) {
        if (m.format === "base64") {
          parts.push({
            inlineData: { mimeType: m.mimeType, data: m.data },
          });
        } else if (m.format === "url") {
          // Only gs:// URLs are allowed (validated earlier)
          parts.push({
            fileData: { mimeType: m.mimeType, fileUri: m.data },
          });
        }
      }
    }

    // Add text prompt
    parts.push({ text: request.prompt });

    // Create model with system instruction
    const model = this.genAI.getGenerativeModel({
      model: modelName,
      systemInstruction: request.systemPrompt,
    });

    // Configure for JSON output with schema
    const generationConfig: GenerationConfig = {
      responseMimeType: "application/json",
      responseSchema: this.convertToGeminiSchema(request.outputSchema),
      temperature: request.temperature ?? 0,
      maxOutputTokens: request.maxTokens,
    };

    const result = await model.generateContent({
      contents: [{ role: "user", parts }],
      generationConfig,
    });

    const latencyMs = Date.now() - startTime;

    const response = result.response;
    const usage = geminiUsage(response.usageMetadata);
    const finishReason = normalizeFinishReason(
      response.candidates?.[0]?.finishReason
    );
    let rawResponse: string;
    try {
      // text() throws when the candidate was blocked; the prompt was billed.
      rawResponse = response.text();
    } catch (error) {
      throw attachUsage(error, usage, modelName);
    }

    // Read the stop reason before parsing, as openai.ts does: JSON cut off at
    // the output ceiling is a truncation, not a malformed model.
    if (finishReason === "length") {
      return {
        content: rawResponse,
        rawResponse,
        usage,
        model: modelName,
        provider: this.providerName,
        latencyMs,
        finishReason,
      };
    }

    let content: unknown;
    try {
      content = JSON.parse(rawResponse);
    } catch (error) {
      throw attachUsage(error, usage, modelName);
    }

    return {
      content,
      rawResponse,
      usage,
      model: modelName,
      provider: this.providerName,
      latencyMs,
      finishReason,
    };
  }

  /**
   * Generate media (images or video) using Imagen or Veo models.
   * Note: This is a simplified implementation - production would use Vertex AI SDK.
   */
  private async generateMedia(
    _request: LLMRequest,
    modelName: string,
    startTime: number
  ): Promise<LLMResponse> {
    // Note: Imagen and Veo require Vertex AI SDK, not the basic genai SDK.
    // This is a placeholder that shows the structure.
    // In production, you would:
    // 1. Use @google-cloud/vertexai package
    // 2. Call imagen.generateImages() or veo.generateVideo()
    // 3. Handle the response appropriately

    if (modelName.includes("imagen")) {
      // Placeholder for Imagen implementation
      console.warn(
        "Imagen generation requires Vertex AI SDK - not implemented in v1"
      );

      // For now, return a structured response indicating the limitation
      return {
        content: {
          error: "Image generation requires Vertex AI SDK",
          model: modelName,
        },
        rawResponse: JSON.stringify({ error: "Not implemented" }),
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        model: modelName,
        provider: this.providerName,
        latencyMs: Date.now() - startTime,
        generatedMedia: undefined,
      };
    }

    if (modelName.includes("veo")) {
      // Placeholder for Veo implementation
      console.warn(
        "Veo generation requires Vertex AI SDK - not implemented in v1"
      );

      return {
        content: {
          error: "Video generation requires Vertex AI SDK",
          model: modelName,
        },
        rawResponse: JSON.stringify({ error: "Not implemented" }),
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        model: modelName,
        provider: this.providerName,
        latencyMs: Date.now() - startTime,
        generatedMedia: undefined,
      };
    }

    throw new Error(`Unknown generative model: ${modelName}`);
  }

  /**
   * Convert standard JSON Schema to Gemini's schema format
   */
  private convertToGeminiSchema(
    jsonSchema: Record<string, unknown>
  ): Record<string, unknown> {
    // Gemini mostly accepts standard JSON Schema, but may need adjustments
    const geminiSchema = { ...jsonSchema };

    // Remove unsupported keywords
    const unsupportedKeywords = ["$schema", "$id", "definitions", "$defs"];
    for (const keyword of unsupportedKeywords) {
      delete geminiSchema[keyword];
    }

    return geminiSchema;
  }

  buildApiPayload(request: LLMRequest): Record<string, unknown> {
    const modelName = request.model ?? this.defaultModel;

    // Build multimodal parts
    const parts: Array<Record<string, unknown>> = [];

    if (request.media?.length) {
      for (const m of request.media) {
        if (m.format === "base64") {
          parts.push({
            inlineData: { mimeType: m.mimeType, data: m.data },
          });
        } else if (m.format === "url") {
          parts.push({
            fileData: { mimeType: m.mimeType, fileUri: m.data },
          });
        }
      }
    }

    parts.push({ text: request.prompt });

    return {
      model: modelName,
      contents: [{ parts }],
      systemInstruction: request.systemPrompt
        ? { parts: [{ text: request.systemPrompt }] }
        : undefined,
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema: this.convertToGeminiSchema(request.outputSchema),
        temperature: request.temperature ?? 0,
        maxOutputTokens: request.maxTokens,
      },
    };
  }
}
