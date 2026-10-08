/**
 * @fileoverview OpenAI LLM provider
 * @description Implements the ILLMProvider interface for OpenAI models.
 * Also used by Mistral, xAI, DeepSeek, Perplexity, and Cohere
 * (OpenAI-compatible APIs). Structured output is a forced function call,
 * except for Cohere, whose compatibility API has no `tool_choice` and uses
 * `response_format` instead. Supports multimodal input (images, audio) and
 * audio output generation.
 */

import OpenAI from "openai";
import type {
  FinishReason,
  GeneratedMedia,
  JsonSchema,
} from "../../types/index.js";
import {
  requiresEntityId,
  type ILLMProvider,
  type LLMRequest,
  type LLMResponse,
  type LLMUsage,
  type ProviderConfig,
} from "./types.js";
import {
  buildOutputStructureSection,
  buildResponseFormatSection,
  buildWebSearchRoutingSection,
} from "../../lib/prompt-builder.js";
import { normalizeFinishReason } from "./finish-reason.js";
import { addUsage, compatibleUsage } from "./compatible-usage.js";
import { attachUsage, getFailedInvocationUsage } from "./usage-error.js";
import { readModelJson } from "./json-repair.js";
import {
  buildOpenAIChatBody,
  type OpenAIChatDialect,
  type StructuredOutputMode,
} from "../../core/payload.js";
import { openAIStructuredText } from "../../core/response.js";

/** Usage from a Responses API response, which names its fields differently. */
function responsesUsage(response: OpenAI.Responses.Response): LLMUsage {
  const inputTokens = response.usage?.input_tokens ?? 0;
  const outputTokens = response.usage?.output_tokens ?? 0;
  const inputDetails = response.usage?.input_tokens_details as
    { cached_tokens?: number; cache_write_tokens?: number } | undefined;
  const cached = inputDetails?.cached_tokens ?? 0;
  // GPT-5.6+ bills cache writes at 1.25x input; older SDK types omit the field.
  const cacheWrite = inputDetails?.cache_write_tokens ?? 0;
  const reasoning =
    response.usage?.output_tokens_details?.reasoning_tokens ?? 0;
  // Only `search` actions carry the tool fee; `open_page` and `find_in_page`
  // are web_search_call items too, but free.
  const searches = response.output.filter(
    item =>
      item.type === "web_search_call" &&
      ((item as { action?: { type?: string } }).action?.type ?? "search") ===
        "search"
  ).length;
  return {
    promptTokens: inputTokens,
    completionTokens: outputTokens,
    totalTokens: response.usage?.total_tokens ?? inputTokens + outputTokens,
    ...(cached ? { cachedInputTokens: cached } : {}),
    ...(cacheWrite ? { cacheWriteInputTokens: cacheWrite } : {}),
    ...(reasoning ? { reasoningTokens: reasoning } : {}),
    ...(searches ? { searchCalls: searches } : {}),
  };
}

const DEFAULT_MODEL = "gpt-4o-mini";

export {
  tokenLimitParamFor,
  type StructuredOutputMode,
  type TokenLimitParam,
} from "../../core/payload.js";

class OpenAIProviderError extends Error {
  details?: Record<string, unknown>;

  constructor(message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "OpenAIProviderError";
    this.details = details;
  }
}

export class OpenAIProvider implements ILLMProvider {
  readonly providerName = "openai" as const;
  private client: OpenAI;
  private defaultModel: string;

  /**
   * This provider's quirks of the chat-completions dialect (DeepSeek's
   * thinking switch, OpenAI's token-cap name, Cohere's response_format). The
   * body itself is built by `buildOpenAIChatBody` in `core/payload.ts`, the
   * same builder `/prompt` describes requests with.
   */
  private dialect: OpenAIChatDialect;

  constructor(
    config: ProviderConfig,
    /**
     * Per-provider quirks. Defaults keep every provider that works today
     * working; only one known to need otherwise opts out.
     */
    options: {
      /**
       * Turn the model's chain of thought off.
       *
       * DeepSeek V4 reasons by default, and that is measurably the wrong
       * trade here: the same two-bar plan cost 844 output tokens and 8.6s
       * thinking, against 126 tokens and 2.4s with it off — and the reasoning
       * is discarded, so the caller pays for working it never sees. Worse,
       * thinking mode *rejects* `tool_choice`, so leaving it on also costs the
       * structured-output guarantee.
       *
       * Sent as DeepSeek's native `thinking` object. Its type error on a
       * boolean — "thinking: invalid type: boolean `false`, expected .." — is
       * what identified the parameter.
       */
      disableThinking?: boolean;
      /**
       * True for OpenAI itself. The OpenAI-compatible providers served by this
       * same class (DeepSeek, Mistral, xAI, Perplexity, Cohere) are not, and
       * only know `max_tokens`.
       */
      isOpenAi?: boolean;
      /** How to request structured output. Default: `tool_call`. */
      structuredOutput?: StructuredOutputMode;
    } = {}
  ) {
    this.dialect = {
      disableThinking: options.disableThinking ?? false,
      isOpenAi: options.isOpenAi ?? false,
      structuredOutput: options.structuredOutput ?? "tool_call",
    };
    if (!config.apiKey) {
      throw new Error("OpenAI API key is required");
    }
    // Honor a custom base URL so OpenAI-compatible providers (Mistral, xAI,
    // DeepSeek, Perplexity) reach their own API instead of api.openai.com.
    this.client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.endpointUrl ? { baseURL: config.endpointUrl } : {}),
    });
    this.defaultModel = config.model ?? DEFAULT_MODEL;
  }

  async generate(request: LLMRequest): Promise<LLMResponse> {
    console.log(
      `[llm] generate() called, webSearch=${request.webSearch}, model=${request.model}`
    );
    // Use Responses API when web search is enabled
    if (request.webSearch) {
      return this.generateWithSearch(request);
    }

    const model = request.model ?? this.defaultModel;
    const startTime = Date.now();
    const generatedMedia: GeneratedMedia[] = [];

    // The same body buildApiPayload() and /prompt describe; only the
    // transport flag is added here.
    const response = (await this.client.chat.completions.create({
      ...buildOpenAIChatBody({ ...request, model }, this.dialect),
      stream: false,
    } as unknown as OpenAI.Chat.ChatCompletionCreateParams)) as OpenAI.Chat.ChatCompletion;

    const latencyMs = Date.now() - startTime;

    // Extract audio from response if present
    const message = response.choices[0]
      ?.message as OpenAI.Chat.ChatCompletionMessage & {
      audio?: { data: string; format: string };
    };
    const audioData = message?.audio;
    if (audioData) {
      const audioMimeType = `audio/${audioData.format}`;

      if (requiresEntityId(request)) {
        // URL output requested but storage upload not implemented in v1
        // For now, return base64 with a warning in the logs
        console.warn(
          "URL output requested but storage upload not implemented. Returning base64."
        );
        generatedMedia.push({
          type: "audio",
          mimeType: audioMimeType,
          data: audioData.data,
        });
      } else {
        // Return base64
        generatedMedia.push({
          type: "audio",
          mimeType: audioMimeType,
          data: audioData.data,
        });
      }
    }

    const usage = compatibleUsage(response.usage);

    // Extract the structured response: the forced function call's arguments,
    // or the message content in response_format mode.
    const structuredText = openAIStructuredText(
      response.choices[0]?.message,
      this.dialect.structuredOutput
    );
    if (structuredText === undefined) {
      throw attachUsage(
        new Error(
          this.dialect.structuredOutput === "response_format"
            ? "Expected JSON message content from the provider"
            : "Expected function call response from OpenAI"
        ),
        usage,
        response.model
      );
    }
    const rawResponse = structuredText;

    const finishReason = normalizeFinishReason(
      response.choices[0]?.finish_reason
    );

    /*
      Why the stop reason is read BEFORE parsing.

      A model that hits the output ceiling returns arguments cut off mid-value,
      and `JSON.parse` then fails with something like "Expected ']'". Return the
      raw truncated arguments with finishReason instead, so the route can mark
      the response as truncated and preserve usage for the caller.
    */
    if (finishReason === "length") {
      return {
        content: rawResponse,
        rawResponse,
        usage,
        model: response.model,
        provider: this.providerName,
        latencyMs,
        finishReason,
        generatedMedia: generatedMedia.length > 0 ? generatedMedia : undefined,
      };
    }

    let content: unknown;
    try {
      content = readModelJson(rawResponse, `${this.providerName}/${model}`);
    } catch (error) {
      // Not truncated, and not repairable either: genuinely malformed. Carry a head of the payload:
      // "Unable to parse JSON string" alone leaves nothing to diagnose from.
      throw attachUsage(
        new Error(
          `Model returned unparseable JSON (${error instanceof Error ? error.message : String(error)}). First 300 chars: ${rawResponse.slice(0, 300)}`
        ),
        usage,
        response.model
      );
    }

    return {
      content,
      rawResponse,
      usage,
      model: response.model,
      provider: this.providerName,
      latencyMs,
      finishReason,
      generatedMedia: generatedMedia.length > 0 ? generatedMedia : undefined,
    };
  }

  /**
   * Generate using the OpenAI Responses API with web search enabled.
   * The model can search the web and then call the structured_response function.
   */
  // ---------------------------------------------------------------------------
  // Web-search-aware generation (3-step)
  // ---------------------------------------------------------------------------

  private static INPUT_LAYOUTS = new Set([
    "input_text",
    "input_numeric",
    "input_date",
    "input_email",
    "input_phone",
    "search",
    "line_select",
    "line_toggle",
    "line_slider",
  ]);

  /** Check if a renderable tree contains input controls (clarification). */
  private containsInputControls(data: unknown): boolean {
    if (typeof data !== "object" || data === null) return false;
    const obj = data as Record<string, unknown>;
    const view = obj.view as Record<string, unknown> | undefined;
    if (view?.layout && OpenAIProvider.INPUT_LAYOUTS.has(view.layout as string))
      return true;
    const children = (view?.children ?? obj.children) as unknown[] | undefined;
    if (Array.isArray(children)) {
      return children.some(child => this.containsInputControls(child));
    }
    return false;
  }

  /**
   * Wrap the user's output schema in a container that lets the AI signal
   * whether it needs a web search before it can answer.
   */
  private buildTriageSchema(
    userSchema: Record<string, unknown>
  ): Record<string, unknown> {
    return {
      type: "object",
      required: ["user_data"],
      properties: {
        user_data: userSchema,
        web_search_needed: {
          type: "boolean",
          default: false,
          description:
            "Set to true when you would say 'I don't have real-time data' or " +
            "'I cannot access current listings'. The system WILL perform a web search " +
            "and provide you the results. Set to false when asking for clarification " +
            "or when you can answer from training data.",
        },
      },
      additionalProperties: false,
    };
  }

  /**
   * Build the system prompt for step 1: try to answer fully, signal
   * web_search_needed only if real-time data is genuinely required.
   */

  /**
   * Call the Responses API with a forced function call.
   * Used for both the triage step and the structuring step.
   */
  private async callResponsesStructured(
    model: string,
    input: OpenAI.Responses.ResponseInputItem[],
    schema: Record<string, unknown>,
    temperature: number,
    maxTokens?: number
  ): Promise<{
    content: unknown;
    rawResponse: string;
    usage: LLMUsage;
    model: string;
    finishReason?: FinishReason;
  }> {
    console.log(
      `[web-search] callResponsesStructured() calling Responses API, model=${model}`
    );
    const response = await this.client.responses.create({
      model,
      input,
      tools: [
        {
          type: "function",
          name: "structured_response",
          description: "Generate structured response matching the schema",
          parameters: schema,
          strict: false,
        },
      ],
      tool_choice: {
        type: "function",
        name: "structured_response",
      },
      temperature,
      max_output_tokens: maxTokens,
    });

    const usage = responsesUsage(response);
    const functionCall = response.output.find(
      (item): item is OpenAI.Responses.ResponseFunctionToolCall =>
        item.type === "function_call" && item.name === "structured_response"
    );

    if (!functionCall) {
      throw attachUsage(
        new OpenAIProviderError(
          "Expected structured_response function call from Responses API",
          {
            model,
            responseId: response.id,
            outputItems: response.output.map(item => ({
              type: item.type,
              ...("name" in item ? { name: item.name } : {}),
            })),
          }
        ),
        usage,
        response.model
      );
    }

    let content: unknown;
    try {
      content = readModelJson(functionCall.arguments, this.providerName);
    } catch (error) {
      throw attachUsage(error, usage, response.model);
    }

    return {
      content,
      rawResponse: functionCall.arguments,
      usage,
      model: response.model,
      // The Responses API reports truncation as an incomplete status with a
      // reason, rather than a finish_reason on the choice.
      finishReason:
        response.status === "incomplete"
          ? normalizeFinishReason(response.incomplete_details?.reason)
          : "stop",
    };
  }

  /**
   * Search the web via the Responses API and return a text summary.
   */
  private async searchWeb(
    model: string,
    query: string,
    temperature: number
  ): Promise<{
    summary: string;
    usage: LLMUsage;
  }> {
    console.log(
      `[web-search] searchWeb() calling Responses API for query: "${query.substring(0, 100)}..."`
    );
    const response = await this.client.responses.create({
      model,
      input: [
        {
          role: "developer" as const,
          content:
            "Search the web for the requested information. " +
            "Return a thorough summary of the results with specific details, " +
            "names, addresses, dates, and URLs when available.",
        },
        { role: "user" as const, content: query },
      ],
      tools: [{ type: "web_search_preview" }],
      temperature,
    });

    return {
      summary: response.output_text ?? "",
      usage: responsesUsage(response),
    };
  }

  /** Build Responses API input from an LLM request. */
  private buildResponsesInput(
    systemPrompt: string | undefined,
    userPrompt: string,
    media?: LLMRequest["media"]
  ): OpenAI.Responses.ResponseInputItem[] {
    const input: OpenAI.Responses.ResponseInputItem[] = [];

    if (systemPrompt) {
      input.push({ role: "developer" as const, content: systemPrompt });
    }

    const userContent: OpenAI.Responses.ResponseInputContent[] = [];
    if (media?.length) {
      for (const m of media) {
        if (m.type === "image") {
          userContent.push({
            type: "input_image",
            image_url:
              m.format === "base64"
                ? `data:${m.mimeType};base64,${m.data}`
                : m.data,
            detail: "auto",
          });
        }
      }
    }
    userContent.push({ type: "input_text", text: userPrompt });
    input.push({ role: "user" as const, content: userContent });

    return input;
  }

  /**
   * Main entry point when web search is enabled on the endpoint.
   *
   * 1. Try to answer via Responses API with triage wrapper schema.
   *    AI provides full answer OR signals web_search_needed.
   * 2. If no search needed → return user_data directly.
   * 3. If search needed → web search → structure results.
   */
  private async generateWithSearch(request: LLMRequest): Promise<LLMResponse> {
    /*
      Up to three billed calls make one invocation. Each step's usage is added
      to the total only once that step succeeds, so a failure part-way reports
      the completed steps plus whatever the failing call attached -- counted
      once -- instead of pricing the whole search as free.
    */
    let total: LLMUsage | undefined;
    const record = (usage: LLMUsage): LLMUsage =>
      (total = total ? addUsage(total, usage) : usage);
    try {
      return await this.runSearchSteps(request, record);
    } catch (error) {
      const failed = getFailedInvocationUsage(error);
      const consumed = failed
        ? total
          ? addUsage(total, failed.usage)
          : failed.usage
        : total;
      if (consumed) {
        throw attachUsage(
          error,
          consumed,
          failed?.model ?? request.model ?? this.defaultModel
        );
      }
      throw error;
    }
  }

  private async runSearchSteps(
    request: LLMRequest,
    record: (usage: LLMUsage) => LLMUsage
  ): Promise<LLMResponse> {
    console.log(`[web-search] generateWithSearch() entered`);
    const model = request.model ?? this.defaultModel;
    const startTime = Date.now();
    const userSchema = request.outputSchema as Record<string, unknown>;
    const temperature = request.temperature ?? 0;

    // --- Step 1: Try to answer (Responses API, forced function call) ------
    console.log(
      `[web-search] Step 1: Triage via Responses API, model=${model}`
    );

    const triageSchema = this.buildTriageSchema(userSchema);

    // Replace the output structure / example / response format sections
    // with ones generated from the triage wrapper schema.
    // Keep everything before "## Output Structure" (base + task description + rules).
    const systemPrompt = request.systemPrompt ?? "";
    const outputStructureIdx = systemPrompt.indexOf("\n## Output Structure");
    const basePart =
      outputStructureIdx >= 0
        ? systemPrompt.substring(0, outputStructureIdx)
        : systemPrompt;

    const triageSystemPrompt =
      basePart +
      buildWebSearchRoutingSection() +
      buildOutputStructureSection(triageSchema as JsonSchema) +
      buildResponseFormatSection();
    console.log(`[web-search] System prompt:\n${triageSystemPrompt}`);

    const triageInput = this.buildResponsesInput(
      triageSystemPrompt,
      request.prompt,
      request.media
    );

    const triageResult = await this.callResponsesStructured(
      model,
      triageInput,
      triageSchema,
      temperature,
      request.maxTokens
    );

    const triageUsage = record(triageResult.usage);

    const triageData = triageResult.content as {
      web_search_needed?: boolean;
      user_data?: unknown;
    };

    console.log(
      `[web-search] Triage result: web_search_needed=${triageData.web_search_needed}`
    );

    // --- If no search needed, or user_data is a clarification form, return directly
    const isClarification = this.containsInputControls(triageData.user_data);
    const shouldSearch =
      triageData.web_search_needed === true && !isClarification;

    if (!shouldSearch) {
      if (isClarification && triageData.web_search_needed) {
        console.log(
          `[web-search] AI set web_search_needed=true but response contains input controls — returning clarification`
        );
      } else {
        console.log(
          `[web-search] No search needed (web_search_needed=${triageData.web_search_needed}), returning user_data`
        );
      }
      const rawResponse = JSON.stringify(triageData.user_data);
      return {
        content: triageData.user_data,
        rawResponse,
        usage: triageUsage,
        model: triageResult.model,
        provider: this.providerName,
        latencyMs: Date.now() - startTime,
        finishReason: triageResult.finishReason,
      };
    }

    // --- Step 2: Web search -----------------------------------------------
    console.log(`[web-search] Step 2: Searching the web`);
    const searchResult = await this.searchWeb(
      model,
      request.prompt,
      temperature
    );

    record(searchResult.usage);
    console.log(
      `[web-search] Step 2 done: ${searchResult.summary.length} chars`
    );

    // --- Step 3: Structure search results (Responses API) -----------------
    console.log(`[web-search] Step 3: Structuring search results`);

    const structureSystemPrompt =
      (request.systemPrompt ?? "") +
      "\n\n## Important\n" +
      "You are being given web search results. Use ONLY the facts from " +
      "these results to build your response. Do not fabricate any names, " +
      "addresses, or data not found in the search results.";

    const structureInput = this.buildResponsesInput(
      structureSystemPrompt,
      `Original request: ${request.prompt}\n\n` +
        `Web search results:\n\n${searchResult.summary}\n\n` +
        "Using ONLY the information from the web search results above, " +
        "generate the structured response."
    );

    const structureResult = await this.callResponsesStructured(
      model,
      structureInput,
      userSchema,
      temperature,
      request.maxTokens
    );

    const totalUsage = record(structureResult.usage);

    console.log(
      `[web-search] Step 3 done, total time: ${Date.now() - startTime}ms`
    );

    return {
      content: structureResult.content,
      rawResponse: structureResult.rawResponse,
      usage: totalUsage,
      model: structureResult.model,
      provider: this.providerName,
      latencyMs: Date.now() - startTime,
      finishReason: structureResult.finishReason,
    };
  }

  buildApiPayload(request: LLMRequest): Record<string, unknown> {
    return buildOpenAIChatBody(
      { ...request, model: request.model ?? this.defaultModel },
      this.dialect
    );
  }
}
