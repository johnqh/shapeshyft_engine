/**
 * @fileoverview Jev (TypeSafe AI) LLM provider
 * @description Jev is not a text-generation model. It is a "System One"
 * model that answers pre-declared Choice/Score/Noul questions -- evaluated in
 * parallel -- with calibrated probabilities instead of writing free-form
 * text or arbitrary JSON. This adapter maps an endpoint's `outputSchema` onto
 * those primitives one top-level property at a time; see
 * {@link classifyJevField} for the exact rule, and {@link checkJevCompatibility}
 * to check a schema without calling Jev.
 *
 * Confirmed against https://docs.typesafe.ai on 2026-09-23. A page at
 * jevapi.org advertises a different endpoint domain ("tokenra.io") for the
 * same model; every official TypeSafe AI source (docs.typesafe.ai, its
 * Wikipedia entry, and the `@typesafe-ai/sdk` npm package's own metadata)
 * agrees only on api.typesafe.ai, so that's the only endpoint this adapter
 * will ever call.
 *
 * @see https://docs.typesafe.ai/api.md
 * @see https://docs.typesafe.ai/primitives.md
 */

import type {
  ILLMProvider,
  LLMRequest,
  LLMResponse,
  LLMUsage,
  ProviderConfig,
} from "./types.js";
import type { JsonSchema } from "../../types/index.js";
import { attachUsage } from "./usage-error.js";

const DEFAULT_MODEL = "jev-latest";

/** The only endpoint TypeSafe AI documents for Jev -- see the file header. */
const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";

type JevQuestion =
  | { type: "noul"; instructions: string }
  | { type: "choice"; instructions: string; criteria: Record<string, null> }
  | { type: "score"; instructions: string; criteria: string[] };

type JevAnswer =
  | { type: "noul"; noul: number }
  | {
      type: "choice";
      choice: string;
      confidence: number;
      probabilities: Record<string, number>;
    }
  | {
      type: "score";
      score: number;
      confidence: number;
      legend: Record<string, string>;
      probabilities: Record<string, number>;
    };

interface JevField {
  key: string;
  question: JevQuestion;
  /** Ordered rubric levels, for a "score" question only (low to high). */
  levels?: string[];
}

export interface JevCompatibility {
  compatible: boolean;
  /** One message per top-level field Jev cannot answer, naming the field. */
  errors: string[];
}

/**
 * Classify one top-level schema property into a Jev question, or explain why
 * it cannot be one:
 *
 * - `{ type: "boolean" }` -> Noul
 * - `{ type: "string", enum: [...] }` -> Choice (enum values become options)
 * - `{ type: "string", enum: [...], "x-jev-kind": "score" }` -> Score (the
 *   enum, in the given order, becomes the rubric's levels from low to high)
 *
 * Anything else (free text, numbers, arrays, nested objects) has no mapping.
 *
 * Exported so a frontend schema editor can give the same answer without a
 * round trip to `generate()`. Keep any such copy in sync with this one until
 * it can be shared as a published dependency instead.
 */
export function classifyJevField(
  key: string,
  schema: JsonSchema
): { field: JevField } | { error: string } {
  const instructions = schema.description ?? key;

  if (schema.type === "boolean") {
    return { field: { key, question: { type: "noul", instructions } } };
  }

  if (
    schema.type === "string" &&
    Array.isArray(schema.enum) &&
    schema.enum.length > 0
  ) {
    const levels = schema.enum.map(v => String(v));

    if (schema["x-jev-kind"] === "score") {
      if (levels.length < 2 || levels.length > 10) {
        return {
          error: `"${key}" has ${levels.length} enum values, but a Jev score rubric ("x-jev-kind": "score") needs between 2 and 10 ordered levels`,
        };
      }
      return {
        field: {
          key,
          question: { type: "score", instructions, criteria: levels },
          levels,
        },
      };
    }

    if (levels.length > 255) {
      return {
        error: `"${key}" has ${levels.length} enum values, but Jev's Choice primitive supports at most 255 options`,
      };
    }
    return {
      field: {
        key,
        question: {
          type: "choice",
          instructions,
          criteria: Object.fromEntries(levels.map(v => [v, null])),
        },
      },
    };
  }

  return {
    error:
      `"${key}" is a ${schema.type ?? "untyped"} field` +
      (schema.type === "string" ? " without an enum" : "") +
      `; Jev only answers boolean fields (Noul) or string fields with an "enum" (Choice, or Score with "x-jev-kind": "score")`,
  };
}

/**
 * Check whether an endpoint's `outputSchema` can be answered by Jev at all,
 * without calling it. `generate()` and `buildApiPayload()` both call this and
 * throw when it fails; it is also safe to call from a schema editor for live
 * feedback before an endpoint is saved.
 */
export function checkJevCompatibility(schema: JsonSchema): JevCompatibility {
  if (
    schema.type !== "object" ||
    !schema.properties ||
    Object.keys(schema.properties).length === 0
  ) {
    return {
      compatible: false,
      errors: [
        "Jev endpoints need an output schema that is a JSON object with at least one property, and each property must be a boolean or a string with an enum",
      ],
    };
  }
  const errors: string[] = [];
  for (const [key, propSchema] of Object.entries(schema.properties)) {
    const result = classifyJevField(key, propSchema);
    if ("error" in result) errors.push(result.error);
  }
  return { compatible: errors.length === 0, errors };
}

/** Builds the per-field question list, or throws with every incompatible field named. */
function buildFields(schema: JsonSchema): JevField[] {
  const check = checkJevCompatibility(schema);
  if (!check.compatible) {
    throw new Error(
      `Output schema is not compatible with Jev: ${check.errors.join("; ")}`
    );
  }
  return Object.entries(schema.properties!).map(([key, propSchema]) => {
    const result = classifyJevField(key, propSchema);
    if ("error" in result) {
      // Unreachable: checkJevCompatibility above already rejected this schema.
      throw new Error(result.error);
    }
    return result.field;
  });
}

/** Turn one of Jev's answers back into the value its field's schema declared. */
function decodeAnswer(field: JevField, answer: JevAnswer): unknown {
  switch (answer.type) {
    case "noul":
      return answer.noul >= 0.5;
    case "choice":
      return answer.choice;
    case "score": {
      const levels = field.levels ?? [];
      const index = Math.min(
        levels.length - 1,
        Math.max(0, Math.round(answer.score))
      );
      return levels[index];
    }
  }
}

export class JevProvider implements ILLMProvider {
  readonly providerName = "jev" as const;
  private apiKey: string;
  private defaultModel: string;

  constructor(config: ProviderConfig) {
    if (!config.apiKey) {
      throw new Error("Jev (TypeSafe AI) API key is required");
    }
    this.apiKey = config.apiKey;
    this.defaultModel = config.model ?? DEFAULT_MODEL;
  }

  async generate(request: LLMRequest): Promise<LLMResponse> {
    if (request.media?.length) {
      throw new Error(
        "Jev is text-only (state must be a string, JSON object, or array of text) and cannot accept media input"
      );
    }

    const model = request.model ?? this.defaultModel;
    const fields = buildFields(request.outputSchema);
    const payload = this.buildPayload(model, request, fields);
    const startTime = Date.now();

    let response: Response;
    try {
      response = await fetch(JEV_ENDPOINT, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    } catch (fetchError) {
      const message =
        fetchError instanceof Error ? fetchError.message : String(fetchError);
      throw new Error(`Failed to reach Jev at ${JEV_ENDPOINT}: ${message}`);
    }

    const latencyMs = Date.now() - startTime;

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(
        `Jev error (${response.status}): ${errorText || response.statusText}`
      );
    }

    const body = (await response.json()) as {
      model?: string;
      answers?: Record<string, JevAnswer>;
      usage?: { input_tokens?: number; output_tokens?: number };
    };

    const usage: LLMUsage = {
      promptTokens: body.usage?.input_tokens ?? 0,
      completionTokens: body.usage?.output_tokens ?? 0,
      totalTokens:
        (body.usage?.input_tokens ?? 0) + (body.usage?.output_tokens ?? 0),
    };

    const answers = body.answers ?? {};
    const content: Record<string, unknown> = {};
    for (const field of fields) {
      const answer = answers[field.key];
      if (!answer) {
        // The call was billed even though this field came back empty.
        throw attachUsage(
          new Error(`Jev did not answer "${field.key}"`),
          usage,
          body.model ?? model
        );
      }
      content[field.key] = decodeAnswer(field, answer);
    }

    return {
      content,
      rawResponse: JSON.stringify(body),
      usage,
      model: body.model ?? model,
      provider: this.providerName,
      latencyMs,
      // Jev answers every declared question in one parallel pass -- there is
      // no partial or truncated generation for it to report.
      finishReason: "stop",
    };
  }

  buildApiPayload(request: LLMRequest): Record<string, unknown> {
    const model = request.model ?? this.defaultModel;
    const fields = buildFields(request.outputSchema);
    return this.buildPayload(model, request, fields);
  }

  private buildPayload(
    model: string,
    request: LLMRequest,
    fields: JevField[]
  ): Record<string, unknown> {
    return {
      model,
      // Docs recommend objects for most requests, and keeping state (the
      // material to evaluate) separate from questions (the judgments to
      // make about it) -- https://docs.typesafe.ai/concepts/state.md
      state: request.systemPrompt
        ? { instructions: request.systemPrompt, input: request.prompt }
        : request.prompt,
      questions: Object.fromEntries(fields.map(f => [f.key, f.question])),
    };
  }
}
