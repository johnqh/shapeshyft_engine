import { describe, expect, it } from "vitest";
import { cohereResponseFormat } from "../../src/services/llm/cohere-schema.js";
import {
  OpenAIProvider,
  type StructuredOutputMode,
} from "../../src/services/llm/openai.js";
import {
  PROVIDER_ENDPOINTS,
  createLLMProvider,
} from "../../src/services/llm/index.js";
import { getFailedInvocationUsage } from "../../src/services/llm/usage-error.js";

const bookSchema = {
  type: "object",
  properties: {
    title: { type: "string", minLength: 1, maxLength: 200 },
    year: { type: "integer", minimum: 1450, maximum: 2100 },
    tags: { type: "array", items: { type: "string" }, maxItems: 5 },
    published: { type: "string", format: "date" },
    isbn: { type: "string", format: "isbn", pattern: "^[0-9-]+$" },
  },
  required: ["title"],
};

describe("cohereResponseFormat", () => {
  it("sends the schema without the keywords Cohere rejects", () => {
    const format = cohereResponseFormat(bookSchema);
    expect(format).toEqual({
      type: "json_object",
      schema: {
        type: "object",
        properties: {
          title: { type: "string" },
          year: { type: "integer" },
          tags: { type: "array", items: { type: "string" } },
          published: { type: "string", format: "date" },
          isbn: { type: "string" },
        },
        required: ["title"],
      },
    });
  });

  it("does not mistake a property named like a keyword for the keyword", () => {
    const format = cohereResponseFormat({
      type: "object",
      properties: { minimum: { type: "number" } },
      required: ["minimum"],
    });
    expect(format.schema?.properties).toEqual({ minimum: { type: "number" } });
  });

  it("falls back to JSON mode when an object has no required field", () => {
    const format = cohereResponseFormat({
      type: "object",
      properties: {
        author: {
          type: "object",
          properties: { name: { type: "string" } },
        },
      },
      required: ["author"],
    });
    expect(format).toEqual({ type: "json_object" });
  });

  it("falls back to JSON mode when the top level is not an object", () => {
    expect(
      cohereResponseFormat({ type: "array", items: { type: "string" } })
    ).toEqual({ type: "json_object" });
  });
});

function withFakeClient(
  mode: StructuredOutputMode,
  reply: Record<string, unknown>
) {
  const provider = new OpenAIProvider(
    { apiKey: "test", endpointUrl: "https://api.cohere.ai/compatibility/v1" },
    { structuredOutput: mode }
  );
  const sent: Record<string, unknown>[] = [];
  (
    provider as unknown as {
      client: {
        chat: {
          completions: {
            create: (body: Record<string, unknown>) => Promise<unknown>;
          };
        };
      };
    }
  ).client = {
    chat: {
      completions: {
        create: async body => {
          sent.push(body);
          return reply;
        },
      },
    },
  };
  return { provider, sent };
}

describe("OpenAIProvider in response_format mode (Cohere)", () => {
  const request = {
    prompt: "Describe a book.",
    systemPrompt: "Respond with valid JSON only.",
    outputSchema: bookSchema,
    model: "command-a-03-2025",
  };

  it("asks for response_format and never sends tools or tool_choice", async () => {
    const { provider, sent } = withFakeClient("response_format", {
      model: "command-a-03-2025",
      usage: { prompt_tokens: 50, completion_tokens: 12, total_tokens: 62 },
      choices: [
        {
          finish_reason: "stop",
          message: { content: '{"title":"Dune"}' },
        },
      ],
    });

    const response = await provider.generate(request);

    expect(sent[0]).toMatchObject({
      response_format: { type: "json_object" },
      max_tokens: undefined,
    });
    expect(sent[0]).not.toHaveProperty("tools");
    expect(sent[0]).not.toHaveProperty("tool_choice");
    expect(response.content).toEqual({ title: "Dune" });
    expect(response.usage).toMatchObject({
      promptTokens: 50,
      completionTokens: 12,
    });
  });

  it("returns truncated content as a truncation, not a parse failure", async () => {
    const { provider } = withFakeClient("response_format", {
      model: "command-a-03-2025",
      usage: { prompt_tokens: 50, completion_tokens: 100, total_tokens: 150 },
      choices: [{ finish_reason: "length", message: { content: '{"title":' } }],
    });

    const response = await provider.generate(request);
    expect(response.finishReason).toBe("length");
    expect(response.content).toBe('{"title":');
  });

  it("carries usage when the reply has no content to parse", async () => {
    const { provider } = withFakeClient("response_format", {
      model: "command-a-03-2025",
      usage: { prompt_tokens: 50, completion_tokens: 0, total_tokens: 50 },
      choices: [{ finish_reason: "stop", message: { content: null } }],
    });

    const failure = await provider.generate(request).catch((e: unknown) => e);
    expect(getFailedInvocationUsage(failure)?.usage.promptTokens).toBe(50);
  });

  it("builds the same response_format into an API payload", () => {
    const { provider } = withFakeClient("response_format", {});
    const payload = provider.buildApiPayload(request);
    expect(payload).toHaveProperty("response_format.type", "json_object");
    expect(payload).not.toHaveProperty("tool_choice");
  });
});

describe("Cohere provider wiring", () => {
  it("reaches Cohere's compatibility API, not api.openai.com", () => {
    const provider = createLLMProvider("cohere", { apiKey: "co-key" });
    const client = (provider as unknown as { client: { baseURL: string } })
      .client;
    expect(client.baseURL).toBe("https://api.cohere.ai/compatibility/v1");
    expect(PROVIDER_ENDPOINTS.cohere).toBe(
      "https://api.cohere.ai/compatibility/v1/chat/completions"
    );
  });
});
