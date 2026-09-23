import { describe, expect, it, vi } from "vitest";
import { OpenAIProvider } from "../../src/services/llm/openai.js";

describe("OpenAIProvider", () => {
  it("returns truncated tool-call arguments with usage instead of parsing them", async () => {
    const provider = new OpenAIProvider({ apiKey: "test" });
    (
      provider as unknown as {
        client: {
          chat: { completions: { create: () => Promise<unknown> } };
        };
      }
    ).client = {
      chat: {
        completions: {
          create: async () => ({
            model: "gpt-5.4",
            usage: {
              prompt_tokens: 12,
              completion_tokens: 99,
              total_tokens: 111,
            },
            choices: [
              {
                finish_reason: "length",
                message: {
                  tool_calls: [
                    {
                      function: {
                        name: "structured_response",
                        arguments: '{"items":[',
                      },
                    },
                  ],
                },
              },
            ],
          }),
        },
      },
    };

    const response = await provider.generate({
      prompt: "write data",
      outputSchema: { type: "object" },
      model: "gpt-5.4",
      maxTokens: 100,
    });

    expect(response.finishReason).toBe("length");
    expect(response.content).toBe('{"items":[');
    expect(response.usage).toEqual({
      promptTokens: 12,
      completionTokens: 99,
      totalTokens: 111,
    });
  });

  /*
    A model's slip in a long answer used to be a failed, billed call. The same
    text is now read, and the repair is logged, because a repair that works
    leaves no error and so no record of what the model got wrong.
  */
  describe("a malformed function-call payload", () => {
    const providerAnswering = (args: string) => {
      const provider = new OpenAIProvider({ apiKey: "test" });
      (provider as unknown as { client: unknown }).client = {
        chat: {
          completions: {
            create: async () => ({
              model: "deepseek-flash",
              usage: {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
              },
              choices: [
                {
                  finish_reason: "stop",
                  message: {
                    tool_calls: [
                      {
                        function: {
                          name: "structured_response",
                          arguments: args,
                        },
                      },
                    ],
                  },
                },
              ],
            }),
          },
        },
      };
      return provider;
    };
    const request = {
      prompt: "write",
      outputSchema: { type: "object" },
      model: "deepseek-flash",
    };

    it("is repaired and returned, with the repair logged", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        const response = await providerAnswering(
          '{"events": [[0, 480, "C4", 90,], [480 480 "D4", 88]]}'
        ).generate(request);
        expect(response.content).toEqual({
          events: [
            [0, 480, "C4", 90],
            [480, 480, "D4", 88],
          ],
        });
        const logged = warn.mock.calls.map(c => String(c[0])).join("\n");
        expect(logged).toContain("repaired 3 JSON fault(s)");
        expect(logged).toContain("trailing comma");
        expect(logged).toContain("missing comma x2");
        // The usage still reaches the caller, as it does for a clean answer.
        expect(response.usage.promptTokens).toBe(10);
      } finally {
        warn.mockRestore();
      }
    });

    it("leaves a valid answer alone and says nothing", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        const response =
          await providerAnswering('{"a": [1, 2, 3]}').generate(request);
        expect(response.content).toEqual({ a: [1, 2, 3] });
        expect(
          warn.mock.calls.filter(c => String(c[0]).includes("repaired"))
        ).toEqual([]);
      } finally {
        warn.mockRestore();
      }
    });

    it("still fails, with the usage attached, when the text is not recoverable", async () => {
      await expect(
        providerAnswering("this is not json at all").generate(request)
      ).rejects.toThrow(/unparseable JSON/);
    });
  });
});
