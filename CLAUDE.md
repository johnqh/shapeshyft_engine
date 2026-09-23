# shapeshyft_engine

> **Git policy — never auto-commit or auto-push.** Run `git commit`, `git push`,
> or publish only when the user explicitly asks in that turn.

Stateless core shared by `shapeshyft_api` and `shaperouter_api` (via
`@sudobility/shapeshyft_service`). No database, no Hono, no `process.env`.

## Entry points

- `@sudobility/shapeshyft_engine` — `createLLMProvider`, provider adapters,
  provider/model catalog (`config/providers`), prompt builder, `ApiHelper`,
  media handling, capability validation, reserved fields, output limits.
- `@sudobility/shapeshyft_engine/types` — domain types and pure helpers. Must
  never gain a runtime `import`; `tests/unit/types-no-runtime-imports.test.ts`
  enforces it. Frontend packages (`shapeshyft_types`, `shaperouter_types`)
  re-export this.

## Rules

- Relative imports carry `.js` (`NodeNext`). The compiler rejects them otherwise.
- Provider SDKs and `sharp` are optional peer dependencies.
- Provider credentials arrive in `ProviderConfig`; the engine never looks them up.

## Commands

    bun run verify   # typecheck + lint + test + build

## Model JSON is read tolerantly

`readModelJson` (`services/llm/json-repair.ts`) replaces `JSON.parse` on model output in the OpenAI-compatible (incl. DeepSeek), Groq and Gemini adapters. **Strict parsing runs first and valid JSON is returned exactly as `JSON.parse` returns it**; only text `JSON.parse` refused reaches the tolerant reader, which fixes trailing/missing/extra commas, unclosed or wrongly closed brackets, malformed numbers (`0480`, `+3`, `.5`), arithmetic in a number (`1920 - 480`), single quotes, unquoted keys and short barewords, Python constants, comments, raw control characters, unescaped quotes and bad escapes, and ignores text after a complete document. It gives up (throws the strict error) on anything with more than 40 repairs, a bareword over 64 characters, or no leading `{`/`[`. Every repair is logged with what was fixed and a window of text around the first one — the failed answers were never kept, so that log is the only record of what a model got wrong. Measured on one product's history: 1,045 of 1,205 failed calls were parse failures (55% "Unable to parse JSON string", which real samples show is mostly a complete document followed by extra text). A repaired answer is only as good as the shape the caller validates it against.
