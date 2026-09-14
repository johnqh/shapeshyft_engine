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
