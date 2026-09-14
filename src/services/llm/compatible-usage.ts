/**
 * @fileoverview Usage from OpenAI-compatible chat completion responses.
 * @description OpenAI, DeepSeek, xAI, Mistral, Perplexity and Groq share the
 * `usage` shape for token totals and each extend it with what their bill
 * depends on. One reader takes all of them, since a field one provider never
 * sends is simply absent.
 */

import type { LLMUsage } from "./types.js";

type Raw = Record<string, unknown> | undefined;

/** 1 USD in xAI's `cost_in_usd_ticks`. */
const XAI_TICKS_PER_USD = 10_000_000_000;

function count(value: unknown): number | undefined {
  const n = typeof value === "string" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : undefined;
}

function nested(raw: Raw, key: string): Raw {
  const value = raw?.[key];
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : undefined;
}

/** A billed amount that may legitimately be zero, unlike a token count. */
function amount(value: unknown): number | undefined {
  const n = typeof value === "string" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) && n >= 0 ? n : undefined;
}

/**
 * Read an OpenAI-compatible `usage` object.
 *
 * - `prompt_tokens_details.cached_tokens`: OpenAI, xAI, Groq, Mistral; DeepSeek
 *   also sends `prompt_cache_hit_tokens`, and Mistral `num_cached_tokens`. A
 *   part of `prompt_tokens`.
 * - `prompt_tokens_details.cache_write_tokens`: OpenAI GPT-5.6+, billed above
 *   the input rate. A part of `prompt_tokens`.
 * - `completion_tokens_details.reasoning_tokens`: a part of `completion_tokens`
 *   for OpenAI, but NOT for xAI, which leaves it out of `completion_tokens` and
 *   bills it as output. Told apart by `total_tokens`: when it only adds up with
 *   reasoning included, reasoning was outside.
 * - `reasoning_tokens`, `citation_tokens`, `num_search_queries` (top level):
 *   Perplexity, each billed on its own line.
 * - `cost_in_usd_ticks` (xAI) and `cost.total_cost` (Perplexity, USD): the
 *   amount the provider billed, which supersedes any estimate.
 */
export function compatibleUsage(usage: unknown): LLMUsage {
  const raw =
    usage && typeof usage === "object"
      ? (usage as Record<string, unknown>)
      : undefined;
  const promptTokens = count(raw?.prompt_tokens) ?? 0;
  let completionTokens = count(raw?.completion_tokens) ?? 0;
  const totalReported = count(raw?.total_tokens);

  const promptDetails = nested(raw, "prompt_tokens_details");
  const cached =
    count(promptDetails?.cached_tokens) ??
    count(raw?.prompt_cache_hit_tokens) ??
    count(raw?.num_cached_tokens);
  const cacheWrite = count(promptDetails?.cache_write_tokens);

  const reasoning = count(
    nested(raw, "completion_tokens_details")?.reasoning_tokens
  );
  if (
    reasoning &&
    totalReported !== undefined &&
    totalReported === promptTokens + completionTokens + reasoning
  ) {
    completionTokens += reasoning;
  }

  const separateReasoning = count(raw?.reasoning_tokens);
  const citations = count(raw?.citation_tokens);
  const searches = count(raw?.num_search_queries);

  const xaiTicks = amount(raw?.cost_in_usd_ticks);
  const perplexityUsd = amount(nested(raw, "cost")?.total_cost);
  const billedCostCents =
    xaiTicks !== undefined
      ? (xaiTicks / XAI_TICKS_PER_USD) * 100
      : perplexityUsd !== undefined
        ? perplexityUsd * 100
        : undefined;

  return {
    promptTokens,
    completionTokens,
    totalTokens: totalReported ?? promptTokens + completionTokens,
    ...(cached ? { cachedInputTokens: cached } : {}),
    ...(cacheWrite ? { cacheWriteInputTokens: cacheWrite } : {}),
    ...(reasoning ? { reasoningTokens: reasoning } : {}),
    ...(separateReasoning
      ? { separateReasoningTokens: separateReasoning }
      : {}),
    ...(citations ? { citationTokens: citations } : {}),
    ...(searches ? { searchCalls: searches } : {}),
    ...(billedCostCents !== undefined ? { billedCostCents } : {}),
  };
}

/** Add the usage of one more provider call to a running total. */
export function addUsage(total: LLMUsage, next: LLMUsage): LLMUsage {
  const merged: LLMUsage = {
    promptTokens: total.promptTokens + next.promptTokens,
    completionTokens: total.completionTokens + next.completionTokens,
    totalTokens: total.totalTokens + next.totalTokens,
    requests: (total.requests ?? 1) + (next.requests ?? 1),
    maxPromptTokens: Math.max(
      total.maxPromptTokens ?? total.promptTokens,
      next.maxPromptTokens ?? next.promptTokens
    ),
  };
  const optional = [
    "cachedInputTokens",
    "cacheWriteInputTokens",
    "audioInputTokens",
    "reasoningTokens",
    "separateReasoningTokens",
    "citationTokens",
    "searchCalls",
  ] as const;
  for (const key of optional) {
    if (total[key] !== undefined || next[key] !== undefined) {
      merged[key] = (total[key] ?? 0) + (next[key] ?? 0);
    }
  }
  // A billed figure is only meaningful for the whole: summing one call that
  // has it with one that does not would understate the total.
  if (
    total.billedCostCents !== undefined &&
    next.billedCostCents !== undefined
  ) {
    merged.billedCostCents = total.billedCostCents + next.billedCostCents;
  }
  return merged;
}
