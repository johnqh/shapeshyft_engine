/**
 * @fileoverview Cost of one invocation, from what the provider reported.
 * @description `estimateCost` prices input and output tokens at one rate each.
 * Real bills differ from that in ways that are large, not rounding: cached
 * input at a tenth of the rate, audio at three times it, thinking tokens that
 * never reach the answer, per-request and per-search fees, long-context tiers,
 * and peak-hour multipliers. This module prices all of them, at full precision.
 */

import type { LlmProvider, ModelPricing } from "../types/index.js";
import type { LLMResponse, LLMUsage } from "../services/llm/types.js";
import { findModelPricing, getModelPricing } from "../config/providers.js";

const PER_MILLION = 1_000_000;

/** The rates in effect at `at`, after any announced change has taken effect. */
function ratesAt(pricing: ModelPricing, at: Date): ModelPricing {
  if (!pricing.priceChanges?.length) return pricing;
  const inEffect = pricing.priceChanges
    .filter(change => Date.parse(change.from) <= at.getTime())
    .sort((a, b) => Date.parse(a.from) - Date.parse(b.from));
  return inEffect.reduce<ModelPricing>(
    (rates, { from: _from, ...change }) => ({ ...rates, ...change }),
    pricing
  );
}

function isPeak(pricing: ModelPricing, at: Date): boolean {
  if (!pricing.peak) return false;
  const day = at.getUTCDay();
  const hour = at.getUTCHours();
  return pricing.peak.windowsUtc.some(
    w => w.days.includes(day) && hour >= w.startHour && hour < w.endHour
  );
}

/**
 * Estimate the cost, in cents, of the usage one invocation reported.
 *
 * When the usage carries `billedCostCents` -- a figure the provider billed, or
 * the adapter priced from a finer breakdown -- that is the answer and the
 * catalog rates are not consulted.
 *
 * @param pricing - Rates for the model that produced the usage
 * @param usage - Token and fee counts from the adapter
 * @param at - When the request was made; selects peak-hour and dated rates. Default: now.
 */
export function estimateUsageCost(
  pricing: ModelPricing,
  usage: LLMUsage,
  at: Date = new Date()
): number {
  if (usage.billedCostCents !== undefined) return usage.billedCostCents;

  const rates = ratesAt(pricing, at);
  const promptTokens = Math.max(0, usage.promptTokens);
  const largestPrompt = usage.maxPromptTokens ?? promptTokens;
  const tier =
    rates.longContext && largestPrompt >= rates.longContext.minPromptTokens
      ? rates.longContext
      : undefined;

  const inputRate = tier?.input ?? rates.input;
  const outputRate = tier?.output ?? rates.output;
  // A tier rate scales the discounts and surcharges set against the base
  // input rate, unless the tier names its own cached rate.
  const scale = rates.input > 0 ? inputRate / rates.input : 1;
  const cachedRate =
    tier?.cachedInput ??
    (rates.cachedInput !== undefined ? rates.cachedInput * scale : inputRate);
  const cacheWriteRate =
    rates.cacheWriteInput !== undefined
      ? rates.cacheWriteInput * scale
      : inputRate;
  const audioRate =
    rates.audioTokenInput !== undefined
      ? rates.audioTokenInput * scale
      : inputRate;

  // Cached, cache-write and audio tokens are parts of promptTokens, never
  // additions to it.
  let remaining = promptTokens;
  const take = (n: number | undefined) => {
    const part = Math.min(Math.max(0, n ?? 0), remaining);
    remaining -= part;
    return part;
  };
  const cached = take(usage.cachedInputTokens);
  const cacheWrite = take(usage.cacheWriteInputTokens);
  const audio = take(usage.audioInputTokens);
  const plainInput = remaining;

  let tokenCost =
    (plainInput * inputRate +
      cached * cachedRate +
      cacheWrite * cacheWriteRate +
      audio * audioRate +
      Math.max(0, usage.completionTokens) * outputRate +
      (usage.separateReasoningTokens ?? 0) *
        (rates.reasoningTokens ?? outputRate) +
      (usage.citationTokens ?? 0) * (rates.citationTokens ?? 0)) /
    PER_MILLION;

  if (isPeak(rates, at)) tokenCost *= rates.peak!.multiplier;

  const fees =
    (usage.requests ?? 1) * (rates.requestFee ?? 0) +
    (usage.searchCalls ?? 0) * (rates.searchCall ?? 0);

  return tokenCost + fees;
}

export interface ResponseCost {
  /** Estimated cost in cents, at full precision. */
  costCents: number;
  /**
   * False when the cost used `DEFAULT_MODEL_PRICING` because the catalog knows
   * neither the reported nor the configured model. Worth logging: the estimate
   * is then a placeholder, not a price.
   */
  pricingKnown: boolean;
}

/**
 * Estimate the total cost of one endpoint call.
 *
 * Pricing is looked up from the model the provider reports, then the model the
 * endpoint is configured with -- providers report dated snapshots and resolved
 * aliases the catalog does not list. Self-hosted providers are free.
 *
 * @param response - The adapter's response, or the usage salvaged from a failure
 * @param options.provider - The provider the credential belongs to. One adapter
 *   serves several providers and reports its own name, so pass this rather
 *   than `response.provider`.
 * @param options.configuredModel - The model the endpoint asked for
 * @param options.at - When the request was made
 */
export function estimateResponseCost(
  response: Pick<LLMResponse, "usage" | "model" | "transcription">,
  options: {
    provider: LlmProvider;
    configuredModel?: string | null;
    at?: Date;
  }
): ResponseCost {
  const lookup = {
    provider: options.provider,
    configuredModel: options.configuredModel,
  };
  const at = options.at ?? new Date();

  let pricingKnown =
    response.usage.billedCostCents !== undefined ||
    findModelPricing(response.model, lookup) !== undefined;
  let costCents = estimateUsageCost(
    getModelPricing(response.model, lookup),
    response.usage,
    at
  );

  if (response.transcription) {
    const { model, billedSeconds } = response.transcription;
    const transcriptionPricing = findModelPricing(model, {
      provider: options.provider,
    });
    if (transcriptionPricing === undefined) pricingKnown = false;
    costCents += (billedSeconds / 60) * (transcriptionPricing?.audioInput ?? 0);
  }

  return { costCents, pricingKnown };
}
