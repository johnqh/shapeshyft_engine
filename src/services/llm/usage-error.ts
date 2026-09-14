/**
 * @fileoverview Usage carried on a failed invocation.
 * @description A provider that answers with something we cannot use -- JSON
 * that does not parse, no tool call -- still bills the tokens it generated.
 * Adapters attach what they know to the error so the caller can record the
 * cost of a failure instead of treating it as free.
 */

import type { LLMResponse, LLMUsage } from "./types.js";

export interface FailedInvocationUsage {
  usage: LLMUsage;
  model: string;
  /** Audio transcribed before the failing step, which is billed regardless. */
  transcription?: LLMResponse["transcription"];
}

const USAGE = Symbol.for("shapeshyft.invocationUsage");

/**
 * Attach usage to an error and return it, for `throw attachUsage(...)`.
 * Replaces any usage already attached: a multi-call flow reads the inner
 * call's usage with `getFailedInvocationUsage`, adds its own, and re-attaches.
 */
export function attachUsage<E>(
  error: E,
  usage: LLMUsage,
  model: string,
  transcription?: LLMResponse["transcription"]
): E {
  if (error !== null && typeof error === "object") {
    Object.defineProperty(error, USAGE, {
      value: {
        usage,
        model,
        ...(transcription ? { transcription } : {}),
      } satisfies FailedInvocationUsage,
      enumerable: false,
      configurable: true,
    });
  }
  return error;
}

/** The usage a failed invocation consumed, if its adapter could tell. */
export function getFailedInvocationUsage(
  error: unknown
): FailedInvocationUsage | undefined {
  if (error === null || typeof error !== "object") return undefined;
  return (error as Record<symbol, FailedInvocationUsage | undefined>)[USAGE];
}
