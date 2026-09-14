import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "../..");

describe("built package under plain Node ESM", () => {
  it("imports both entry points without a bundler", () => {
    const script = `
      const engine = await import(${JSON.stringify(`${ROOT}/dist/index.js`)});
      const types = await import(${JSON.stringify(`${ROOT}/dist/types/index.js`)});
      if (typeof engine.createLLMProvider !== "function") throw new Error("engine");
      if (typeof types.successResponse !== "function") throw new Error("types");
      console.log("ok");
    `;
    const out = execFileSync("node", ["--input-type=module", "-e", script], {
      cwd: ROOT,
      encoding: "utf8",
    });
    expect(out.trim()).toBe("ok");
  });
});
