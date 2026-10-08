import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import ts from "typescript";

/**
 * `./core` is imported by React Native and browser apps. Everything it loads at
 * runtime -- followed through every relative import -- must be plain
 * TypeScript: no provider SDK, no `sharp`, no Node built-in, no `Buffer`, no
 * `process`. Type-only imports are erased by the compiler and are fine.
 */
const SRC = resolve(import.meta.dirname, "../../src");
const ENTRY = resolve(SRC, "core/index.ts");

function runtimeSpecifiers(file: string): string[] {
  const source = ts.createSourceFile(
    file,
    readFileSync(file, "utf8"),
    ts.ScriptTarget.ES2022,
    true
  );
  const specs: string[] = [];
  for (const stmt of source.statements) {
    if (
      ts.isImportDeclaration(stmt) &&
      !stmt.importClause?.isTypeOnly &&
      ts.isStringLiteral(stmt.moduleSpecifier)
    ) {
      // `import { type A, type B }` is erased too
      const named = stmt.importClause?.namedBindings;
      const allTypes =
        stmt.importClause &&
        !stmt.importClause.name &&
        named &&
        ts.isNamedImports(named) &&
        named.elements.every(e => e.isTypeOnly);
      if (!allTypes) specs.push(stmt.moduleSpecifier.text);
    }
    if (
      ts.isExportDeclaration(stmt) &&
      !stmt.isTypeOnly &&
      stmt.moduleSpecifier &&
      ts.isStringLiteral(stmt.moduleSpecifier)
    ) {
      const clause = stmt.exportClause;
      const allTypes =
        clause &&
        ts.isNamedExports(clause) &&
        clause.elements.every(e => e.isTypeOnly);
      if (!allTypes) specs.push(stmt.moduleSpecifier.text);
    }
  }
  return specs;
}

function closure(): Map<string, string[]> {
  const seen = new Map<string, string[]>();
  const queue = [ENTRY];
  while (queue.length > 0) {
    const file = queue.pop()!;
    if (seen.has(file)) continue;
    const specs = runtimeSpecifiers(file);
    seen.set(file, specs);
    for (const spec of specs) {
      if (spec.startsWith(".")) {
        queue.push(resolve(dirname(file), spec.replace(/\.js$/, ".ts")));
      }
    }
  }
  return seen;
}

describe("./core entry point", () => {
  const files = closure();

  it("walks the whole runtime graph", () => {
    // payload, request, response, endpoints, json-repair, providers config...
    expect(files.size).toBeGreaterThan(8);
  });

  it("loads only relative modules at runtime", () => {
    const external = [...files.entries()].flatMap(([file, specs]) =>
      specs
        .filter(s => !s.startsWith("."))
        .map(s => `${file.slice(SRC.length)}: ${s}`)
    );
    expect(external).toEqual([]);
  });

  it("never touches Buffer, process or a Node built-in", () => {
    const offenders: string[] = [];
    for (const file of files.keys()) {
      const text = readFileSync(file, "utf8")
        // comments may mention them
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/.*$/gm, "");
      for (const pattern of [/\bBuffer\b/, /\bprocess\./, /["']node:/]) {
        if (pattern.test(text)) {
          offenders.push(`${file.slice(SRC.length)}: ${pattern}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });

  it("does not reach the SDK adapters", () => {
    const adapters = [...files.keys()].filter(f =>
      /services\/llm\/(openai|anthropic|gemini|groq|custom|jev|index)\.ts$/.test(
        f
      )
    );
    expect(adapters).toEqual([]);
  });
});
