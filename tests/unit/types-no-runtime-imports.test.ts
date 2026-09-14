import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import ts from "typescript";

/**
 * `./types` is re-exported by frontend packages. A runtime import here would
 * drag that module -- and whatever it pulls in -- into every browser bundle.
 * Type-only imports are erased by the compiler and are fine.
 */
describe("./types entry point", () => {
  it("has no runtime import or re-export-from statements", () => {
    const file = resolve(import.meta.dirname, "../../src/types/index.ts");
    const source = ts.createSourceFile(
      file,
      readFileSync(file, "utf8"),
      ts.ScriptTarget.ES2022,
      true
    );

    const runtimeImports: string[] = [];
    for (const stmt of source.statements) {
      if (ts.isImportDeclaration(stmt) && !stmt.importClause?.isTypeOnly) {
        runtimeImports.push(stmt.getText());
      }
      if (
        ts.isExportDeclaration(stmt) &&
        stmt.moduleSpecifier &&
        !stmt.isTypeOnly
      ) {
        runtimeImports.push(stmt.getText());
      }
    }

    expect(runtimeImports).toEqual([]);
  });
});
