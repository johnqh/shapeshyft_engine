/**
 * @fileoverview Structured output for Cohere's OpenAI-compatible API.
 * @description Cohere's Compatibility API accepts `tools` but not
 * `tool_choice`, so the forced `structured_response` call every other
 * compatible provider uses cannot be made. Its structured output is
 * `response_format: { type: "json_object", schema }` instead, which Cohere
 * guarantees the reply follows -- but only for a subset of JSON Schema.
 *
 * @see https://docs.cohere.com/docs/compatibility-api
 * @see https://docs.cohere.com/docs/structured-outputs
 */

/**
 * Keywords Cohere rejects. Dropping them loosens validation, not the answer:
 * the system prompt still describes the full schema, bounds included.
 */
const UNSUPPORTED_KEYWORDS = new Set([
  "allOf",
  "oneOf",
  "not",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "minItems",
  "maxItems",
  "minLength",
  "maxLength",
  "uniqueItems",
  "$schema",
  "$id",
]);

/** The only string formats Cohere supports. */
const SUPPORTED_FORMATS = new Set(["date-time", "uuid", "date", "time"]);

type Schema = Record<string, unknown>;

function isSchema(value: unknown): value is Schema {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Copy `schema` without the keywords Cohere rejects. */
function stripUnsupported(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(stripUnsupported);
  if (!isSchema(schema)) return schema;

  const out: Schema = {};
  for (const [key, value] of Object.entries(schema)) {
    if (UNSUPPORTED_KEYWORDS.has(key)) continue;
    if (key === "format" && !SUPPORTED_FORMATS.has(String(value))) continue;
    // A pattern using anchors or lookaheads is rejected; the prompt keeps it.
    if (key === "pattern" && /\^|\$|\?=|\?!/.test(String(value))) continue;
    // `properties`, `$defs` and `definitions` map names to schemas; every
    // other object-valued keyword is a schema itself.
    if (
      (key === "properties" || key === "$defs" || key === "definitions") &&
      isSchema(value)
    ) {
      out[key] = Object.fromEntries(
        Object.entries(value).map(([name, child]) => [
          name,
          stripUnsupported(child),
        ])
      );
    } else {
      out[key] = stripUnsupported(value);
    }
  }
  return out;
}

/**
 * Whether Cohere will accept `schema` in JSON Schema mode: the top level must
 * be an object, and every object must list at least one `required` field.
 */
function meetsCohereConstraints(schema: unknown, topLevel = true): boolean {
  if (Array.isArray(schema)) {
    return schema.every(s => meetsCohereConstraints(s, false));
  }
  if (!isSchema(schema)) return true;

  const isObject = schema.type === "object" || isSchema(schema.properties);
  if (topLevel && !isObject) return false;
  if (
    isObject &&
    !(Array.isArray(schema.required) && schema.required.length > 0)
  ) {
    return false;
  }

  return Object.entries(schema).every(([key, value]) =>
    (key === "properties" || key === "$defs" || key === "definitions") &&
    isSchema(value)
      ? Object.values(value).every(child =>
          meetsCohereConstraints(child, false)
        )
      : meetsCohereConstraints(value, false)
  );
}

/**
 * The `response_format` to send Cohere for an endpoint's output schema.
 *
 * JSON Schema mode when the schema fits Cohere's constraints. Otherwise JSON
 * mode, which still guarantees a JSON object: the schema's shape then comes
 * from the system prompt, which always describes it and asks for JSON --
 * Cohere warns JSON mode can run away without that instruction.
 */
export function cohereResponseFormat(outputSchema: unknown): {
  type: "json_object";
  schema?: Schema;
} {
  const stripped = stripUnsupported(outputSchema);
  return meetsCohereConstraints(stripped)
    ? { type: "json_object", schema: stripped as Schema }
    : { type: "json_object" };
}
