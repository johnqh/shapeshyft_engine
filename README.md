# @sudobility/shapeshyft_engine

Stateless LLM structured-output engine used by ShapeShyft and ShapeRouter.

```ts
import { createLLMProvider } from "@sudobility/shapeshyft_engine";
import type { LlmProvider } from "@sudobility/shapeshyft_engine/types";

const provider = createLLMProvider("openai", { apiKey });
const result = await provider.generate(request);
```
