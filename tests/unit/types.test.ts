import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  successResponse,
  errorResponse,
  estimateCost,
  estimateMultimodalCost,
  formatCost,
  formatCostPerMillion,
  detectRequiredCapabilities,
  isValidModelForProvider,
  PROVIDER_MODELS,
  LLM_PROVIDERS,
  DEFAULT_MAX_OUTPUT_TOKENS,
  FINISH_REASONS,
  type FinishReason,
  type LlmProvider,
  type HttpMethod,
  type JsonSchema,
  type ModelPricing,
  type MultimodalUsage,
  type BooleanQueryParam,
  type User,
  type Project,
  type EndpointBase,
  type UsageAnalytics,
  type UserSettings,
  type UsageAggregate,
  type UsageByEndpoint,
  type UsageByDate,
  type AnalyticsResponse,
  type LlmRequest,
  type LlmResponse,
  type AiExecutionRequest,
  type AiExecutionResponse,
  type AiPromptResponse,
  type PromptInput,
  type ApiHelperRequestInput,
  type ApiHelperRequestOutput,
  type RefreshApiKeyResponse,
  type GetApiKeyResponse,
  type HealthCheckData,
  type UserCreateRequest,
  type UserUpdateRequest,
  type UserSettingsUpdateRequest,
  type ProjectCreateRequest,
  type ProjectUpdateRequest,
  type ProjectQueryParams,
  type EndpointQueryParams,
  type UsageAnalyticsQueryParams,
} from '../../src/types/index.js';

describe('shapeshyft_types', () => {
  describe('successResponse', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-15T10:30:00.000Z'));
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should create a success response with data', () => {
      const data = { id: '123', name: 'Test' };
      const response = successResponse(data);

      expect(response.success).toBe(true);
      expect(response.data).toEqual({ id: '123', name: 'Test' });
      expect(response.timestamp).toBe('2025-01-15T10:30:00.000Z');
    });

    it('should create a success response with string data', () => {
      const response = successResponse('test string');

      expect(response.success).toBe(true);
      expect(response.data).toBe('test string');
    });

    it('should create a success response with array data', () => {
      const data = [{ id: 1 }, { id: 2 }];
      const response = successResponse(data);

      expect(response.success).toBe(true);
      expect(response.data).toEqual([{ id: 1 }, { id: 2 }]);
    });

    it('should create a success response with null data', () => {
      const response = successResponse(null);

      expect(response.success).toBe(true);
      expect(response.data).toBeNull();
    });

    it('should create a success response with number data', () => {
      const response = successResponse(42);

      expect(response.success).toBe(true);
      expect(response.data).toBe(42);
    });

    it('should create a success response with boolean data', () => {
      const response = successResponse(true);

      expect(response.success).toBe(true);
      expect(response.data).toBe(true);
    });

    it('should not include error field', () => {
      const response = successResponse({ test: true });

      expect(response).not.toHaveProperty('error');
    });
  });

  describe('errorResponse', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-15T10:30:00.000Z'));
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should create an error response with message', () => {
      const response = errorResponse('Something went wrong');

      expect(response.success).toBe(false);
      expect(response.error).toBe('Something went wrong');
      expect(response.timestamp).toBe('2025-01-15T10:30:00.000Z');
    });

    it('should create an error response with empty message', () => {
      const response = errorResponse('');

      expect(response.success).toBe(false);
      expect(response.error).toBe('');
    });

    it('should not include data field', () => {
      const response = errorResponse('Error occurred');

      expect(response).not.toHaveProperty('data');
    });

    it('should handle long error messages', () => {
      const longMessage = 'A'.repeat(1000);
      const response = errorResponse(longMessage);

      expect(response.success).toBe(false);
      expect(response.error).toBe(longMessage);
      expect(response.error.length).toBe(1000);
    });

    it('should handle error messages with special characters', () => {
      const specialMessage = 'Error: "Invalid <input> & {data}';
      const response = errorResponse(specialMessage);

      expect(response.success).toBe(false);
      expect(response.error).toBe(specialMessage);
    });
  });

  describe('enum type values', () => {
    it('should support all LlmProvider values', () => {
      const providers: LlmProvider[] = [
        'openai',
        'anthropic',
        'gemini',
        'mistral',
        'cohere',
        'groq',
        'xai',
        'deepseek',
        'perplexity',
        'lm_studio',
      ];

      expect(providers).toHaveLength(10);
      expect(providers).toContain('openai');
      expect(providers).toContain('anthropic');
      expect(providers).toContain('gemini');
      expect(providers).toContain('mistral');
      expect(providers).toContain('cohere');
      expect(providers).toContain('groq');
      expect(providers).toContain('xai');
      expect(providers).toContain('deepseek');
      expect(providers).toContain('perplexity');
      expect(providers).toContain('lm_studio');
    });

    it('should support all HttpMethod values', () => {
      const methods: HttpMethod[] = ['GET', 'POST'];

      expect(methods).toHaveLength(2);
      expect(methods).toContain('GET');
      expect(methods).toContain('POST');
    });
  });

  describe('JsonSchema type', () => {
    it('should support basic object schema', () => {
      const schema: JsonSchema = {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'number' },
        },
        required: ['name'],
      };

      expect(schema.type).toBe('object');
      expect(schema.properties?.name?.type).toBe('string');
      expect(schema.required).toContain('name');
    });

    it('should support array schema', () => {
      const schema: JsonSchema = {
        type: 'array',
        items: {
          type: 'string',
        },
      };

      expect(schema.type).toBe('array');
      expect(schema.items?.type).toBe('string');
    });

    it('should support enum values', () => {
      const schema: JsonSchema = {
        type: 'string',
        enum: ['red', 'green', 'blue'],
      };

      expect(schema.enum).toEqual(['red', 'green', 'blue']);
    });

    it('should support validation constraints', () => {
      const schema: JsonSchema = {
        type: 'string',
        minLength: 1,
        maxLength: 100,
        pattern: '^[a-z]+$',
      };

      expect(schema.minLength).toBe(1);
      expect(schema.maxLength).toBe(100);
      expect(schema.pattern).toBe('^[a-z]+$');
    });

    it('should support numeric constraints', () => {
      const schema: JsonSchema = {
        type: 'number',
        minimum: 0,
        maximum: 100,
      };

      expect(schema.minimum).toBe(0);
      expect(schema.maximum).toBe(100);
    });

    it('should support nested schemas', () => {
      const schema: JsonSchema = {
        type: 'object',
        properties: {
          address: {
            type: 'object',
            properties: {
              street: { type: 'string' },
              city: { type: 'string' },
            },
          },
        },
      };

      expect(schema.properties?.address?.properties?.street?.type).toBe(
        'string'
      );
    });

    it('should support additionalProperties', () => {
      const schema: JsonSchema = {
        type: 'object',
        additionalProperties: false,
      };

      expect(schema.additionalProperties).toBe(false);
    });
  });

  describe('entity types', () => {
    it('should create valid User object', () => {
      const user: User = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        firebase_uid: 'firebase-uid-123',
        email: 'test@example.com',
        display_name: 'Test User',
        created_at: new Date(),
        updated_at: new Date(),
      };

      expect(user.id).toBeDefined();
      expect(user.firebase_uid).toBe('firebase-uid-123');
      expect(user.email).toBe('test@example.com');
    });

    it('should allow nullable User fields', () => {
      const user: User = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        firebase_uid: 'firebase-uid-123',
        email: null,
        display_name: null,
        created_at: null,
        updated_at: null,
      };

      expect(user.email).toBeNull();
      expect(user.display_name).toBeNull();
    });

    it('should create valid Project object', () => {
      const project: Project = {
        uuid: '123e4567-e89b-12d3-a456-426614174000',
        entity_id: '123e4567-e89b-12d3-a456-426614174001',
        project_name: 'my-project',
        display_name: 'My Project',
        description: 'A test project',
        is_active: true,
        api_key_prefix: 'sk_live_ab',
        api_key_created_at: new Date(),
        created_at: new Date(),
        updated_at: new Date(),
      };

      expect(project.entity_id).toBeDefined();
      expect(project.project_name).toBe('my-project');
      expect(project.is_active).toBe(true);
    });

    it('should create valid UsageAnalytics object', () => {
      const analytics: UsageAnalytics = {
        uuid: '123e4567-e89b-12d3-a456-426614174000',
        endpoint_id: '123e4567-e89b-12d3-a456-426614174001',
        timestamp: new Date(),
        success: true,
        error_message: null,
        tokens_input: 100,
        tokens_output: 50,
        latency_ms: 500,
        estimated_cost_cents: 10,
        request_metadata: { user_agent: 'test-agent' },
      };

      expect(analytics.success).toBe(true);
      expect(analytics.tokens_input).toBe(100);
      expect(analytics.estimated_cost_cents).toBe(10);
    });

    it('should create valid UserSettings object with is_default flag', () => {
      const settings: UserSettings = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        firebase_uid: 'firebase-uid-123',
        organization_name: 'My Org',
        organization_path: 'my_org',
        is_default: false,
        created_at: new Date(),
        updated_at: new Date(),
      };

      expect(settings.is_default).toBe(false);
      expect(settings.organization_path).toBe('my_org');
    });

    it('should support default UserSettings (not yet saved)', () => {
      const defaultSettings: UserSettings = {
        id: null,
        firebase_uid: 'firebase-uid-123',
        organization_name: null,
        organization_path: 'abc12345',
        is_default: true,
        created_at: null,
        updated_at: null,
      };

      expect(defaultSettings.is_default).toBe(true);
      expect(defaultSettings.id).toBeNull();
    });
  });

  describe('analytics aggregation types', () => {
    it('should create valid UsageAggregate object', () => {
      const aggregate: UsageAggregate = {
        total_requests: 100,
        successful_requests: 95,
        failed_requests: 5,
        total_tokens_input: 10000,
        total_tokens_output: 5000,
        total_estimated_cost_cents: 150,
        average_latency_ms: 250,
      };

      expect(aggregate.total_requests).toBe(100);
      expect(aggregate.successful_requests + aggregate.failed_requests).toBe(
        100
      );
    });

    it('should create valid UsageByEndpoint object', () => {
      const byEndpoint: UsageByEndpoint = {
        endpoint_id: '123e4567-e89b-12d3-a456-426614174000',
        endpoint_name: 'analyze-text',
        total_requests: 50,
        successful_requests: 48,
        failed_requests: 2,
        total_tokens_input: 5000,
        total_tokens_output: 2500,
        total_estimated_cost_cents: 75,
        average_latency_ms: 200,
      };

      expect(byEndpoint.endpoint_id).toBeDefined();
      expect(byEndpoint.endpoint_name).toBe('analyze-text');
    });

    it('should create valid AnalyticsResponse object', () => {
      const response: AnalyticsResponse = {
        aggregate: {
          total_requests: 100,
          successful_requests: 95,
          failed_requests: 5,
          total_tokens_input: 10000,
          total_tokens_output: 5000,
          total_estimated_cost_cents: 150,
          average_latency_ms: 250,
        },
        by_endpoint: [
          {
            endpoint_id: '123e4567-e89b-12d3-a456-426614174000',
            endpoint_name: 'endpoint-1',
            total_requests: 50,
            successful_requests: 48,
            failed_requests: 2,
            total_tokens_input: 5000,
            total_tokens_output: 2500,
            total_estimated_cost_cents: 75,
            average_latency_ms: 200,
          },
        ],
      };

      expect(response.aggregate.total_requests).toBe(100);
      expect(response.by_endpoint).toHaveLength(1);
    });
  });

  describe('LLM request/response types', () => {
    it('should create valid LlmRequest object', () => {
      const request: LlmRequest = {
        prompt: 'Analyze this text',
        system_prompt: 'You are a helpful assistant',
        output_schema: { type: 'object' },
        model: 'gpt-4o',
        temperature: 0.7,
        max_tokens: 1000,
      };

      expect(request.prompt).toBe('Analyze this text');
      expect(request.temperature).toBe(0.7);
    });

    it('should create valid LlmResponse object', () => {
      const response: LlmResponse = {
        content: { result: 'Analysis complete' },
        raw_response: '{"result": "Analysis complete"}',
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
        },
        model: 'gpt-4o',
        provider: 'openai',
      };

      expect(response.usage.total_tokens).toBe(150);
      expect(response.provider).toBe('openai');
    });
  });

  describe('AI execution types', () => {
    it('should create valid AiExecutionRequest object', () => {
      const request: AiExecutionRequest = {
        input: { text: 'Hello world' },
      };

      expect(request.input).toEqual({ text: 'Hello world' });
    });

    it('should create valid AiExecutionResponse object', () => {
      const response: AiExecutionResponse = {
        output: { sentiment: 'positive' },
        usage: {
          tokens_input: 10,
          tokens_output: 5,
          latency_ms: 200,
          estimated_cost_cents: 1,
        },
      };

      expect(response.output).toEqual({ sentiment: 'positive' });
      expect(response.usage.latency_ms).toBe(200);
    });
  });

  describe('ApiHelper types', () => {
    it('should create valid PromptInput object', () => {
      const input: PromptInput = {
        inputData: { text: 'Sample text' },
        outputSchema: { type: 'object' },
        instructions: 'Process the text',
        context: 'You are a processor',
        provider: 'openai',
      };

      expect(input.provider).toBe('openai');
      expect(input.instructions).toBe('Process the text');
    });

    it('should create valid ApiHelperRequestInput object', () => {
      const input: ApiHelperRequestInput = {
        prompt: 'Analyze this',
        outputSchema: { type: 'object' },
        provider: 'anthropic',
        providerConfig: {
          apiKey: 'sk-test',
          model: 'claude-3-5-sonnet-20241022',
        },
        options: {
          temperature: 0.5,
          maxTokens: 500,
        },
      };

      expect(input.provider).toBe('anthropic');
      expect(input.providerConfig.model).toBe('claude-3-5-sonnet-20241022');
    });

    it('should create valid ApiHelperRequestOutput object', () => {
      const output: ApiHelperRequestOutput = {
        apiPayload: {
          model: 'gpt-4o',
          messages: [{ role: 'user', content: 'Hello' }],
        },
        endpointUrl: 'https://api.openai.com/v1/chat/completions',
        provider: 'openai',
      };

      expect(output.endpointUrl).toContain('openai.com');
      expect(output.provider).toBe('openai');
    });
  });

  describe('API key types', () => {
    it('should create valid RefreshApiKeyResponse object', () => {
      const response: RefreshApiKeyResponse = {
        api_key: 'shyft_test_abcdefghijklmnop1234567890',
        api_key_prefix: 'shyft_test_ab...',
        api_key_created_at: '2025-01-15T10:30:00.000Z',
      };

      expect(response.api_key).toContain('shyft_test_');
      expect(response.api_key_prefix).toContain('...');
    });
  });

  describe('health check types', () => {
    it('should create valid HealthCheckData object', () => {
      const health: HealthCheckData = {
        name: 'shapeshyft_api',
        version: '1.0.16',
        status: 'healthy',
      };

      expect(health.name).toBe('shapeshyft_api');
      expect(health.status).toBe('healthy');
    });
  });

  describe('request body types', () => {
    it('should create valid UserCreateRequest', () => {
      const request: UserCreateRequest = {
        firebase_uid: 'firebase-uid-123',
        email: 'test@example.com',
        display_name: 'Test User',
      };

      expect(request.firebase_uid).toBe('firebase-uid-123');
      expect(request.email).toBe('test@example.com');
    });

    it('should allow undefined optional fields in UserCreateRequest', () => {
      const request: UserCreateRequest = {
        firebase_uid: 'firebase-uid-123',
        email: undefined,
        display_name: undefined,
      };

      expect(request.firebase_uid).toBeDefined();
      expect(request.email).toBeUndefined();
    });

    it('should create valid UserUpdateRequest', () => {
      const request: UserUpdateRequest = {
        email: 'new@example.com',
        display_name: 'New Name',
      };

      expect(request.email).toBe('new@example.com');
      expect(request.display_name).toBe('New Name');
    });

    it('should allow partial UserUpdateRequest', () => {
      const request: UserUpdateRequest = {
        email: undefined,
        display_name: 'Only Name',
      };

      expect(request.email).toBeUndefined();
      expect(request.display_name).toBe('Only Name');
    });

    it('should create valid UserSettingsUpdateRequest', () => {
      const request: UserSettingsUpdateRequest = {
        organization_name: 'New Org',
        organization_path: 'new_org',
      };

      expect(request.organization_name).toBe('New Org');
      expect(request.organization_path).toBe('new_org');
    });

    it('should create valid ProjectCreateRequest', () => {
      const request: ProjectCreateRequest = {
        project_name: 'my-project',
        display_name: 'My Project',
        description: 'A test project',
      };

      expect(request.project_name).toBe('my-project');
      expect(request.display_name).toBe('My Project');
      expect(request.description).toBe('A test project');
    });

    it('should create valid ProjectUpdateRequest', () => {
      const request: ProjectUpdateRequest = {
        project_name: 'updated-project',
        display_name: 'Updated Project',
        description: undefined,
        is_active: false,
      };

      expect(request.is_active).toBe(false);
    });
  });

  describe('query parameter types', () => {
    it('should create valid ProjectQueryParams', () => {
      const params: ProjectQueryParams = {
        is_active: 'true',
      };

      expect(params.is_active).toBe('true');
    });

    it('should allow undefined ProjectQueryParams fields', () => {
      const params: ProjectQueryParams = {
        is_active: undefined,
      };

      expect(params.is_active).toBeUndefined();
    });

    it('should create valid EndpointQueryParams', () => {
      const params: EndpointQueryParams = {
        is_active: 'false',
      };

      expect(params.is_active).toBe('false');
    });

    it('should create valid UsageAnalyticsQueryParams with all fields', () => {
      const params: UsageAnalyticsQueryParams = {
        endpoint_id: '123e4567-e89b-12d3-a456-426614174000',
        project_id: '123e4567-e89b-12d3-a456-426614174001',
        start_date: '2025-01-01',
        end_date: '2025-01-31',
        success: 'true',
      };

      expect(params.endpoint_id).toBeDefined();
      expect(params.project_id).toBeDefined();
      expect(params.start_date).toBe('2025-01-01');
      expect(params.end_date).toBe('2025-01-31');
      expect(params.success).toBe('true');
    });

    it('should allow partial UsageAnalyticsQueryParams', () => {
      const params: UsageAnalyticsQueryParams = {
        endpoint_id: undefined,
        project_id: '123e4567-e89b-12d3-a456-426614174001',
        start_date: undefined,
        end_date: undefined,
        success: undefined,
      };

      expect(params.project_id).toBeDefined();
      expect(params.start_date).toBeUndefined();
    });
  });

  describe('additional response types', () => {
    it('should create valid UsageByDate object', () => {
      const byDate: UsageByDate = {
        date: '2025-01-15',
        total_requests: 100,
        successful_requests: 95,
        failed_requests: 5,
        total_tokens_input: 10000,
        total_tokens_output: 5000,
        total_estimated_cost_cents: 150,
        average_latency_ms: 250,
      };

      expect(byDate.date).toBe('2025-01-15');
      expect(byDate.total_requests).toBe(100);
    });

    it('should create valid AiPromptResponse object', () => {
      const response: AiPromptResponse = {
        prompt:
          'Given the following input:\n{"text": "Hello"}\n\nAnalyze the sentiment.',
      };

      expect(response.prompt).toContain('sentiment');
    });

    it('should create valid GetApiKeyResponse object', () => {
      const response: GetApiKeyResponse = {
        api_key: 'shyft_live_abcdefghijklmnop1234567890abcdef',
      };

      expect(response.api_key).toContain('shyft_live_');
    });
  });

  // ===========================================================================
  // Improvement #2: Tests for Runtime Functions
  // ===========================================================================

  describe('estimateCost', () => {
    const pricing: ModelPricing = {
      input: 300, // $3.00 per 1M input tokens
      output: 1500, // $15.00 per 1M output tokens
    };

    it('should calculate cost for typical token usage', () => {
      const cost = estimateCost(pricing, 1000, 500);
      // (1000/1M)*300 + (500/1M)*1500 = 0.3 + 0.75 = 1.05 cents
      expect(cost).toBe(1.05);
    });

    it('should return 0 for zero tokens', () => {
      expect(estimateCost(pricing, 0, 0)).toBe(0);
    });

    it('should handle zero input tokens with non-zero output', () => {
      const cost = estimateCost(pricing, 0, 1_000_000);
      expect(cost).toBe(1500);
    });

    it('should handle zero output tokens with non-zero input', () => {
      const cost = estimateCost(pricing, 1_000_000, 0);
      expect(cost).toBe(300);
    });

    it('should calculate cost for exactly 1M tokens', () => {
      const cost = estimateCost(pricing, 1_000_000, 1_000_000);
      expect(cost).toBe(1800); // 300 + 1500
    });

    it('should handle very large token counts', () => {
      const cost = estimateCost(pricing, 100_000_000, 50_000_000);
      // (100M/1M)*300 + (50M/1M)*1500 = 30000 + 75000 = 105000
      expect(cost).toBe(105000);
    });

    it('should round to 2 decimal places', () => {
      // Create a scenario that produces many decimal places
      const cost = estimateCost(pricing, 1, 1);
      // (1/1M)*300 + (1/1M)*1500 = 0.0003 + 0.0015 = 0.0018 -> rounds to 0
      expect(cost).toBe(0);
    });

    it('should handle pricing with zero cost', () => {
      const freePricing: ModelPricing = { input: 0, output: 0 };
      expect(estimateCost(freePricing, 1_000_000, 1_000_000)).toBe(0);
    });
  });

  describe('estimateMultimodalCost', () => {
    const fullPricing: ModelPricing = {
      input: 300,
      output: 1500,
      imageInput: 50, // 50 cents per input image
      imageOutput: 200, // $2 per generated image
      audioInput: 10, // 10 cents per minute
      audioOutput: 30, // 30 cents per minute
      videoInput: 100, // $1 per minute
      videoOutput: 500, // $5 per minute
    };

    it('should return 0 for empty usage', () => {
      expect(estimateMultimodalCost(fullPricing, {})).toBe(0);
    });

    it('should calculate text-only cost', () => {
      const usage: MultimodalUsage = {
        inputTokens: 1_000_000,
        outputTokens: 1_000_000,
      };
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(1800);
    });

    it('should calculate image-only cost', () => {
      const usage: MultimodalUsage = {
        imagesInput: 3,
        imagesOutput: 1,
      };
      // 3 * 50 + 1 * 200 = 150 + 200 = 350
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(350);
    });

    it('should calculate audio-only cost', () => {
      const usage: MultimodalUsage = {
        audioInputMinutes: 5,
        audioOutputMinutes: 2,
      };
      // 5 * 10 + 2 * 30 = 50 + 60 = 110
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(110);
    });

    it('should calculate video-only cost', () => {
      const usage: MultimodalUsage = {
        videoInputMinutes: 2,
        videoOutputMinutes: 1,
      };
      // 2 * 100 + 1 * 500 = 200 + 500 = 700
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(700);
    });

    it('should calculate combined multimodal cost', () => {
      const usage: MultimodalUsage = {
        inputTokens: 1_000_000,
        outputTokens: 500_000,
        imagesInput: 2,
        imagesOutput: 1,
        audioInputMinutes: 3,
      };
      // text: 300 + 750 = 1050
      // images: 2*50 + 1*200 = 300
      // audio: 3*10 = 30
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(1380);
    });

    it('should ignore media usage when pricing lacks media fields', () => {
      const textOnlyPricing: ModelPricing = { input: 300, output: 1500 };
      const usage: MultimodalUsage = {
        inputTokens: 1_000_000,
        imagesInput: 5,
        audioInputMinutes: 10,
        videoOutputMinutes: 2,
      };
      // Only text cost counts: 300
      expect(estimateMultimodalCost(textOnlyPricing, usage)).toBe(300);
    });

    it('should handle zero values in usage', () => {
      const usage: MultimodalUsage = {
        inputTokens: 0,
        outputTokens: 0,
        imagesInput: 0,
        imagesOutput: 0,
      };
      expect(estimateMultimodalCost(fullPricing, usage)).toBe(0);
    });

    it('should round to 2 decimal places', () => {
      const usage: MultimodalUsage = { inputTokens: 333 };
      const cost = estimateMultimodalCost(fullPricing, usage);
      // (333/1M)*300 = 0.0999 -> rounds to 0.1
      expect(cost).toBe(0.1);
    });
  });

  describe('formatCost', () => {
    it('should format sub-cent costs with 4 decimals', () => {
      // 0.5 cents = $0.005
      expect(formatCost(0.5)).toBe('$0.0050');
    });

    it('should format zero cost', () => {
      expect(formatCost(0)).toBe('$0.0000');
    });

    it('should format costs under $1 with 3 decimals', () => {
      // 50 cents = $0.50
      expect(formatCost(50)).toBe('$0.500');
    });

    it('should format costs $1 and above with 2 decimals', () => {
      // 150 cents = $1.50
      expect(formatCost(150)).toBe('$1.50');
    });

    it('should format exactly $1', () => {
      expect(formatCost(100)).toBe('$1.00');
    });

    it('should format large costs', () => {
      // 100000 cents = $1000.00
      expect(formatCost(100000)).toBe('$1000.00');
    });

    it('should format 1 cent exactly at the $0.01 boundary', () => {
      // 1 cent = $0.01 -> this is < $1 but >= $0.01
      expect(formatCost(1)).toBe('$0.010');
    });

    it('should format very small sub-cent costs', () => {
      // 0.01 cents = $0.0001
      expect(formatCost(0.01)).toBe('$0.0001');
    });
  });

  describe('formatCostPerMillion', () => {
    it('should format standard pricing', () => {
      const pricing: ModelPricing = { input: 300, output: 1500 };
      expect(formatCostPerMillion(pricing)).toBe(
        '$3.00 / $15.00 per 1M tokens'
      );
    });

    it('should format zero pricing', () => {
      const pricing: ModelPricing = { input: 0, output: 0 };
      expect(formatCostPerMillion(pricing)).toBe('$0.00 / $0.00 per 1M tokens');
    });

    it('should format cheap pricing with decimals', () => {
      const pricing: ModelPricing = { input: 15, output: 60 };
      expect(formatCostPerMillion(pricing)).toBe('$0.15 / $0.60 per 1M tokens');
    });

    it('should format expensive pricing', () => {
      const pricing: ModelPricing = { input: 6000, output: 18000 };
      expect(formatCostPerMillion(pricing)).toBe(
        '$60.00 / $180.00 per 1M tokens'
      );
    });
  });

  describe('detectRequiredCapabilities', () => {
    it('should return empty object for null schemas', () => {
      expect(detectRequiredCapabilities(null, null)).toEqual({});
    });

    it('should return empty object for undefined schemas', () => {
      expect(detectRequiredCapabilities(undefined, undefined)).toEqual({});
    });

    it('should return empty object for schemas without media types', () => {
      const schema = {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'number' },
        },
      };
      expect(detectRequiredCapabilities(schema, schema)).toEqual({});
    });

    it('should detect image input from contentMediaType', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          photo: {
            type: 'string',
            contentMediaType: 'image/png',
            contentEncoding: 'base64',
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        visionInput: true,
      });
    });

    it('should detect audio input from contentMediaType', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          recording: {
            type: 'string',
            contentMediaType: 'audio/mp3',
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        audioInput: true,
      });
    });

    it('should detect video input from contentMediaType', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          clip: {
            type: 'string',
            contentMediaType: 'video/mp4',
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        videoInput: true,
      });
    });

    it('should detect image output from output schema', () => {
      const outputSchema = {
        type: 'object',
        properties: {
          generated_image: {
            type: 'string',
            contentMediaType: 'image/jpeg',
          },
        },
      };
      expect(detectRequiredCapabilities(null, outputSchema)).toEqual({
        imageOutput: true,
      });
    });

    it('should detect audio output from output schema', () => {
      const outputSchema = {
        type: 'object',
        properties: {
          speech: {
            type: 'string',
            contentMediaType: 'audio/wav',
          },
        },
      };
      expect(detectRequiredCapabilities(null, outputSchema)).toEqual({
        audioOutput: true,
      });
    });

    it('should detect multiple media types from a single schema', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          photo: {
            type: 'string',
            contentMediaType: 'image/png',
          },
          recording: {
            type: 'string',
            contentMediaType: 'audio/mp3',
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        visionInput: true,
        audioInput: true,
      });
    });

    it('should detect both input and output capabilities', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          photo: {
            type: 'string',
            contentMediaType: 'image/png',
          },
        },
      };
      const outputSchema = {
        type: 'object',
        properties: {
          result_image: {
            type: 'string',
            contentMediaType: 'image/jpeg',
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, outputSchema)).toEqual({
        visionInput: true,
        imageOutput: true,
      });
    });

    it('should detect media types in deeply nested schemas', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          level1: {
            type: 'object',
            properties: {
              level2: {
                type: 'object',
                properties: {
                  deep_image: {
                    type: 'string',
                    contentMediaType: 'image/webp',
                  },
                },
              },
            },
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        visionInput: true,
      });
    });

    it('should detect media types in array items', () => {
      const inputSchema = {
        type: 'object',
        properties: {
          images: {
            type: 'array',
            items: {
              type: 'string',
              contentMediaType: 'image/png',
            },
          },
        },
      };
      expect(detectRequiredCapabilities(inputSchema, null)).toEqual({
        visionInput: true,
      });
    });

    it('should handle empty schemas', () => {
      expect(detectRequiredCapabilities({}, {})).toEqual({});
    });

    it('should handle schemas with empty properties', () => {
      const schema = { type: 'object', properties: {} };
      expect(detectRequiredCapabilities(schema, schema)).toEqual({});
    });

    it('should ignore unrecognized contentMediaType prefixes', () => {
      const schema = {
        type: 'object',
        properties: {
          data: {
            type: 'string',
            contentMediaType: 'application/json',
          },
        },
      };
      expect(detectRequiredCapabilities(schema, null)).toEqual({});
    });
  });

  // ===========================================================================
  // Improvement #3: PROVIDER_MODELS and isValidModelForProvider
  // ===========================================================================

  describe('PROVIDER_MODELS', () => {
    it('should have an entry for every LLM provider', () => {
      for (const provider of LLM_PROVIDERS) {
        expect(PROVIDER_MODELS).toHaveProperty(provider);
        expect(Array.isArray(PROVIDER_MODELS[provider])).toBe(true);
      }
    });

    it('should have non-empty model lists for cloud providers', () => {
      const cloudProviders: LlmProvider[] = [
        'openai',
        'anthropic',
        'gemini',
        'mistral',
        'cohere',
        'groq',
        'xai',
        'deepseek',
        'perplexity',
      ];
      for (const provider of cloudProviders) {
        expect(PROVIDER_MODELS[provider].length).toBeGreaterThan(0);
      }
    });

    it('should have an empty list for lm_studio (any model accepted)', () => {
      expect(PROVIDER_MODELS['lm_studio']).toHaveLength(0);
    });

    it('should contain known OpenAI models', () => {
      expect(PROVIDER_MODELS['openai']).toContain('gpt-5.6-sol');
      expect(PROVIDER_MODELS['openai']).toContain('gpt-5.6-terra');
      expect(PROVIDER_MODELS['openai']).toContain('gpt-4o');
    });

    it('should contain known Anthropic models', () => {
      expect(PROVIDER_MODELS['anthropic']).toContain('claude-opus-5');
      expect(PROVIDER_MODELS['anthropic']).toContain('claude-sonnet-5');
      expect(PROVIDER_MODELS['anthropic']).toContain('claude-haiku-4-5');
    });

    it('should use dateless Claude IDs for the 4.6 generation and later', () => {
      // Claude 4.6+ IDs are pinned snapshots without a date suffix; appending
      // one (claude-sonnet-4-6-20260217) is rejected by the Claude API.
      for (const model of PROVIDER_MODELS['anthropic']) {
        if (/^claude-(opus|sonnet)-4-[6-9]/.test(model)) {
          expect(model).not.toMatch(/-\d{8}$/);
        }
      }
    });

    it('should contain known Gemini models', () => {
      expect(PROVIDER_MODELS['gemini']).toContain('gemini-2.5-pro');
      expect(PROVIDER_MODELS['gemini']).toContain('gemini-2.5-flash');
    });
  });

  describe('isValidModelForProvider', () => {
    it('should return true for valid OpenAI model', () => {
      expect(isValidModelForProvider('openai', 'gpt-4o')).toBe(true);
    });

    it('should return false for invalid OpenAI model', () => {
      expect(isValidModelForProvider('openai', 'not-a-real-model')).toBe(false);
    });

    it('should return true for valid Anthropic model', () => {
      expect(isValidModelForProvider('anthropic', 'claude-sonnet-5')).toBe(
        true
      );
    });

    it('should return false for model from wrong provider', () => {
      expect(isValidModelForProvider('openai', 'claude-sonnet-5')).toBe(false);
    });

    it('should return true for any lm_studio model string', () => {
      expect(isValidModelForProvider('lm_studio', 'any-custom-model')).toBe(
        true
      );
      expect(isValidModelForProvider('lm_studio', '')).toBe(true);
      expect(isValidModelForProvider('lm_studio', 'my-local-llama-3')).toBe(
        true
      );
    });

    it('should return true for valid Gemini model', () => {
      expect(isValidModelForProvider('gemini', 'gemini-2.5-pro')).toBe(true);
    });

    it('should return true for valid DeepSeek model', () => {
      expect(isValidModelForProvider('deepseek', 'deepseek-v4-pro')).toBe(true);
      expect(isValidModelForProvider('deepseek', 'deepseek-v4-flash')).toBe(
        true
      );
    });

    it('should return false for empty string on cloud providers', () => {
      expect(isValidModelForProvider('openai', '')).toBe(false);
      expect(isValidModelForProvider('anthropic', '')).toBe(false);
    });

    it('should list models without duplicates for every cloud provider', () => {
      // Deliberately an invariant rather than hardcoded counts: provider
      // lineups turn over every few weeks, and a count assertion only ever
      // fails for the catalog being refreshed -- which is the correct action,
      // not a regression.
      for (const [provider, models] of Object.entries(PROVIDER_MODELS)) {
        if (provider === 'lm_studio') continue;
        expect(models.length).toBeGreaterThan(0);
        expect(new Set(models).size).toBe(models.length);
        for (const model of models) {
          expect(model.trim()).toBe(model);
          expect(model.length).toBeGreaterThan(0);
        }
      }
    });
  });

  // ===========================================================================
  // Improvement #4: BooleanQueryParam type
  // ===========================================================================

  describe('BooleanQueryParam type', () => {
    it('should accept "true" as a valid BooleanQueryParam', () => {
      const param: BooleanQueryParam = 'true';
      expect(param).toBe('true');
    });

    it('should accept "false" as a valid BooleanQueryParam', () => {
      const param: BooleanQueryParam = 'false';
      expect(param).toBe('false');
    });

    it('should accept undefined as a valid BooleanQueryParam', () => {
      const param: BooleanQueryParam = undefined;
      expect(param).toBeUndefined();
    });

    it('should work in ProjectQueryParams with narrowed type', () => {
      const params: ProjectQueryParams = { is_active: 'true' };
      expect(params.is_active).toBe('true');

      const params2: ProjectQueryParams = { is_active: undefined };
      expect(params2.is_active).toBeUndefined();
    });

    it('should work in EndpointQueryParams with narrowed type', () => {
      const params: EndpointQueryParams = { is_active: 'false' };
      expect(params.is_active).toBe('false');
    });

    it('should work in UsageAnalyticsQueryParams success field', () => {
      const params: UsageAnalyticsQueryParams = {
        endpoint_id: 'abc',
        project_id: undefined,
        start_date: undefined,
        end_date: undefined,
        success: 'true',
      };
      expect(params.success).toBe('true');
    });
  });

  describe('response with typed data', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-15T10:30:00.000Z'));
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should create typed success response with Project', () => {
      const project: Project = {
        uuid: '123e4567-e89b-12d3-a456-426614174000',
        entity_id: '123e4567-e89b-12d3-a456-426614174001',
        project_name: 'test',
        display_name: 'Test',
        description: null,
        is_active: true,
        api_key_prefix: null,
        api_key_created_at: null,
        created_at: new Date(),
        updated_at: new Date(),
      };

      const response = successResponse(project);

      expect(response.success).toBe(true);
      expect(response.data.project_name).toBe('test');
      expect(response.data.entity_id).toBeDefined();
    });

    it('should create typed success response with AnalyticsResponse', () => {
      const analytics: AnalyticsResponse = {
        aggregate: {
          total_requests: 0,
          successful_requests: 0,
          failed_requests: 0,
          total_tokens_input: 0,
          total_tokens_output: 0,
          total_estimated_cost_cents: 0,
          average_latency_ms: 0,
        },
        by_endpoint: [],
      };

      const response = successResponse(analytics);

      expect(response.success).toBe(true);
      expect(response.data.aggregate.total_requests).toBe(0);
      expect(response.data.by_endpoint).toEqual([]);
    });
  });
  describe('output limit types', () => {
    it('should expose a default output ceiling that is a sane positive integer', () => {
      expect(Number.isInteger(DEFAULT_MAX_OUTPUT_TOKENS)).toBe(true);
      expect(DEFAULT_MAX_OUTPUT_TOKENS).toBeGreaterThan(0);
      // Large enough for a real structured answer, small enough that a runaway
      // is caught in seconds rather than minutes.
      expect(DEFAULT_MAX_OUTPUT_TOKENS).toBeGreaterThanOrEqual(1000);
      expect(DEFAULT_MAX_OUTPUT_TOKENS).toBeLessThanOrEqual(32000);
    });

    it('should enumerate every finish reason exactly once', () => {
      expect(new Set(FINISH_REASONS).size).toBe(FINISH_REASONS.length);
      expect(FINISH_REASONS).toContain('stop');
      expect(FINISH_REASONS).toContain('length');
    });

    it('should allow an endpoint to carry a null output ceiling meaning no limit', () => {
      const unlimited: Pick<EndpointBase, 'max_output_tokens'> = {
        max_output_tokens: null,
      };
      const capped: Pick<EndpointBase, 'max_output_tokens'> = {
        max_output_tokens: 4000,
      };

      expect(unlimited.max_output_tokens).toBeNull();
      expect(capped.max_output_tokens).toBe(4000);
    });

    it('should report the finish reason alongside token usage', () => {
      const reason: FinishReason = 'length';
      const response: AiExecutionResponse = {
        output: {},
        usage: {
          tokens_input: 3143,
          tokens_output: 948,
          latency_ms: 200,
          estimated_cost_cents: 1,
          finish_reason: reason,
        },
        truncated: true,
      };

      expect(response.usage.finish_reason).toBe('length');
      expect(response.truncated).toBe(true);
    });
  });
});
