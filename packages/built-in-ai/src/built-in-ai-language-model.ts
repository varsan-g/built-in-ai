import {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
  LoadSettingError,
  JSONValue,
  LanguageModelV2Message,
  LanguageModelV2FunctionTool,
  LanguageModelV2ToolCall,
} from "@ai-sdk/provider";
import { convertToBuiltInAIMessages } from "./convert-to-built-in-ai-messages";
import {
  buildToolSystemPrompt,
  parseToolCallRequest,
} from "./tool-calling-polyfill";

export type BuiltInAIChatModelId = "text";

export interface BuiltInAIChatSettings extends LanguageModelCreateOptions {
  /**
   * Expected input types for the session, for multimodal inputs.
   */
  expectedInputs?: Array<{
    type: "text" | "image" | "audio";
    languages?: string[];
  }>;
}

/**
 * Check if the browser supports the built-in AI API
 * @returns true if the browser supports the built-in AI API, false otherwise
 */
export function doesBrowserSupportBuiltInAI(): boolean {
  return typeof LanguageModel !== "undefined";
}

/**
 * Check if the Prompt API is available
 * @deprecated Use `doesBrowserSupportBuiltInAI()` instead for clearer naming
 * @returns true if the browser supports the built-in AI API, false otherwise
 */
export function isBuiltInAIModelAvailable(): boolean {
  return typeof LanguageModel !== "undefined";
}

type BuiltInAIConfig = {
  provider: string;
  modelId: BuiltInAIChatModelId;
  options: BuiltInAIChatSettings;
};

/**
 * Detect if the prompt contains multimodal content
 */
function hasMultimodalContent(prompt: LanguageModelV2Prompt): boolean {
  for (const message of prompt) {
    if (message.role === "user") {
      for (const part of message.content) {
        if (part.type === "file") {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Get expected inputs based on prompt content
 */
function getExpectedInputs(
  prompt: LanguageModelV2Prompt,
): Array<{ type: "text" | "image" | "audio" }> {
  const inputs = new Set<"text" | "image" | "audio">();
  // Don't add text by default - it's assumed by the Prompt API

  for (const message of prompt) {
    if (message.role === "user") {
      for (const part of message.content) {
        if (part.type === "file") {
          if (part.mediaType?.startsWith("image/")) {
            inputs.add("image");
          } else if (part.mediaType?.startsWith("audio/")) {
            inputs.add("audio");
          }
        }
      }
    }
  }

  return Array.from(inputs).map((type) => ({ type }));
}

export class BuiltInAIChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";
  readonly modelId: BuiltInAIChatModelId;
  readonly provider = "browser-ai";

  private readonly config: BuiltInAIConfig;
  private session!: LanguageModel;

  constructor(
    modelId: BuiltInAIChatModelId,
    options: BuiltInAIChatSettings = {},
  ) {
    this.modelId = modelId;
    this.config = {
      provider: this.provider,
      modelId,
      options,
    };
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^https?:\/\/.+$/],
    "audio/*": [/^https?:\/\/.+$/],
  };

  private async getSession(
    options?: LanguageModelCreateOptions,
    expectedInputs?: Array<{ type: "text" | "image" | "audio" }>,
    systemMessage?: string,
    onDownloadProgress?: (progress: number) => void,
  ): Promise<LanguageModel> {
    if (typeof LanguageModel === "undefined") {
      throw new LoadSettingError({
        message:
          "Prompt API is not available. This library requires Chrome or Edge browser with built-in AI capabilities.",
      });
    }

    if (this.session) return this.session;

    const availability = await LanguageModel.availability();

    if (availability === "unavailable") {
      throw new LoadSettingError({ message: "Built-in model not available" });
    }

    const mergedOptions = {
      ...this.config.options,
      ...options,
    };

    // Add system message to initialPrompts if provided
    if (systemMessage) {
      mergedOptions.initialPrompts = [
        { role: "system", content: systemMessage },
      ];
    }

    // Add expected inputs if provided
    if (expectedInputs && expectedInputs.length > 0) {
      mergedOptions.expectedInputs = expectedInputs;
    }

    // Add download progress monitoring if callback provided
    if (onDownloadProgress) {
      mergedOptions.monitor = (m: CreateMonitor) => {
        m.addEventListener("downloadprogress", (e: ProgressEvent) => {
          onDownloadProgress(e.loaded); // e.loaded is between 0 and 1
        });
      };
    }

    this.session = await LanguageModel.create(mergedOptions);

    return this.session;
  }

  private getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    tools,
  }: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const warnings: LanguageModelV2CallWarning[] = [];

    // Tool calling is now enabled by default - no warnings needed

    if (maxOutputTokens != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "maxOutputTokens",
        details: "maxOutputTokens is not supported by Prompt API",
      });
    }

    if (stopSequences != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "stopSequences",
        details: "stopSequences is not supported by Prompt API",
      });
    }

    if (topP != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topP",
        details: "topP is not supported by Prompt API",
      });
    }

    if (presencePenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "presencePenalty",
        details: "presencePenalty is not supported by Prompt API",
      });
    }

    if (frequencyPenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "frequencyPenalty",
        details: "frequencyPenalty is not supported by Prompt API",
      });
    }

    if (seed != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "seed",
        details: "seed is not supported by Prompt API",
      });
    }

    // Check if this is a multimodal prompt
    const hasMultiModalInput = hasMultimodalContent(prompt);

    // Convert messages to the DOM API format
    const { systemMessage, messages } = convertToBuiltInAIMessages(prompt);

    // Handle response format for Prompt API
    const promptOptions: LanguageModelPromptOptions &
      LanguageModelCreateCoreOptions = {};
    if (responseFormat?.type === "json") {
      promptOptions.responseConstraint = responseFormat.schema as Record<
        string,
        JSONValue
      >;
    }

    // Map supported settings
    if (temperature !== undefined) {
      promptOptions.temperature = temperature;
    }

    if (topK !== undefined) {
      promptOptions.topK = topK;
    }

    return {
      systemMessage,
      messages,
      warnings,
      promptOptions,
      hasMultiModalInput,
      expectedInputs: hasMultiModalInput
        ? getExpectedInputs(prompt)
        : undefined,
    };
  }

  /**
   * Generates a complete text response using the browser's built-in Prompt API
   * @param options
   * @returns Promise resolving to the generated content with finish reason, usage stats, and any warnings
   * @throws {LoadSettingError} When the Prompt API is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doGenerate(options: LanguageModelV2CallOptions) {
    const converted = this.getArgs(options);
    const { systemMessage, messages, warnings, promptOptions, expectedInputs } =
      converted;

    const { tools = [] } = options;

    // If tools are provided, use tool calling flow (enabled by default)
    if (tools.length > 0) {
      return this.doGenerateWithTools(
        options,
        systemMessage,
        messages,
        promptOptions,
        expectedInputs,
        warnings
      );
    }

    // Standard generation without tools
    const session = await this.getSession(
      undefined,
      expectedInputs,
      systemMessage,
    );

    const text = await session.prompt(messages, promptOptions);

    const content: LanguageModelV2Content[] = [
      {
        type: "text",
        text,
      },
    ];

    return {
      content,
      finishReason: "stop" as LanguageModelV2FinishReason,
      usage: {
        inputTokens: undefined,
        outputTokens: undefined,
        totalTokens: undefined,
      },
      request: { body: { messages, options: promptOptions } },
      warnings,
    };
  }

  /**
   * Generate with tool calling support (polyfill implementation)
   */
  private async doGenerateWithTools(
    options: LanguageModelV2CallOptions,
    systemMessage: string | undefined,
    messages: LanguageModelMessage[],
    promptOptions: LanguageModelPromptOptions & LanguageModelCreateCoreOptions,
    expectedInputs: Array<{ type: "text" | "image" | "audio" }> | undefined,
    warnings: LanguageModelV2CallWarning[]
  ) {
    const { tools = [] } = options;

    // Filter to only function tools (provider-defined tools not supported in beta)
    const functionTools = tools.filter(
      (tool): tool is LanguageModelV2FunctionTool =>
        tool.type === 'function'
    );

    // Build tool-enhanced system prompt
    const toolSystemPrompt = buildToolSystemPrompt(functionTools, systemMessage);

    const session = await this.getSession(
      undefined,
      expectedInputs,
      toolSystemPrompt,
    );

    const text = await session.prompt(messages, promptOptions);
    const parsedToolCall = parseToolCallRequest(text);

    // Build response content
    const content: LanguageModelV2Content[] = [];

    if (parsedToolCall) {
      // Tool calls detected - return them for AI SDK to execute
      const { toolCalls } = parsedToolCall;

      for (const toolCall of toolCalls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.toolCallId,
          toolName: toolCall.toolName,
          input: toolCall.input,
        });
      }

      return {
        content,
        finishReason: "tool-calls" as LanguageModelV2FinishReason,
        usage: {
          inputTokens: undefined,
          outputTokens: undefined,
          totalTokens: undefined,
        },
        request: { body: { messages, options: promptOptions } },
        warnings,
      };
    }

    // No tool call detected - return the text response
    content.push({
      type: "text",
      text,
    });

    return {
      content,
      finishReason: "stop" as LanguageModelV2FinishReason,
      usage: {
        inputTokens: undefined,
        outputTokens: undefined,
        totalTokens: undefined,
      },
      request: { body: { messages, options: promptOptions } },
      warnings,
    };
  }

  /**
   * Check the availability of the built-in AI model
   * @returns Promise resolving to "unavailable", "available", or "available-after-download"
   */
  public async availability(): Promise<Availability> {
    if (typeof LanguageModel === "undefined") {
      return "unavailable";
    }
    return LanguageModel.availability();
  }

  /**
   * Creates a session with download progress monitoring.
   *
   * @example
   * ```typescript
   * const session = await model.createSessionWithProgress(
   *   (progress) => {
   *     console.log(`Download progress: ${Math.round(progress * 100)}%`);
   *   }
   * );
   * ```
   *
   * @param onDownloadProgress Optional callback receiving progress values 0-1 during model download
   * @returns Promise resolving to a configured LanguageModel session
   * @throws {LoadSettingError} When the Prompt API is not available or model is unavailable
   */
  public async createSessionWithProgress(
    onDownloadProgress?: (progress: number) => void,
  ): Promise<LanguageModel> {
    return this.getSession(undefined, undefined, undefined, onDownloadProgress);
  }

  /**
   * Generates a streaming text response using the browser's built-in Prompt API
   * @param options
   * @returns Promise resolving to a readable stream of text chunks and request metadata
   * @throws {LoadSettingError} When the Prompt API is not available or model needs to be downloaded
   * @throws {UnsupportedFunctionalityError} When unsupported features like file input are used
   */
  public async doStream(options: LanguageModelV2CallOptions) {
    const converted = this.getArgs(options);
    const {
      systemMessage,
      messages,
      warnings,
      promptOptions,
      expectedInputs,
      hasMultiModalInput,
    } = converted;

    const { tools = [] } = options;

    // If tools are provided, use tool calling flow (enabled by default)
    if (tools.length > 0) {
      return this.doStreamWithTools(
        options,
        systemMessage,
        messages,
        promptOptions,
        expectedInputs,
        warnings
      );
    }

    // Standard streaming without tools
    const session = await this.getSession(
      undefined,
      expectedInputs,
      systemMessage,
    );

    // Pass abort signal to the native streaming method
    const streamOptions = {
      ...promptOptions,
      signal: options.abortSignal,
    };

    const promptStream = session.promptStreaming(messages, streamOptions);

    let isFirstChunk = true;
    const textId = "text-0";

    const stream = promptStream.pipeThrough(
      new TransformStream<string, LanguageModelV2StreamPart>({
        start(controller) {
          // Send stream start event with warnings
          controller.enqueue({
            type: "stream-start",
            warnings,
          });

          // Handle abort signal
          if (options.abortSignal) {
            options.abortSignal.addEventListener("abort", () => {
              controller.terminate();
            });
          }
        },

        transform(chunk, controller) {
          if (isFirstChunk) {
            // Send text start event
            controller.enqueue({
              type: "text-start",
              id: textId,
            });
            isFirstChunk = false;
          }

          // Send text delta
          controller.enqueue({
            type: "text-delta",
            id: textId,
            delta: chunk,
          });
        },

        flush(controller) {
          // Send text end event
          controller.enqueue({
            type: "text-end",
            id: textId,
          });

          // Send finish event
          controller.enqueue({
            type: "finish",
            finishReason: "stop" as LanguageModelV2FinishReason,
            usage: {
              inputTokens: session.inputUsage,
              outputTokens: undefined,
              totalTokens: undefined,
            },
          });
        },
      }),
    );

    return {
      stream,
      request: { body: { messages, options: promptOptions } },
    };
  }

  /**
   * Stream with tool calling support (polyfill implementation)
   */
  private async doStreamWithTools(
    options: LanguageModelV2CallOptions,
    systemMessage: string | undefined,
    messages: LanguageModelMessage[],
    promptOptions: LanguageModelPromptOptions & LanguageModelCreateCoreOptions,
    expectedInputs: Array<{ type: "text" | "image" | "audio" }> | undefined,
    warnings: LanguageModelV2CallWarning[]
  ) {
    const { tools = [] } = options;

    // Filter to only function tools (provider-defined tools not supported in beta)
    const functionTools = tools.filter(
      (tool): tool is LanguageModelV2FunctionTool =>
        tool.type === 'function'
    );

    // Build tool-enhanced system prompt
    const toolSystemPrompt = buildToolSystemPrompt(functionTools, systemMessage);

    const session = await this.getSession(
      undefined,
      expectedInputs,
      toolSystemPrompt,
    );

    const streamOptions = {
      ...promptOptions,
      signal: options.abortSignal,
    };

    // Create a readable stream that handles tool call detection
    const stream = new ReadableStream<LanguageModelV2StreamPart>({
      async start(controller) {
        try {
          controller.enqueue({
            type: "stream-start",
            warnings,
          });

          let response = "";
          const promptStream = session.promptStreaming(messages, streamOptions);
          const reader = promptStream.getReader();

          // LIMITATION: We must buffer the complete response before emitting anything.
          // This is because the polyfill needs to parse the full response to determine
          // if it's a tool call (JSON) or regular text. Real streaming isn't possible
          // until native tool calling support is added to the Prompt API.
          // 
          // This means users won't see token-by-token streaming when tools are enabled.
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              response += value;
            }
          } finally {
            reader.releaseLock();
          }

          const parsedToolCall = parseToolCallRequest(response);

          if (!parsedToolCall) {
            // No tool call detected - stream the final response
            const textId = "text-0";

            controller.enqueue({
              type: "text-start",
              id: textId,
            });

            // Stream the final text
            controller.enqueue({
              type: "text-delta",
              id: textId,
              delta: response,
            });

            controller.enqueue({
              type: "text-end",
              id: textId,
            });

            controller.enqueue({
              type: "finish",
              finishReason: "stop" as LanguageModelV2FinishReason,
              usage: {
                inputTokens: session.inputUsage,
                outputTokens: undefined,
                totalTokens: undefined,
              },
            });

            controller.close();
            return;
          }

          // Tool calls detected - emit them for AI SDK to execute
          const { toolCalls } = parsedToolCall;

          for (const toolCall of toolCalls) {
            controller.enqueue({
              type: "tool-call",
              toolCallId: toolCall.toolCallId,
              toolName: toolCall.toolName,
              input: toolCall.input,
            });
          }

          // Finish the stream - AI SDK will handle execution and re-submission
          controller.enqueue({
            type: "finish",
            finishReason: "tool-calls" as LanguageModelV2FinishReason,
            usage: {
              inputTokens: session.inputUsage,
              outputTokens: undefined,
              totalTokens: undefined,
            },
          });

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return {
      stream,
      request: { body: { messages, options: promptOptions } },
    };
  }
}
