import {
  LanguageModelV2Prompt,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";

export interface ConvertedMessages {
  systemMessage?: string;
  messages: LanguageModelMessage[];
}

/**
 * Convert base64 string to Uint8Array for built-in AI compatibility
 * Built-in AI supports BufferSource (including Uint8Array) for image/audio data
 */
function convertBase64ToUint8Array(base64: string): Uint8Array {
  try {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  } catch (error) {
    throw new Error(`Failed to convert base64 to Uint8Array: ${error}`);
  }
}

/**
 * Convert file data to the appropriate format for built-in AI
 * Built-in AI supports: Blob, BufferSource (Uint8Array), URLs
 */
function convertFileData(data: any, mediaType: string): Uint8Array | string {
  // Handle different data types from Vercel AI SDK
  if (data instanceof URL) {
    // URLs - keep as string (if supported by provider)
    return data.toString();
  }

  if (data instanceof Uint8Array) {
    // Already in correct format
    return data;
  }

  if (typeof data === "string") {
    // Base64 string from AI SDK - convert to Uint8Array
    return convertBase64ToUint8Array(data);
  }

  // Fallback for other types (shouldn't happen with current AI SDK)
  console.warn(`Unexpected data type for ${mediaType}:`, typeof data);
  return data;
}

/**
 * Convert Vercel AI SDK prompt format to built-in AI Prompt API format
 * Returns system message (for initialPrompts) and regular messages (for prompt method)
 */
export function convertToBuiltInAIMessages(
  prompt: LanguageModelV2Prompt,
): ConvertedMessages {
  let systemMessage: string | undefined;
  const messages: LanguageModelMessage[] = [];

  for (const message of prompt) {
    switch (message.role) {
      case "system": {
        // There's only ever one system message from AI SDK
        systemMessage = message.content;
        break;
      }

      case "user": {
        messages.push({
          role: "user",
          content: message.content.map((part) => {
            switch (part.type) {
              case "text": {
                return {
                  type: "text",
                  value: part.text,
                } as LanguageModelMessageContent;
              }

              case "file": {
                const { mediaType, data, filename } = part;

                if (mediaType?.startsWith("image/")) {
                  const convertedData = convertFileData(data, mediaType);

                  return {
                    type: "image",
                    value: convertedData,
                  } as LanguageModelMessageContent;
                } else if (mediaType?.startsWith("audio/")) {
                  const convertedData = convertFileData(data, mediaType);

                  return {
                    type: "audio",
                    value: convertedData,
                  } as LanguageModelMessageContent;
                } else {
                  throw new UnsupportedFunctionalityError({
                    functionality: `file type: ${mediaType}`,
                  });
                }
              }

              default: {
                throw new UnsupportedFunctionalityError({
                  functionality: `content type: ${(part as any).type}`,
                });
              }
            }
          }),
        } as LanguageModelMessage);
        break;
      }

      case "assistant": {
        let text = "";
        const toolCalls: any[] = [];

        for (const part of message.content) {
          switch (part.type) {
            case "text": {
              text += part.text;
              break;
            }
            case "tool-call": {
              // For beta tool calling support, we convert tool calls to JSON format
              toolCalls.push({
                id: part.toolCallId,
                name: part.toolName,
                input: typeof part.input === 'string' ? JSON.parse(part.input) : part.input,
              });
              break;
            }
          }
        }

        // If there are tool calls, format them as JSON (polyfill approach)
        if (toolCalls.length > 0) {
          const toolCallJson = JSON.stringify({ tool_calls: toolCalls });
          text = toolCallJson;
        }

        messages.push({
          role: "assistant",
          content: text,
        } as LanguageModelMessage);
        break;
      }

      case "tool": {
        // Convert tool results to a format the model can understand
        const toolResults = message.content.map((part) => {
          if (part.type === "tool-result") {
            // Handle both old format (result) and new format (output)
            const output = (part as any).output || (part as any).result;
            const outputStr =
              typeof output === "string"
                ? output
                : JSON.stringify(output);
            return `Tool: ${part.toolName} (ID: ${part.toolCallId})\nResult: ${outputStr}`;
          }
          return "";
        });

        messages.push({
          role: "user",
          content: `Tool results:\n${toolResults.join("\n\n")}`,
        } as LanguageModelMessage);
        break;
      }

      default: {
        throw new Error(`Unsupported role: ${(message as any).role}`);
      }
    }
  }

  return { systemMessage, messages };
}
