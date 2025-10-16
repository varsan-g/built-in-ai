/// <reference types="@types/dom-chromium-ai" />

import { LanguageModelV2FunctionTool, LanguageModelV2ToolCall } from "@ai-sdk/provider";

/**
 * Polyfill for the Prompt API tool calling functionality.
 *
 * This polyfill enables tool use with the Prompt API by implementing tool-calling 
 * logic manually. This is necessary because native browser implementations don't 
 * yet support the `tools` parameter defined in the spec.
 *
 * Spec: https://webmachinelearning.github.io/prompt-api/
 *
 * NOTE: This is a workaround implementation. Native implementations will handle 
 * tool calls internally without needing JSON parsing or manual tool execution.
 */

/**
 * A JSON structure that the model is instructed to use when it decides to call a tool.
 * This is a polyfill-specific format, not part of the official spec.
 */
export interface ToolCallRequest {
  tool_calls: Array<{
    id?: string;
    name: string;
    input: Record<string, any>;
  }>;
}

/**
 * Result of parsing a tool call response
 */
export interface ParsedToolCall {
  toolCalls: LanguageModelV2ToolCall[];
  rawResponse: string;
}

/**
 * Helper function to detect if a response contains a tool call request.
 * Handles both plain JSON and markdown-wrapped JSON.
 */
export function parseToolCallRequest(response: string): ParsedToolCall | null {
  let jsonToParse = response.trim();

  // Check if the response is wrapped in a markdown code block and extract the JSON.
  // Handles both: ```json\n{...}\n``` and ```{...}```
  const markdownMatch = response.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (markdownMatch && markdownMatch[1]) {
    jsonToParse = markdownMatch[1].trim();
  }

  // Remove trailing underscore and other non-JSON characters (model hallucination)
  jsonToParse = jsonToParse.replace(/[_\s]+$/, '');

  // Common model errors to fix
  const cleanupPatterns = [
    (s: string) => s, // Try original first
    (s: string) => s.replace(/\}\s*\}+\s*$/, '}'), // Remove trailing extra braces
    (s: string) => s.replace(/\]\s*,\s*\{/g, ', {'), // Fix: [obj1], {obj2} -> [obj1, {obj2}
  ];

  for (const cleanup of cleanupPatterns) {
    try {
      const cleaned = cleanup(jsonToParse);
      const parsedResponse = JSON.parse(cleaned);

      if (
        parsedResponse.tool_calls &&
        Array.isArray(parsedResponse.tool_calls) &&
        parsedResponse.tool_calls.length > 0
      ) {
        const toolCallRequest = parsedResponse as ToolCallRequest;

        // Convert to Vercel AI SDK format
        const toolCalls: LanguageModelV2ToolCall[] = toolCallRequest.tool_calls.map(
          (call, index) => ({
            type: 'tool-call' as const,
            toolCallId: call.id || `call_${index}_${Date.now()}`,
            toolName: call.name,
            input: JSON.stringify(call.input), // input must be stringified JSON
          })
        );

        return {
          toolCalls,
          rawResponse: response,
        };
      }
    } catch (e) {
      // Try next cleanup pattern
      continue;
    }
  }

  return null;
}

/**
 * Helper function to build the system prompt that instructs the model how to use tools.
 */
export function buildToolSystemPrompt(
  tools: Array<LanguageModelV2FunctionTool & { execute?: (args: any) => Promise<any> }>,
  systemMessage?: string
): string {
  const toolDescriptions = tools
    .map((tool) => {
      // Extract properties from the JSON schema for clearer parameter documentation
      const schema = tool.inputSchema as any;
      const properties = schema?.properties || {};
      const required = schema?.required || [];

      const params = Object.entries(properties).map(([key, value]: [string, any]) => {
        const isRequired = required.includes(key);
        const description = value.description || '';
        return `    - ${key}${isRequired ? ' (required)' : ''}: ${value.type}${description ? ` - ${description}` : ''}`;
      }).join('\n');

      return `- ${tool.name}: ${tool.description || "No description"}\n${params || '    (no parameters)'}`;
    })
    .join("\n\n");

  const baseSystemPrompt = systemMessage || "You are a helpful assistant.";

  return `${baseSystemPrompt}

You are a helpful assistant. You have access to the following tools.
To use a tool, respond with a JSON object with a 'tool_calls' key, like this:
{"tool_calls": [{"name": "tool_name", "input": {"arg1": "value1", "arg2": "value2"}}]}

Available tools:
${toolDescriptions}

When you need to use a tool, respond ONLY with the JSON tool call request. After receiving the tool results, provide your final answer to the user.`
}

/**
 * Execute tool calls and return formatted results for Vercel AI SDK.
 */
export async function executeToolCalls(
  toolCalls: LanguageModelV2ToolCall[],
  tools: Array<LanguageModelV2FunctionTool & { execute?: (args: any) => Promise<any> }>
): Promise<
  Array<{
    toolCallId: string;
    toolName: string;
    output: any;
    error?: string;
  }>
> {
  const toolResults = [];

  for (const call of toolCalls) {
    const tool = tools.find((t) => t.name === call.toolName);

    if (!tool) {
      toolResults.push({
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        output: null,
        error: `Tool '${call.toolName}' not found.`,
      });
      continue;
    }

    try {
      // Parse the input (it's a stringified JSON)
      const input = JSON.parse(call.input);

      // Log the tool structure to debug
      console.log('[Tool Structure]:', {
        name: tool.name,
        hasExecute: !!tool.execute,
        toolKeys: Object.keys(tool),
        tool: tool
      });

      // Execute the tool with the provided input
      if (tool.execute) {
        const output = await tool.execute(input);
        toolResults.push({
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          output,
        });
      } else {
        toolResults.push({
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          output: null,
          error: `Tool '${call.toolName}' does not have an execute function.`,
        });
      }
    } catch (error) {
      toolResults.push({
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        output: null,
        error: `Error executing tool '${call.toolName}': ${error instanceof Error ? error.message : String(error)
          }`,
      });
    }
  }

  return toolResults;
}

/**
 * Format tool results as a message string for the model
 */
export function formatToolResults(
  toolResults: Array<{
    toolCallId: string;
    toolName: string;
    output: any;
    error?: string;
  }>
): string {
  const formattedResults = toolResults.map((r) => {
    if (r.error) {
      return `Tool: ${r.toolName} (ID: ${r.toolCallId})\nError: ${r.error}`;
    }
    const outputStr =
      typeof r.output === "string" ? r.output : JSON.stringify(r.output);
    return `Tool: ${r.toolName} (ID: ${r.toolCallId})\nResult: ${outputStr}`;
  });

  return `Tool results:\n${formattedResults.join("\n\n")}`;
}

