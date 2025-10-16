# Tool Calling with Built-in AI (Beta)

This guide shows you how to use tool calling with the browser's built-in AI models.

## Installation

```bash
npm install @built-in-ai/core@beta
```

## Overview

Tool calling allows the AI model to use functions/tools to gather information before responding to the user. Since the browser's built-in AI doesn't natively support tool calling yet, we use a **polyfill** that implements tool calling through structured JSON responses.

### How It Works

1. Model receives a special system prompt with tool descriptions
2. When the model needs a tool, it responds with JSON: `{"tool_calls": [...]}`
3. The polyfill parses the JSON and emits tool-call events
4. Your client-side code executes the tools using `onToolCall`
5. Results are sent back to the model via `addToolResult`
6. Model generates a final natural language response

## Basic Setup

### 1. Define Your Tools

In your client-side transport or component, define tools **without** `execute` functions (client-side execution):

```typescript
import { tool } from "ai";
import { z } from "zod";

const tools = {
  getWeather: tool({
    description: "Get the weather in a location (fahrenheit)",
    inputSchema: z.object({
      location: z.string().describe("The location to get the weather for"),
    }),
  }),
  getCurrentTime: tool({
    description: "Get the current time",
    inputSchema: z.object({}), // No parameters
  }),
};
```

### 2. Pass Tools to streamText

```typescript
import { streamText } from "ai";
import { builtInAI } from "@built-in-ai/core";

const result = streamText({
  model: builtInAI(),
  messages: prompt,
  tools,
});
```

### 3. Handle Tool Execution in useChat

```typescript
import { useChat } from "@ai-sdk/react";
import { lastAssistantMessageIsCompleteWithToolCalls } from "ai";

const { messages, addToolResult } = useChat({
  transport: new ClientSideChatTransport(),
  
  // Automatically re-submit when all tool results are available
  sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
  
  // Execute tools client-side
  async onToolCall({ toolCall }) {
    // Check if dynamic tool for type narrowing
    if (toolCall.dynamic) {
      return;
    }

    if (toolCall.toolName === 'getWeather') {
      const input = toolCall.input as { location: string };
      const temperature = Math.round(Math.random() * (90 - 32) + 32);
      
      addToolResult({
        tool: 'getWeather',
        toolCallId: toolCall.toolCallId,
        output: {
          location: input.location,
          temperature,
          conditions: 'Partly cloudy',
        },
      });
    }

    if (toolCall.toolName === 'getCurrentTime') {
      addToolResult({
        tool: 'getCurrentTime',
        toolCallId: toolCall.toolCallId,
        output: {
          time: new Date().toLocaleTimeString(),
          date: new Date().toLocaleDateString(),
        },
      });
    }
  },
});
```

## Displaying Tool Calls in UI

Tool calls appear as typed parts in messages. Access them using `message.parts`:

```tsx
{messages.map((message) => (
  <div key={message.id}>
    {message.parts.map((part, index) => {
      switch (part.type) {
        case 'text':
          return <p key={index}>{part.text}</p>;

        case 'tool-getWeather':
        case 'tool-getCurrentTime':
          return (
            <Tool key={index}>
              <ToolHeader type={part.type} state={part.state} />
              <ToolContent>
                {part.input !== undefined && (
                  <ToolInput input={part.input} />
                )}
                {(part.output || part.errorText) && (
                  <ToolOutput
                    output={part.output ? JSON.stringify(part.output, null, 2) : undefined}
                    errorText={part.errorText}
                  />
                )}
              </ToolContent>
            </Tool>
          );

        default:
          return null;
      }
    })}
  </div>
))}
```

## Tool States

Each tool part has a `state` property:

- **`input-streaming`**: Tool input is being generated (streaming)
- **`input-available`**: Tool call is ready, waiting for execution
- **`output-available`**: Tool has been executed successfully
- **`output-error`**: Tool execution failed

```tsx
switch (part.state) {
  case 'input-streaming':
    return <div>Loading tool call...</div>;
  case 'input-available':
    return <div>Executing {part.type}...</div>;
  case 'output-available':
    return <div>Result: {JSON.stringify(part.output)}</div>;
  case 'output-error':
    return <div>Error: {part.errorText}</div>;
}
```

## Error Handling

Handle errors during tool execution:

```typescript
async onToolCall({ toolCall }) {
  if (toolCall.dynamic) return;

  if (toolCall.toolName === 'getWeather') {
    try {
      const weather = await fetchWeather(toolCall.input);
      
      addToolResult({
        tool: 'getWeather',
        toolCallId: toolCall.toolCallId,
        output: weather,
      });
    } catch (err) {
      addToolResult({
        tool: 'getWeather',
        toolCallId: toolCall.toolCallId,
        state: 'output-error',
        errorText: 'Unable to fetch weather information',
      });
    }
  }
}
```

## Complete Example

```typescript
import { useChat } from "@ai-sdk/react";
import { ClientSideChatTransport } from "./client-side-chat-transport";
import { lastAssistantMessageIsCompleteWithToolCalls } from "ai";

export default function Chat() {
  const { messages, sendMessage, addToolResult } = useChat({
    transport: new ClientSideChatTransport(),
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
    
    async onToolCall({ toolCall }) {
      if (toolCall.dynamic) return;

      if (toolCall.toolName === 'getWeather') {
        const input = toolCall.input as { location: string };
        addToolResult({
          tool: 'getWeather',
          toolCallId: toolCall.toolCallId,
          output: {
            location: input.location,
            temperature: 72,
            conditions: 'Sunny',
          },
        });
      }

      if (toolCall.toolName === 'getCurrentTime') {
        addToolResult({
          tool: 'getCurrentTime',
          toolCallId: toolCall.toolCallId,
          output: {
            time: new Date().toLocaleTimeString(),
            date: new Date().toLocaleDateString(),
          },
        });
      }
    },
  });

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>
          <strong>{m.role}:</strong>
          {m.parts.map((part, i) => {
            switch (part.type) {
              case 'text':
                return <p key={i}>{part.text}</p>;
              case 'tool-getWeather':
              case 'tool-getCurrentTime':
                return (
                  <div key={i}>
                    Tool: {part.type} - {part.state}
                    {part.output && <pre>{JSON.stringify(part.output, null, 2)}</pre>}
                  </div>
                );
              default:
                return null;
            }
          })}
        </div>
      ))}

      <form onSubmit={(e) => {
        e.preventDefault();
        sendMessage({ text: input });
      }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} />
      </form>
    </div>
  );
}
```

## Important Notes

### Beta Limitations

- **Polyfill Implementation**: Tool calling is implemented via JSON parsing, not native browser support
- **Model Reliability**: The built-in model may occasionally produce malformed JSON (the polyfill handles most cases)
- **Client-Side Only**: Tools must be executed client-side; server-side auto-execution is not supported
- **No Streaming Tool Inputs**: Tool inputs are buffered before being emitted

### Best Practices

1. **Always check `toolCall.dynamic`** in your `onToolCall` handler for proper TypeScript type narrowing
2. **Don't use `await`** when calling `addToolResult` to avoid potential deadlocks
3. **Use `sendAutomaticallyWhen`** with `lastAssistantMessageIsCompleteWithToolCalls` for automatic re-submission
4. **Handle errors gracefully** by using `state: 'output-error'` when tools fail
5. **Keep tools simple** - complex multi-step tools may confuse the model

## Future

When the browser's Prompt API adds native tool calling support, this polyfill will be replaced with the native implementation, and your code will continue to work without changes.

## Troubleshooting

**Tools not executing:**
- Ensure `onToolCall` is defined in `useChat`
- Check that `addToolResult` is being called
- Verify `sendAutomaticallyWhen` is set for automatic re-submission

**Malformed JSON errors:**
- The polyfill handles most cases automatically
- Check console for parsing errors
- The model may need clearer tool descriptions

**Type errors:**
- Always check `if (toolCall.dynamic) return;` before accessing `toolCall.toolName`
- Cast `toolCall.input` to your expected type: `const input = toolCall.input as { location: string }`

