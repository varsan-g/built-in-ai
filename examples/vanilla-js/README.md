# Built-in AI - Vanilla JS Example

A minimal example demonstrating `@built-in-ai/core` with plain JavaScript and Vite.

## Browser Setup

### Chrome

1. Use Chrome version **138 or higher**
2. Navigate to `chrome://flags/`
3. Enable **`#optimization-guide-on-device-model`**
4. Enable **`#prompt-api-for-gemini-nano-multimodal-input`**
5. Click **Relaunch**
6. Go to `chrome://components/` and verify **Optimization Guide On Device Model** shows "Up-to-date" (the model download may take a few minutes)

### Edge

1. Use Edge Dev/Canary version **138.0.3309.2 or higher**
2. Navigate to `edge://flags/#prompt-api-for-phi-mini`
3. Set it to **Enabled**
4. Restart Edge
5. Go to `edge://on-device-internals` and verify **Device performance class** is "High" or greater

## Running the Example

```bash
# From the monorepo root
npm install

# Navigate to this example
cd examples/vanilla-js

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

## Learn More

- [Built-in AI Documentation](https://built-in-ai.dev/docs/ai-sdk-v6)
- [Chrome Prompt API Guide](https://developer.chrome.com/docs/ai/prompt-api)
- [AI SDK Documentation](https://ai-sdk.dev/docs)
