import { streamText } from "ai";
import { builtInAI, doesBrowserSupportBuiltInAI } from "@built-in-ai/core";

const warning = document.getElementById("warning");
const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const submitBtn = document.getElementById("submit-btn");
const outputContainer = document.getElementById("output-container");
const output = document.getElementById("output");

async function checkBrowserSupport() {
  const isSupported = doesBrowserSupportBuiltInAI();
  warning.hidden = isSupported;

  if (!isSupported) {
    submitBtn.disabled = true;
    return;
  }

  // Check if model is available
  const availability = await LanguageModel.availability();

  if (availability === "unavailable") {
    warning.hidden = false;
    warning.querySelector("strong").textContent = "Model unavailable.";
    warning.querySelector("p").textContent =
      "The built-in AI model is not available on this device.";
    submitBtn.disabled = true;
    return;
  }

  if (availability === "downloadable") {
    warning.hidden = false;
    warning.querySelector("strong").textContent = "Model downloading...";
    warning.querySelector("p").textContent =
      "The AI model is being downloaded. Check chrome://components for progress.";
    submitBtn.disabled = true;
    return;
  }

  submitBtn.disabled = false;
}

// Stream AI response
async function chat(userPrompt) {
  output.textContent = "";
  outputContainer.hidden = false;

  try {
    const result = streamText({
      model: builtInAI(),
      messages: [{ role: "user", content: userPrompt }],
    });

    for await (const chunk of result.textStream) {
      output.textContent += chunk;
    }
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
    console.error("Chat error:", error);
  }
}

// Handle form submission
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const userPrompt = promptInput.value.trim();
  if (!userPrompt) return;

  submitBtn.disabled = true;
  submitBtn.textContent = "Generating...";

  try {
    await chat(userPrompt);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Send";
  }
});

checkBrowserSupport();
