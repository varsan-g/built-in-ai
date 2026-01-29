import {
  WebLLMLanguageModel,
  WebLLMModelId,
  WebLLMSettings,
} from "./web-llm-language-model";
import {
  WebLLMEmbeddingModel,
  WebLLMEmbeddingModelId,
  WebLLMEmbeddingSettings,
} from "./web-llm-embedding-model";

/**
 * Create a new WebLLMLanguageModel.
 * @param modelId The model ID to use (e.g., 'Llama-3.1-8B-Instruct-q4f32_1-MLC')
 * @param settings Options for the model
 */
export function webLLM(
  modelId: WebLLMModelId,
  settings?: WebLLMSettings,
): WebLLMLanguageModel {
  return new WebLLMLanguageModel(modelId, settings);
}

/**
 * Create a new WebLLMEmbeddingModel.
 * @param modelId The embedding model ID to use (e.g., 'snowflake-arctic-embed-m-q0f32-MLC-b32')
 * @param settings Options for the embedding model
 */
export function webLLMEmbedding(
  modelId: WebLLMEmbeddingModelId,
  settings?: WebLLMEmbeddingSettings,
): WebLLMEmbeddingModel {
  return new WebLLMEmbeddingModel(modelId, settings);
}
