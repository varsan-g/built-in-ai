import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "esnext",
  },
  optimizeDeps: {
    include: ["@built-in-ai/core", "ai"],
  },
});
