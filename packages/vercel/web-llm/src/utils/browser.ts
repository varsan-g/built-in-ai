declare global {
  interface Navigator {
    gpu?: GPU;
  }
}

export function isMobile(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  );
}

export function checkWebGPU(): boolean {
  try {
    return !!globalThis?.navigator?.gpu;
  } catch {
    return false;
  }
}

/**
 * Check if the browser supports WebLLM (WebGPU)
 */
export function doesBrowserSupportWebLLM(): boolean {
  return checkWebGPU();
}
