import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY || "";
const ai = new GoogleGenAI({ apiKey });

// Models in priority order — skips to next on 404, retries on 503/429
const ANALYSIS_MODELS = [
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
  "gemini-2.5-flash",
];

const IMAGE_GEN_MODELS = [
  "gemini-2.0-flash-exp-image-generation",
];

function isOverloaded(error: any): boolean {
  const msg = (error?.message || "").toLowerCase();
  return (
    msg.includes("resource_exhausted") ||
    msg.includes("quota") ||
    msg.includes("unavailable") ||
    msg.includes("overload") ||
    msg.includes("high demand") ||
    error?.code === 429 ||
    error?.code === 503
  );
}

function isNotFound(error: any): boolean {
  return (
    (error?.message || "").toLowerCase().includes("not found") ||
    error?.code === 404
  );
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Try each model once (no aggressive retries that burn quota)
async function tryModels<T>(
  models: string[],
  fn: (model: string) => Promise<T>
): Promise<T> {
  let lastError: any;
  for (const model of models) {
    try {
      return await fn(model);
    } catch (err: any) {
      lastError = err;
      if (isNotFound(err)) {
        console.warn(`Model "${model}" not available, trying next...`);
        continue; // skip immediately to next model
      }
      if (isOverloaded(err)) {
        console.warn(`Model "${model}" busy, waiting 8s then trying next...`);
        await sleep(8000);
        continue; // try next model after wait
      }
      throw err; // any other error — surface immediately
    }
  }
  throw lastError;
}

export interface AnalysisResult {
  isReal: boolean;
  confidence: number;
  source: string;
  reasoning: string;
  metadata: {
    resolution?: string;
    format?: string;
    artifacts?: string[];
  };
  heatmaps?: Record<string, string>;
}

export type HeatmapMode = 'artifacts' | 'noise' | 'lighting';

export async function generateHeatmap(base64Image: string, mode: HeatmapMode = 'artifacts'): Promise<string> {
  const prompts: Record<HeatmapMode, string> = {
    artifacts: "Analyze this image for deepfake artifacts. Generate a heatmap overlay highlighting areas that look AI-generated or manipulated. Use a red-to-yellow thermal gradient for manipulated regions. Return only the heatmap image.",
    noise: "Perform a noise level analysis on this image. Highlight inconsistent pixel noise patterns in bright cyan and purple, like an Error Level Analysis map. Return only the resulting image.",
    lighting: "Analyze lighting and shadow inconsistencies in this image. Use a blue-to-white gradient to highlight light direction anomalies common in deepfakes. Return only the resulting image."
  };

  return tryModels(IMAGE_GEN_MODELS, async (model) => {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          { inlineData: { data: base64Image.split(',')[1], mimeType: "image/jpeg" } },
          { text: prompts[mode] }
        ]
      },
      config: { responseModalities: ["IMAGE", "TEXT"] }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    throw new Error(`No image returned for ${mode} heatmap.`);
  });
}

export async function analyzeImage(base64Image: string): Promise<AnalysisResult> {
  const prompt = `Analyze this image for authenticity.
Determine if it is a real photograph or AI-generated/deepfake.
Respond ONLY with raw JSON (no markdown, no code fences):
{
  "isReal": boolean,
  "confidence": number between 0 and 1,
  "source": "e.g. Phone Camera, DSLR, Stable Diffusion, Midjourney, DALL-E",
  "reasoning": "detailed explanation of artifacts, lighting, or clues",
  "metadata": {
    "artifacts": ["list of detected anomalies"]
  }
}`;

  const raw = await tryModels(ANALYSIS_MODELS, async (model) => {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          { text: prompt },
          { inlineData: { data: base64Image.split(',')[1], mimeType: "image/jpeg" } }
        ]
      },
      config: { responseMimeType: "application/json" }
    });
    return response.text || "";
  });

  // Strip accidental markdown code fences if present
  const cleaned = raw
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch (e) {
    console.error("Failed to parse response:", raw);
    throw new Error("Received invalid data from API. Please try again.");
  }
}
