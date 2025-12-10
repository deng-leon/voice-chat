"use client"

import { useState, useRef, useCallback } from "react"
// @ts-ignore - types will be available after npm install
import * as webllm from "@mlc-ai/web-llm"

export type WebLLMStatus = "idle" | "loading" | "ready" | "generating" | "error"

// Small models suitable for browser
export const WEBLLM_MODELS = [
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen 1.5B", size: "~1GB" },
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", name: "Qwen 0.5B", size: "~400MB" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", name: "Llama 1B", size: "~700MB" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", name: "Llama 3B", size: "~2GB" },
  { id: "SmolLM2-1.7B-Instruct-q4f16_1-MLC", name: "SmolLM 1.7B", size: "~1GB" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", name: "Gemma 2B", size: "~1.5GB" },
] as const

export type WebLLMModel = typeof WEBLLM_MODELS[number]["id"]

interface UseWebLLMOptions {
  onStatusChange?: (status: WebLLMStatus) => void
  onError?: (error: Error) => void
}

interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
}

export function useWebLLM(options: UseWebLLMOptions = {}) {
  const { onStatusChange, onError } = options

  const [status, setStatus] = useState<WebLLMStatus>("idle")
  const [loadProgress, setLoadProgress] = useState(0)
  const [currentModel, setCurrentModel] = useState<WebLLMModel | null>(null)
  
  const engineRef = useRef<webllm.MLCEngine | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const updateStatus = useCallback((newStatus: WebLLMStatus) => {
    console.log("[WebLLM] Status:", newStatus)
    setStatus(newStatus)
    onStatusChange?.(newStatus)
  }, [onStatusChange])

  // Load model
  const loadModel = useCallback(async (modelId: WebLLMModel = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC") => {
    if (status === "loading") return
    
    updateStatus("loading")
    setLoadProgress(0)

    try {
      const engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (progress: { progress?: number; text?: string }) => {
          // progress.progress is 0-1
          const pct = Math.round((progress.progress || 0) * 100)
          setLoadProgress(pct)
          console.log(`[WebLLM] Loading: ${progress.text} (${pct}%)`)
        },
      })

      engineRef.current = engine
      setCurrentModel(modelId)
      updateStatus("ready")
      console.log(`[WebLLM] Model ${modelId} loaded successfully`)
    } catch (error) {
      console.error("[WebLLM] Load error:", error)
      updateStatus("error")
      onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }, [status, updateStatus, onError])

  // Generate chat completion
  const chat = useCallback(async (
    messages: ChatMessage[],
    systemPrompt?: string
  ): Promise<string> => {
    if (!engineRef.current) {
      throw new Error("WebLLM not loaded")
    }

    updateStatus("generating")
    abortControllerRef.current = new AbortController()

    try {
      // Prepend system prompt if provided
      const allMessages = systemPrompt
        ? [{ role: "system" as const, content: systemPrompt }, ...messages]
        : messages

      const response = await engineRef.current.chat.completions.create({
        messages: allMessages,
        max_tokens: 256,
        temperature: 0.7,
      })

      const content = response.choices[0]?.message?.content || ""
      updateStatus("ready")
      return content
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        console.log("[WebLLM] Generation aborted")
        updateStatus("ready")
        return ""
      }
      console.error("[WebLLM] Chat error:", error)
      updateStatus("error")
      throw error
    } finally {
      abortControllerRef.current = null
    }
  }, [updateStatus])

  // Stream chat completion
  const chatStream = useCallback(async function* (
    messages: ChatMessage[],
    systemPrompt?: string
  ): AsyncGenerator<string, void, unknown> {
    if (!engineRef.current) {
      throw new Error("WebLLM not loaded")
    }

    updateStatus("generating")

    try {
      const allMessages = systemPrompt
        ? [{ role: "system" as const, content: systemPrompt }, ...messages]
        : messages

      const chunks = await engineRef.current.chat.completions.create({
        messages: allMessages,
        max_tokens: 256,
        temperature: 0.7,
        stream: true,
      })

      for await (const chunk of chunks) {
        const delta = chunk.choices[0]?.delta?.content
        if (delta) {
          yield delta
        }
      }

      updateStatus("ready")
    } catch (error) {
      console.error("[WebLLM] Stream error:", error)
      updateStatus("error")
      throw error
    }
  }, [updateStatus])

  // Abort current generation
  const abort = useCallback(() => {
    abortControllerRef.current?.abort()
    // WebLLM doesn't have native abort, but we can track it
  }, [])

  // Unload model
  const unload = useCallback(async () => {
    if (engineRef.current) {
      await engineRef.current.unload()
      engineRef.current = null
      setCurrentModel(null)
      updateStatus("idle")
    }
  }, [updateStatus])

  return {
    status,
    loadProgress,
    currentModel,
    isReady: status === "ready",
    isLoading: status === "loading",
    isGenerating: status === "generating",
    loadModel,
    chat,
    chatStream,
    abort,
    unload,
    models: WEBLLM_MODELS,
  }
}
