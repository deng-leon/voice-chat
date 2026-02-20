/**
 * Hook for browser-based TTS using Supertonic WebGPU
 */

import { useCallback, useEffect, useRef, useState } from "react"
import { loadPipeline, loadEmbeddings, streamTTS } from "@/lib/tts"
import type { TextToAudioPipeline } from "@huggingface/transformers"

export type TTSStatus = "idle" | "loading" | "ready" | "speaking" | "error"
export type TTSVoice = "F1" | "F2" | "F3" | "F4" | "F5" | "M1" | "M2" | "M3" | "M4" | "M5"

interface UseTTSOptions {
  onStatusChange?: (status: TTSStatus) => void
  onError?: (error: Error) => void
}

export function useTTS(options: UseTTSOptions = {}) {
  const { onStatusChange, onError } = options

  const [status, setStatus] = useState<TTSStatus>("idle")
  const [loadProgress, setLoadProgress] = useState(0)
  const [voice, setVoice] = useState<TTSVoice>("F1")

  const ttsRef = useRef<TextToAudioPipeline | null>(null)
  const embeddingsRef = useRef<Record<string, Float32Array> | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null)
  const [muted, setMutedState] = useState(false)
  const voiceRef = useRef<TTSVoice>("F1")
  const abortControllerRef = useRef<AbortController | null>(null)

  // Update status and notify
  const updateStatus = useCallback((newStatus: TTSStatus) => {
    setStatus(newStatus)
    onStatusChange?.(newStatus)
  }, [onStatusChange])

  // Load TTS models
  const loadModels = useCallback(async () => {
    if (ttsRef.current) return // Already loaded

    updateStatus("loading")

    try {
      const progressMap = new Map<string, number>()
      const onProgress = (info: any) => {
        if (info.status === "progress" && info.file?.endsWith(".onnx_data")) {
          progressMap.set(info.file, info.loaded / info.total)
          const total = Array.from(progressMap.values()).reduce((a, b) => a + b, 0)
          setLoadProgress((total / 3) * 100)
        }
      }

      const [pipeline, embeddings] = await Promise.all([
        loadPipeline(onProgress),
        loadEmbeddings(),
      ])

      ttsRef.current = pipeline
      embeddingsRef.current = embeddings

      updateStatus("ready")
    } catch (error) {
      console.error("TTS load error:", error)
      updateStatus("error")
      onError?.(error as Error)
    }
  }, [updateStatus, onError])

  // Normalize text for TTS - replace fancy unicode with plain ASCII
  const normalizeText = (text: string, lang?: string): string => {
    let normalized = text
      // Smart quotes to straight quotes
      .replace(/[\u2018\u2019]/g, "'")  // ' ' -> '
      .replace(/[\u201C\u201D]/g, '"')  // " " -> "
      // Dashes
      .replace(/[\u2013\u2014]/g, "-")  // – — -> -
      // Ellipsis
      .replace(/\u2026/g, "...")
      // Remove emojis (they cause issues)
      .replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, "")
      // Clean up extra spaces
      .replace(/\s+/g, " ")
      .trim()

    // For Chinese, remove spaces between characters as Whisper/LLM might add them
    if (lang === 'zh') {
      normalized = normalized.replace(/([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])/g, '$1$2');
    }
    
    return normalized
  }

  // Speak text
  const speak = useCallback(async (text: string, lang?: string): Promise<void> => {
    if (!ttsRef.current || !embeddingsRef.current) {
      throw new Error("TTS not loaded")
    }

    const normalizedText = normalizeText(text, lang)
    if (normalizedText !== text) {
      console.debug("[TTS] Text normalized:", { original: text, normalized: normalizedText })
    }
    updateStatus("speaking")

    // Create abort controller for this speak request
    abortControllerRef.current = new AbortController()
    const signal = abortControllerRef.current.signal

    try {
      // Create audio context if needed
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext()
      }
      const ctx = audioContextRef.current
      if (ctx.state === "suspended") {
        await ctx.resume()
      }

      const currentVoice = voiceRef.current
      const speakerEmbedding = embeddingsRef.current[currentVoice]
      console.debug("[TTS] Using voice:", currentVoice, "embedding length:", speakerEmbedding?.length)
      const quality = 20 // Number of inference steps (higher = better quality, slower)
      const speed = 1.1 // Slightly faster speech

      // Collect all audio chunks
      const audioChunks: Float32Array[] = []
      let sampleRate = 44100

      console.debug("[TTS] Starting streamTTS...")
      for await (const result of streamTTS(normalizedText, ttsRef.current, speakerEmbedding, quality, speed, lang, signal)) {
        if (signal.aborted) {
          console.debug("[TTS] streamTTS aborted")
          break
        }
        console.debug("[TTS] Got chunk", result.index, "/", result.total)
        audioChunks.push(result.audio.audio)
        sampleRate = result.audio.sampling_rate
      }

      if (signal.aborted) {
        updateStatus("ready")
        return
      }

      console.debug("[TTS] streamTTS complete, chunks:", audioChunks.length)

      // Merge chunks
      const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0)
      const mergedAudio = new Float32Array(totalLength)
      let offset = 0
      for (const chunk of audioChunks) {
        mergedAudio.set(chunk, offset)
        offset += chunk.length
      }

      // Play audio with gain node for volume control
      const audioBuffer = ctx.createBuffer(1, mergedAudio.length, sampleRate)
      audioBuffer.getChannelData(0).set(mergedAudio)

      // Create gain node if needed
      if (!gainNodeRef.current) {
        gainNodeRef.current = ctx.createGain()
        gainNodeRef.current.connect(ctx.destination)
      }

      const source = ctx.createBufferSource()
      source.buffer = audioBuffer
      source.connect(gainNodeRef.current)
      sourceNodeRef.current = source

      await new Promise<void>((resolve) => {
        source.onended = () => {
          sourceNodeRef.current = null
          resolve()
        }
        source.start()
      })

      updateStatus("ready")
    } catch (error) {
      console.error("TTS speak error:", error)
      // Don't set error state - recover to ready so we can try again
      updateStatus("ready")
      onError?.(error as Error)
      // Don't throw - let the caller continue
    }
  }, [voice, updateStatus, onError])

  // Stop playback
  const stop = useCallback(() => {
    // Abort pending inference
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }

    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop()
      } catch (e) {
        // Source might already be stopped
      }
      sourceNodeRef.current = null
    }
    updateStatus("ready")
  }, [updateStatus])

  // Set muted state - controls gain node
  const setMuted = useCallback((muted: boolean) => {
    setMutedState(muted)
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = muted ? 0 : 1
    }
  }, [])

  // Keep voice ref in sync
  useEffect(() => {
    voiceRef.current = voice
  }, [voice])

  // Cleanup
  useEffect(() => {
    return () => {
      audioContextRef.current?.close()
    }
  }, [])

  return {
    status,
    loadProgress,
    voice,
    setVoice,
    loadModels,
    speak,
    stop,
    muted,
    setMuted,
    isReady: status === "ready",
    isLoading: status === "loading",
    isSpeaking: status === "speaking",
    modelLoaded: !!ttsRef.current && !!embeddingsRef.current,
  }
}
