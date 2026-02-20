/**
 * STT Worker - ES Module version
 * Handles VAD + Whisper transcription
 */

// Suppress noisy ONNX/hub warnings in worker
const originalLog = console.log
const originalWarn = console.warn
const originalError = console.error
const originalInfo = console.info
const originalDebug = console.debug

const suppress = (...args) => {
  const msg = args.map(arg => {
    try { return String(arg); } catch(e) { return ""; }
  }).join(' ')
  
  // Strip ANSI color codes and normalize
  const cleanMsg = msg.replace(/\\x1B\\[[0-9;]*[mK]/g, '').toLowerCase()
  
  return (
    cleanMsg.includes('onnxruntime') || 
    cleanMsg.includes('verifyeachnodeisassignedtoanep') ||
    cleanMsg.includes('session_state.cc') ||
    cleanMsg.includes('execution providers') ||
    cleanMsg.includes('unknown model class') ||
    cleanMsg.includes('content-length')
  )
}

const interceptor = (original) => (...args) => {
  if (suppress(...args)) return
  original.apply(console, args)
}

console.log = interceptor(originalLog)
console.warn = interceptor(originalWarn)
console.error = interceptor(originalError)
console.info = interceptor(originalInfo)
console.debug = interceptor(originalDebug)

import { AutoModel, pipeline, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm"

// ============ Constants ============
const INPUT_SAMPLE_RATE = 16000
const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000
const SPEECH_THRESHOLD = 0.5
const EXIT_THRESHOLD = 0.15
const MIN_SILENCE_DURATION_MS = 1000
const MIN_SILENCE_DURATION_SAMPLES = MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE_MS
const SPEECH_PAD_MS = 80
const SPEECH_PAD_SAMPLES = SPEECH_PAD_MS * INPUT_SAMPLE_RATE_MS
const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE_MS
const MAX_BUFFER_DURATION = 30
const MAX_NUM_PREV_BUFFERS = Math.ceil(SPEECH_PAD_SAMPLES / 512)

// ============ State ============
let sileroVad = null
let transcriber = null
let currentLanguage = "en"
let uniqueId = "default"

const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE)
let bufferPointer = 0

let vadSr = null
let vadState = null
let isRecording = false
let postSpeechSamples = 0
const prevBuffers = []
let isProcessing = false  // Lock to prevent concurrent processing
const audioQueue = []     // Queue for audio chunks while processing

// Configure - use Python backend proxy for models (port 8000)
env.useBrowserCache = true
env.allowLocalModels = false
env.remoteHost = "http://localhost:8000/hf-proxy/"
env.remotePathTemplate = "{model}/resolve/{revision}/"

// Suppress ONNX runtime warnings (3=error, 4=fatal)
if (env.backends?.onnx) {
  env.backends.onnx.logSeverityLevel = 3
  env.backends.onnx.logVerbosityLevel = 0
}

// Detect WebGPU support and platform
async function getDevice() {
  // Check if iOS - use WASM there as WebGPU is unstable
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || 
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  
  if (isIOS) {
    console.debug("[STT Worker] iOS detected, using WASM for stability")
    return "wasm"
  }
  
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter()
      if (adapter) {
        console.debug("[STT Worker] WebGPU available")
        return "webgpu"
      }
    } catch (e) {
      console.debug("[STT Worker] WebGPU check failed:", e)
    }
  }
  console.debug("[STT Worker] Falling back to WASM")
  return "wasm"
}

let selectedDevice = null

console.debug("[STT Worker ESM] Loaded, AutoModel:", !!AutoModel)

// ============ Model Loading ============
async function loadModels() {
  console.debug("[STT Worker] Starting model load...")
  
  // Detect best available device
  selectedDevice = await getDevice()
  console.debug("[STT Worker] Using device:", selectedDevice)
  
  self.postMessage({ type: "status", status: "loading", message: `Loading VAD model (${selectedDevice})...` })

  // Load Silero VAD from onnx-community (public, no auth required)
  console.debug("[STT Worker] Loading Silero VAD...")
  sileroVad = await AutoModel.from_pretrained("onnx-community/silero-vad", {
    config: { model_type: "custom" },
    dtype: "fp32",
    device: selectedDevice,
    progress_callback: (progress) => {
      if (progress.progress !== undefined) {
        self.postMessage({ type: "progress", progress: progress.progress, message: `VAD: ${progress.status}` })
      }
    },
  })
  console.debug("[STT Worker] VAD loaded!")

  // Init VAD tensors
  vadSr = new Tensor("int64", [INPUT_SAMPLE_RATE], [])
  vadState = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128])

  self.postMessage({ type: "status", status: "loading", message: "Loading Whisper model..." })

  // Load Whisper from onnx-community (public, no auth required)
  console.debug("[STT Worker] Loading Whisper base...")
  // TODO: Add whisper-tiny-en to R2 for mobile
  const whisperModel = "onnx-community/whisper-base"
  console.debug("[STT Worker] Using Whisper model:", whisperModel)
  
  try {
    transcriber = await pipeline("automatic-speech-recognition", whisperModel, {
      dtype: "fp32",
      device: selectedDevice,
      progress_callback: (progress) => {
        if (progress.progress !== undefined) {
          self.postMessage({ type: "progress", progress: progress.progress, message: `Whisper: ${progress.status}` })
        }
      },
    })
    console.debug("[STT Worker] Whisper loaded!")
  } catch (e) {
    console.error("[STT Worker] Whisper load failed:", e)
    self.postMessage({ type: "error", message: `Whisper failed: ${e.message}` })
    return
  }

  // Warm up
  try {
    console.debug("[STT Worker] Warming up Whisper...")
    await transcriber(new Float32Array(INPUT_SAMPLE_RATE))
    console.debug("[STT Worker] Warmup complete!")
  } catch (e) {
    console.error("[STT Worker] Warmup failed:", e)
    self.postMessage({ type: "error", message: `Warmup failed: ${e.message}` })
    return
  }

  console.log("[STT Worker] Ready!")
  self.postMessage({ type: "status", status: "ready", message: "Models loaded!" })
}

// ============ VAD ============
async function vad(buffer) {
  if (!sileroVad || !buffer || !vadSr) return false

  const input = new Tensor("float32", buffer, [1, buffer.length])
  const { stateN, output } = await sileroVad({ input, sr: vadSr, state: vadState })
  vadState = stateN

  const isSpeech = output.data[0]
  return isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD)
}

// ============ Transcription ============
async function transcribe(buffer) {
  if (!transcriber) return ""

  self.postMessage({ type: "status", status: "transcribing", message: "Transcribing..." })

  // Multilingual support - specify language if provided
  const result = await transcriber(buffer, {
    language: currentLanguage,
    task: "transcribe"
  })

  return result.text.trim()
}

// ============ Buffer Management ============
function resetAfterRecording(offset = 0) {
  BUFFER.fill(0, offset)
  bufferPointer = offset
  isRecording = false
  postSpeechSamples = 0
  prevBuffers.length = 0
}

async function dispatchForTranscription(overflow) {
  const overflowLength = overflow?.length ?? 0

  const buffer = BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES)
  const prevLength = prevBuffers.reduce((acc, b) => acc + b.length, 0)
  const paddedBuffer = new Float32Array(prevLength + buffer.length)
  
  let offset = 0
  for (const prev of prevBuffers) {
    paddedBuffer.set(prev, offset)
    offset += prev.length
  }
  paddedBuffer.set(buffer, offset)

  const text = await transcribe(paddedBuffer)

  if (text && !["", "[BLANK_AUDIO]"].includes(text)) {
    self.postMessage({ type: "transcript", text, isFinal: true })
  }

  if (overflow) {
    BUFFER.set(overflow, 0)
  }
  resetAfterRecording(overflowLength)

  self.postMessage({ type: "status", status: "listening", message: "Listening..." })
}

// Queue audio and process sequentially
function queueAudio(buffer) {
  audioQueue.push(buffer)
  if (!isProcessing) {
    processQueue()
  }
}

async function processQueue() {
  if (isProcessing || audioQueue.length === 0) return
  isProcessing = true
  
  try {
    while (audioQueue.length > 0) {
      const buffer = audioQueue.shift()
      await processAudioChunk(buffer)
    }
  } finally {
    isProcessing = false
  }
}

async function processAudioChunk(buffer) {
  const wasRecording = isRecording
  const isSpeech = await vad(buffer)

  if (!wasRecording && !isSpeech) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) {
      prevBuffers.shift()
    }
    prevBuffers.push(buffer)
    return
  }

  const remaining = BUFFER.length - bufferPointer
  if (buffer.length >= remaining) {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer)
    bufferPointer += remaining
    await dispatchForTranscription(buffer.subarray(remaining))
    return
  }

  BUFFER.set(buffer, bufferPointer)
  bufferPointer += buffer.length

  if (isSpeech) {
    if (!isRecording) {
      self.postMessage({ type: "status", status: "recording", message: "Recording..." })
    }
    isRecording = true
    postSpeechSamples = 0
    return
  }

  postSpeechSamples += buffer.length

  if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
    return
  }

  if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
    resetAfterRecording()
    self.postMessage({ type: "status", status: "listening", message: "Listening..." })
    return
  }

  await dispatchForTranscription()
}

// ============ Message Handler ============
self.onmessage = async (event) => {
  const { type, buffer, language, uniqueId: newUniqueId } = event.data

  switch (type) {
    case "init":
      try {
        await loadModels()
      } catch (err) {
        console.error("[STT Worker] Init error:", err)
        self.postMessage({ type: "error", message: err.toString() })
      }
      break

    case "setLanguage":
      if (language) {
        currentLanguage = language
        console.log(`[STT Worker] Language set to: ${language}`)
      }
      break

    case "setUniqueId":
      if (newUniqueId) {
        uniqueId = newUniqueId
        console.log(`[STT Worker] UniqueId set to: ${newUniqueId}`)
      }
      break

    case "audio":
      if (sileroVad && vadState) {
        queueAudio(buffer)
      }
      break

    case "stop":
      if (bufferPointer > MIN_SPEECH_DURATION_SAMPLES) {
        await dispatchForTranscription()
      } else {
        resetAfterRecording()
      }
      break
  }
}
