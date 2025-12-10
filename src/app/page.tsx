"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { LiveWaveform } from "@/components/ui/live-waveform"
import { Conversation, ConversationContent, ConversationScrollButton } from "@/components/ui/conversation"
import { Message, MessageContent } from "@/components/ui/message"
import { Mic, MicOff, Volume2, VolumeX, Phone, PhoneOff, ChevronDown } from "lucide-react"
import { useTTS, type TTSVoice } from "@/hooks/use-tts"
import { useWebLLM } from "@/hooks/use-webllm"

type Status = "idle" | "loading" | "ready" | "listening" | "recording" | "transcribing" | "thinking" | "speaking" | "error"

/*
 * USING A DIFFERENT LLM:
 * 
 * This demo uses WebLLM (Qwen 1.5B) for fully local operation.
 * To use an external LLM instead, replace the webllm.chat() call
 * in handleLLMResponse() with a fetch to your API endpoint.
 * See README.md for example code.
 */

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export default function VoiceChat() {
  const [status, setStatus] = useState<Status>("idle")
  const [statusMessage, setStatusMessage] = useState("Click 'Initialize' to load models")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isCallActive, setIsCallActive] = useState(false)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const [showVoiceMenu, setShowVoiceMenu] = useState(false)
  const [textInput, setTextInput] = useState("")
  
  const workerRef = useRef<Worker | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const isCallActiveRef = useRef(false)
  const messagesRef = useRef<ChatMessage[]>([])
  const isProcessingRef = useRef(false)  // Lock to prevent parallel LLM/TTS calls
  const abortControllerRef = useRef<AbortController | null>(null)  // For cancelling LLM requests
  const pendingUserInputRef = useRef<string | null>(null)  // Queue user input during processing

  // WebGPU TTS
  const tts = useTTS({
    onStatusChange: (ttsStatus) => {
      if (ttsStatus === "speaking") {
        setStatus("speaking")
        setStatusMessage("Speaking...")
      }
    },
    onError: (error) => {
      console.error("TTS error:", error)
      setStatusMessage(`TTS error: ${error.message}`)
    }
  })

  // WebLLM (in-browser)
  const webllm = useWebLLM({
    onStatusChange: (llmStatus) => {
      if (llmStatus === "generating") {
        setStatus("thinking")
        setStatusMessage("Thinking...")
      }
    },
    onError: (error) => {
      console.error("WebLLM error:", error)
      setStatusMessage(`LLM error: ${error.message}`)
    }
  })

  // Keep refs in sync for use in callbacks
  const webllmRef = useRef(webllm)
  useEffect(() => {
    webllmRef.current = webllm
  }, [webllm])



  useEffect(() => {
    isCallActiveRef.current = isCallActive
  }, [isCallActive])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  // Initialize STT worker
  const initWorker = useCallback(() => {
    if (workerRef.current) return

    const worker = new Worker("/stt-worker-esm.js", { type: "module" })
    
    worker.onmessage = async (event) => {
      const { type, status: msgStatus, message, text, isFinal } = event.data

      switch (type) {
        case "status":
          if (msgStatus === "ready") {
            // STT ready, now load TTS
            setStatusMessage("Loading TTS model...")
            await tts.loadModels()
            
            // Load the in-browser LLM
            setStatusMessage("Loading LLM model (this may take a minute)...")
            await webllm.loadModel()
            
            setStatus("ready")
            setStatusMessage("Ready! Click 'Start Call' to begin.")
            console.log("[Voice] Ready - STT, TTS, LLM loaded")
          } else if (msgStatus === "loading") {
            setStatus("loading")
            setStatusMessage(message)
          } else if (msgStatus === "listening") {
            if (isCallActiveRef.current) {
              setStatus("listening")
              setStatusMessage("Listening...")
            }
          } else if (msgStatus === "recording") {
            setStatus("recording")
            setStatusMessage("Recording...")
          } else if (msgStatus === "transcribing") {
            setStatus("transcribing")
            setStatusMessage("Transcribing...")
          }
          break

        case "transcript":
          if (isFinal && text && text.trim()) {
            console.log("[STT]", text)
            
            // If we're currently processing, interrupt and queue the new input
            if (isProcessingRef.current) {
              console.debug("[Voice] Interrupting - new user input")
              // Cancel ongoing LLM request
              abortControllerRef.current?.abort()
              // Stop TTS playback
              tts.stop()
              // Queue this input
              pendingUserInputRef.current = text.trim()
              return
            }
            
            const userMessage: ChatMessage = { role: "user", content: text.trim() }
            setMessages(prev => [...prev, userMessage])
            handleLLMResponse([...messagesRef.current, userMessage])
          }
          break

        case "error":
          setStatus("error")
          setStatusMessage(`Error: ${message}`)
          break
      }
    }

    worker.onerror = (error) => {
      console.error("Worker error:", error)
      setStatus("error")
      setStatusMessage(`Worker error: ${error.message}`)
    }

    workerRef.current = worker
  }, [tts])

  // Load models
  const loadModels = useCallback(async () => {
    initWorker()
    setStatus("loading")
    setStatusMessage("Loading STT models...")
    workerRef.current?.postMessage({ type: "init" })
  }, [initWorker])

  const SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise and conversational - typically 1-3 sentences. Be warm and friendly. Use plain ASCII characters only - no emojis, no smart quotes, no fancy punctuation."

  // Handle LLM response with interruption support
  const handleLLMResponse = async (conversationHistory: ChatMessage[]) => {
    // Prevent parallel LLM/TTS calls
    if (isProcessingRef.current) {
      console.debug("[Voice] Ignoring request - already processing")
      return
    }
    isProcessingRef.current = true
    pendingUserInputRef.current = null

    // Create abort controller for API requests
    abortControllerRef.current = new AbortController()

    setStatus("thinking")
    setStatusMessage("Thinking...")

    try {
      const currentWebllm = webllmRef.current
      
      if (!currentWebllm.isReady) {
        throw new Error("LLM not ready")
      }
      
      // Use in-browser WebLLM
      console.debug("[Voice] Using WebLLM (in-browser)")
      const assistantMessage = await currentWebllm.chat(
        conversationHistory.map(m => ({ role: m.role, content: m.content })),
        SYSTEM_PROMPT
      )

      setMessages(prev => [...prev, { role: "assistant", content: assistantMessage }])
      console.log("[LLM]", assistantMessage)

      // Speak the response (can be interrupted)
      setStatus("speaking")
      setStatusMessage("Speaking...")
      await tts.speak(assistantMessage)

      if (isCallActiveRef.current) {
        setStatus("listening")
        setStatusMessage("Listening...")
      }

    } catch (error) {
      // Check if this was an intentional abort (user interrupted)
      if (error instanceof Error && error.name === "AbortError") {
        console.debug("[Voice] Request aborted by user interruption")
      } else {
        console.error("LLM error:", error)
        setStatusMessage(`LLM error: ${error}`)
      }
      if (isCallActiveRef.current) {
        setStatus("listening")
        setStatusMessage("Listening...")
      }
    } finally {
      isProcessingRef.current = false
      abortControllerRef.current = null
      
      // Process any pending user input that came in during processing
      if (pendingUserInputRef.current) {
        const pendingText = pendingUserInputRef.current
        pendingUserInputRef.current = null
        console.debug("[Voice] Processing pending input:", pendingText)
        const userMessage: ChatMessage = { role: "user", content: pendingText }
        setMessages(prev => [...prev, userMessage])
        // Use setTimeout to avoid call stack issues
        setTimeout(() => {
          handleLLMResponse([...messagesRef.current, userMessage])
        }, 0)
      }
    }
  }

  // Start microphone and VAD
  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream

      const audioContext = new AudioContext({ sampleRate: 16000 })
      audioContextRef.current = audioContext

      await audioContext.audioWorklet.addModule("/vad-processor.js")

      const workletNode = new AudioWorkletNode(audioContext, "vad-processor")
      workletNodeRef.current = workletNode

      const source = audioContext.createMediaStreamSource(stream)
      source.connect(workletNode)

      workletNode.port.onmessage = (event) => {
        const { buffer } = event.data
        workerRef.current?.postMessage({ type: "audio", buffer })
      }

      setStatus("listening")
      setStatusMessage("Listening...")
    } catch (error) {
      console.error("Microphone error:", error)
      setStatus("error")
      setStatusMessage(`Microphone error: ${error}`)
    }
  }

  // Stop microphone
  const stopListening = () => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    workerRef.current?.postMessage({ type: "stop" })
  }

  // Toggle mic mute
  const toggleMicMute = () => {
    if (streamRef.current) {
      const audioTrack = streamRef.current.getAudioTracks()[0]
      if (audioTrack) {
        audioTrack.enabled = isMicMuted
        setIsMicMuted(!isMicMuted)
      }
    }
  }

  // Start call
  const startCall = async () => {
    setIsCallActive(true)
    setMessages([])
    await startListening()
  }

  // End call
  const endCall = () => {
    // Abort any pending LLM request
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    isProcessingRef.current = false
    
    setIsCallActive(false)
    stopListening()
    tts.stop()
    setStatus("ready")
    setStatusMessage("Ready! Click mic to start a new call.")
  }

  // Auto-initialize on mount (empty deps - run once)
  useEffect(() => {
    console.debug("[Voice] Auto-initializing...")
    loadModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening()
      workerRef.current?.terminate()
    }
  }, [])

  // Waveform states
  const waveformActive = status === "listening" || status === "recording"
  const waveformProcessing = status === "speaking" || status === "thinking" || status === "transcribing"

  const voices: { id: TTSVoice; name: string; desc: string }[] = [
    { id: "F1", name: "Female 1", desc: "Calm, steady" },
    { id: "F2", name: "Female 2", desc: "Bright, cheerful" },
    { id: "F3", name: "Female 3", desc: "Professional" },
    { id: "F4", name: "Female 4", desc: "Confident" },
    { id: "F5", name: "Female 5", desc: "Gentle" },
    { id: "M1", name: "Male 1", desc: "Lively, upbeat" },
    { id: "M2", name: "Male 2", desc: "Deep, calm" },
    { id: "M3", name: "Male 3", desc: "Authoritative" },
    { id: "M4", name: "Male 4", desc: "Soft, friendly" },
    { id: "M5", name: "Male 5", desc: "Warm" },
  ]

  return (
    <div className="h-screen bg-zinc-950 flex flex-col">
      {/* Conversation area with auto-scroll */}
      <Conversation className="flex-1 pb-32">
        <ConversationContent className="max-w-2xl mx-auto">
          {messages.length === 0 ? (
            <div className="text-center py-20">
              <h1 className="text-2xl font-semibold text-white mb-2">AI Voice Chat</h1>
              <p className="text-zinc-400 text-sm mb-4">100% in-browser — nothing leaves your device</p>
              <p className="text-zinc-500">
                {status === "idle" 
                  ? "Click Initialize to load the voice models"
                  : status === "loading"
                  ? statusMessage
                  : isCallActive 
                  ? "Start speaking..." 
                  : "Click the phone to start a call"}
              </p>
              {status === "loading" && (tts.loadProgress > 0 || webllm.loadProgress > 0) && (
                <div className="mt-4 w-64 mx-auto space-y-2">
                  {tts.loadProgress > 0 && tts.loadProgress < 100 && (
                    <div>
                      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${tts.loadProgress}%` }}
                        />
                      </div>
                      <p className="text-xs text-zinc-600 mt-1">TTS: {Math.round(tts.loadProgress)}%</p>
                    </div>
                  )}
                  {webllm.loadProgress > 0 && (
                    <div>
                      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-purple-500 transition-all duration-300"
                          style={{ width: `${webllm.loadProgress}%` }}
                        />
                      </div>
                      <p className="text-xs text-zinc-600 mt-1">LLM: {Math.round(webllm.loadProgress)}%</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            messages.map((msg, i) => (
              <Message key={i} from={msg.role === "user" ? "user" : "assistant"}>
                <MessageContent variant="contained">
                  {msg.content}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {/* Fixed bottom bar */}
      <div className="fixed bottom-0 left-0 right-0 p-4">
        <div className="max-w-2xl mx-auto">
          <div className="bg-zinc-800/90 backdrop-blur-xl rounded-2xl border border-zinc-700/50 p-3 shadow-2xl">
            {/* Text input / Status area */}
            {isCallActive ? (
              <div className="text-zinc-500 text-sm mb-3 px-2">
                {status === "listening" ? "Listening..." : status === "recording" ? "Recording..." : status === "thinking" ? "Thinking..." : status === "speaking" ? "Speaking..." : "..."}
              </div>
            ) : (
              <form 
                onSubmit={(e) => {
                  e.preventDefault()
                  if (!textInput.trim() || status !== "ready") return
                  const userMessage: ChatMessage = { role: "user", content: textInput.trim() }
                  setMessages(prev => [...prev, userMessage])
                  handleLLMResponse([...messagesRef.current, userMessage])
                  setTextInput("")
                }}
                className="mb-3"
              >
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder={status === "idle" ? "Initialize to start..." : status === "loading" ? statusMessage : "How can I help?"}
                  disabled={status !== "ready"}
                  className="w-full bg-transparent text-zinc-200 text-sm px-2 py-1 outline-none placeholder:text-zinc-500 disabled:text-zinc-500"
                />
              </form>
            )}
            
            {/* Controls row */}
            <div className="flex items-center gap-2">
              {/* Waveform - takes remaining space */}
              <div className="flex-1 min-w-0 h-8">
                <LiveWaveform
                  active={waveformActive}
                  processing={waveformProcessing}
                  barWidth={2}
                  barGap={2}
                  barRadius={1}
                  fadeEdges={true}
                  fadeWidth={24}
                  sensitivity={2}
                  smoothingTimeConstant={0.8}
                  height={32}
                  mode="static"
                  className={waveformActive ? "text-green-400" : waveformProcessing ? "text-blue-400" : "text-zinc-600"}
                />
              </div>

              {/* Buttons - flex-shrink-0 to prevent shrinking */}
              <div className="flex items-center gap-1 flex-shrink-0">
                {/* Mic mute - black/white icon */}
                {status !== "loading" && status !== "idle" && (
                  <Button
                    onClick={toggleMicMute}
                    size="icon"
                    variant="ghost"
                    disabled={!isCallActive}
                    className={`h-10 w-10 rounded-full ${
                      !isCallActive
                        ? "text-zinc-600 cursor-not-allowed"
                        : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                    }`}
                    title={isMicMuted ? "Unmute mic" : "Mute mic"}
                  >
                    {isMicMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
                  </Button>
                )}

                {/* Speaker mute - black/white icon */}
                <Button
                  onClick={() => tts.setMuted(!tts.muted)}
                  size="icon"
                  variant="ghost"
                  className="h-10 w-10 rounded-full text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                  title={tts.muted ? "Unmute speaker" : "Mute speaker"}
                >
                  {tts.muted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
                </Button>

                {/* Voice selector */}
                <div className="relative">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowVoiceMenu(!showVoiceMenu)}
                    className="text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 gap-1"
                  >
                    <span className="text-xs">{tts.voice}</span>
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                  {showVoiceMenu && (
                    <>
                      {/* Click outside to close */}
                      <div className="fixed inset-0 z-10" onClick={() => setShowVoiceMenu(false)} />
                      <div className="absolute bottom-full mb-2 right-0 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl p-2 min-w-[140px] z-20">
                        {voices.map((voice) => (
                          <button
                            key={voice.id}
                            onClick={() => {
                              tts.setVoice(voice.id)
                              setShowVoiceMenu(false)
                            }}
                            className={`w-full text-left px-3 py-2 rounded text-sm hover:bg-zinc-700 ${
                              tts.voice === voice.id ? "bg-zinc-700 text-white" : "text-zinc-300"
                            }`}
                          >
                            <div className="font-medium">{voice.name}</div>
                            <div className="text-xs text-zinc-500">{voice.desc}</div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>



                {/* Call toggle - colored (green start / red end) - far right */}
                {status !== "loading" && status !== "idle" && (
                  <Button
                    onClick={isCallActive ? endCall : startCall}
                    size="icon"
                    variant="ghost"
                    className={`h-10 w-10 rounded-full ${
                      isCallActive
                        ? "bg-red-600 text-white hover:bg-red-700"
                        : "bg-green-600 text-white hover:bg-green-700"
                    }`}
                    title={isCallActive ? "End call" : "Start call"}
                  >
                    {isCallActive ? <PhoneOff className="h-5 w-5" /> : <Phone className="h-5 w-5" />}
                  </Button>
                )}

              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
