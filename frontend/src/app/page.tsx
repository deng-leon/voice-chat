"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { LiveWaveform } from "@/components/ui/live-waveform"
import { Conversation, ConversationContent, ConversationScrollButton } from "@/components/ui/conversation"
import { Message, MessageContent } from "@/components/ui/message"
import { Mic, MicOff, Volume2, VolumeX, Phone, PhoneOff, ChevronDown, Settings, X, ChevronUp } from "lucide-react"
import { useTTS, type TTSVoice } from "@/hooks/use-tts"
import { useWebLLM } from "@/hooks/use-webllm"

type Status = "idle" | "loading" | "ready" | "listening" | "recording" | "transcribing" | "thinking" | "speaking" | "error"
type LLMMode = "webllm" | "webllm-small"

const LANGUAGES = [
  { id: "en", name: "English" },
  { id: "es", name: "Spanish" },
  { id: "fr", name: "French" },
] as const

type LanguageCode = typeof LANGUAGES[number]["id"]

// Detect iOS/iPadOS
const isIOS = typeof navigator !== "undefined" && /iPad|iPhone|iPod/.test(navigator.userAgent)

// Model selection based on device
const DEFAULT_MODEL = isIOS ? "Qwen2.5-0.5B-Instruct-q4f16_1-MLC" : "Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
const SMALL_MODEL = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
const LARGE_MODEL = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC"

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
  const [showHistory, setShowHistory] = useState(true)
  const [systemPrompt, setSystemPromptState] = useState<string>("You are a helpful voice assistant.")
  const [isCallActive, setIsCallActive] = useState(false)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const isMicMutedRef = useRef(false)
  useEffect(() => { isMicMutedRef.current = isMicMuted }, [isMicMuted])

  const [showVoiceMenu, setShowVoiceMenu] = useState(false)
  const [showLanguageMenu, setShowLanguageMenu] = useState(false)
  const [language, setLanguage] = useState<LanguageCode>("en")
  const [textInput, setTextInput] = useState("")
  const [llmMode, setLLMMode] = useState<LLMMode>(isIOS ? "webllm-small" : "webllm")
  const [showDebugPanel, setShowDebugPanel] = useState(false)
  const [debugInfo, setDebugInfo] = useState({
    webgpu: "checking...",
    sttBackend: "unknown",
    llmMode: isIOS ? "webllm-small" : "webllm",
    vadLoaded: false,
    sttLoaded: false,
    ttsLoaded: false,
    llmLoaded: false,
  })

  // Fetch configuration from Python backend
  useEffect(() => {
    fetch("http://localhost:8000/api/config")
      .then(res => res.json())
      .then(data => {
        if (data.systemPrompt) {
          console.log("[Config] System Prompt updated from backend.")
          setSystemPromptState(data.systemPrompt)
        }
      })
      .catch(err => console.error("[Config] Failed to fetch backend config:", err))
  }, [])
  
  const workerRef = useRef<Worker | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const isCallActiveRef = useRef(false)
  const uniqueIdRef = useRef<string>("default")
  const messagesRef = useRef<ChatMessage[]>([])
  const isProcessingRef = useRef(false)  // Lock to prevent parallel LLM/TTS calls
  const abortControllerRef = useRef<AbortController | null>(null)  // For cancelling LLM requests
  const pendingUserInputRef = useRef<string | null>(null)  // Queue user input during processing
  const stallingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const isAITurnRef = useRef(false) // Blocks STT during AI response sequence
  const botReplyQueueRef = useRef<string | null>(null)
  const lastSeenFactualReplyRef = useRef<string | null>(null)

  // Ref to store latest handleLLMResponse to avoid closure issues in worker callbacks
  const handleLLMResponseRef = useRef<Function | null>(null)
  
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

  // Sync ref to latest LLM handler to avoid stale closures in worker
  useEffect(() => {
    handleLLMResponseRef.current = handleLLMResponse
  }, [messages, isCallActive, llmMode, language, tts])

  // Update language in worker
  useEffect(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: "setLanguage", language })
    }
  }, [language])

  // Handle incoming botReply from Zeebe
  const handleBotReply = useCallback(async (reply: string, fromQueue: boolean = false) => {
    if (!isCallActiveRef.current) return
    
    // De-duplicate: If the exact same reply is already handled (and not from our internal queue), ignore it
    const lastMsg = messagesRef.current[messagesRef.current.length - 1]?.content
    if (!fromQueue && (reply === lastSeenFactualReplyRef.current || reply === lastMsg)) {
      return
    }

    if (!fromQueue) {
      // Immediately mark as seen to avoid WebSocket deduplication races
      lastSeenFactualReplyRef.current = reply
      console.log(`[Zeebe-Worker] Received factual data: ${reply}`)
      
      // Clear the stalling loop (if any)
      if (stallingIntervalRef.current) {
        clearTimeout(stallingIntervalRef.current as any)
        stallingIntervalRef.current = null
      }

      // Abort any ongoing LLM stalling calculations (not speech)
      abortControllerRef.current?.abort()
      
      const normalizedReply = reply.replace(/^hello[!,.\s]*/i, "")
      
      // Add factual data to conversation history as assistant
      const botMessage: ChatMessage = { role: "assistant", content: normalizedReply }
      setMessages(prev => [...prev, botMessage])

      // If we're already processing (either calculating or currently speaking a stalling message), 
      // queue this factual answer and return. The 'finally' block in handleLLMResponse 
      // will pick it up as soon as the current speech finishes.
      if (isProcessingRef.current) {
        console.debug("[Voice] Still processing/speaking - queuing factual answer")
        botReplyQueueRef.current = normalizedReply
        return
      }
    } else {
      console.debug("[Voice] Executing queued factual answer")
      // Clear the queue item since we're now starting processing
      botReplyQueueRef.current = null
    }

    isProcessingRef.current = true
    setStatus("speaking")
    setStatusMessage("Providing final answer...")

    try {
      // Direct pass to TTS - no LLM rephrasing here
      if (tts.modelLoaded) {
        const normalizedReply = reply.replace(/^hello[!,.\s]*/i, "")
        await tts.speak(normalizedReply, language)
      }
    } catch (e) {
      console.error("[Voice] TTS Error in final reply:", e)
    } finally {
      isProcessingRef.current = false
      isAITurnRef.current = false // Finished whole sequence - allow user to speak again
      
      if (isCallActiveRef.current) {
        setStatus("listening")
        setStatusMessage("Listening...")
      }
    }
  }, [language, tts])

  // Relay transcription to Zeebe via Python backend
  const relayToZeebe = async (text: string) => {
    try {
      console.log(`[Zeebe] Relaying transcript: "${text.slice(0, 30)}..." with uniqueId: ${uniqueIdRef.current}`)
      const response = await fetch("http://localhost:8000/api/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          metadata: {
            source: "frontend-ui",
            uniqueId: uniqueIdRef.current
          }
        })
      })
      const data = await response.json()
      console.debug("[Zeebe] Response:", data)
    } catch (error) {
      console.error("[Zeebe] Failed to relay message:", error)
    }
  }

  // Initialize STT worker
  const initWorker = useCallback(() => {
    if (workerRef.current) return

    const worker = new Worker("/stt-worker-esm.js", { type: "module" })
    
    worker.onmessage = async (event) => {
      const { type, status: msgStatus, message, text, isFinal } = event.data

      switch (type) {
        case "status":
          if (msgStatus === "ready") {
            setDebugInfo(prev => ({ ...prev, vadLoaded: true, sttLoaded: true }))
            
            // STT ready, now load TTS
            setStatusMessage("Loading TTS model...")
            await tts.loadModels()
            setDebugInfo(prev => ({ ...prev, ttsLoaded: true }))
            
            // Load LLM - use small model on iOS/mobile
            const modelToLoad = llmMode === "webllm-small" ? SMALL_MODEL : LARGE_MODEL
            setStatusMessage(`Loading LLM (${llmMode === "webllm-small" ? "0.5B" : "1.5B"})...`)
            await webllm.loadModel(modelToLoad as Parameters<typeof webllm.loadModel>[0])
            setDebugInfo(prev => ({ ...prev, llmLoaded: true }))
            
            setStatus("ready")
            setStatusMessage("Ready! Click 'Start Call' to begin.")
            console.log("[Voice] Ready - STT, TTS, LLM loaded")
          } else if (msgStatus === "loading") {
            setStatus("loading")
            setStatusMessage(message)
          } else if (msgStatus === "listening") {
            if (isCallActiveRef.current) {
              if (isAITurnRef.current) {
                setStatus("thinking")
                setStatusMessage("Agent is working...")
              } else {
                setStatus("listening")
                setStatusMessage("Listening...")
              }
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
            // Block user input if the AI is currently in its turn (speaking or waiting for a final reply)
            if (isAITurnRef.current || isProcessingRef.current) {
              console.debug("[Voice] Blocking input - AI turn or processing active")
              return
            }

            console.log("[STT]", text)
            
            // Relay to Camunda 8
            relayToZeebe(text.trim())
            
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
            const newHistory = [...messagesRef.current, userMessage]
            setMessages(newHistory)
            if (handleLLMResponseRef.current) {
              handleLLMResponseRef.current(newHistory, true) // Force stalling acknowledge first
            }
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

  const getSystemPrompt = useCallback((langCode: LanguageCode) => {
    const langMap: Record<LanguageCode, string> = {
      en: "English", fr: "French", es: "Spanish"
    }
    const langName = langMap[langCode] || "English"
    return `${systemPrompt} Keep responses concise and conversational - typically 1-3 sentences. Be warm and friendly. CRITICAL: You must respond ONLY in ${langName}. Use plain text, no markdown, no emojis.`
  }, [systemPrompt])

  // Handle LLM response with interruption support
  // Handle LLM response with stalling support
  const handleLLMResponse = async (conversationHistory: ChatMessage[], isStalling: boolean = true) => {
    // Lock the turn to AI until the entire stalling/fetching sequence is complete
    isAITurnRef.current = true
    
    // Skip stalling if we already have a factual answer waiting to be spoken
    if (isStalling && botReplyQueueRef.current) {
      console.debug("[Voice] Skipping stall - factual answer already queued")
      return
    }

    // Check if we are already processing (to avoid overlapping LLM calls)
    if (isProcessingRef.current && isStalling) {
      console.debug("[Voice] Already processing - skip redundant stall request")
      return
    }

    // Clear any existing stall timer to avoid overlapping
    if (stallingIntervalRef.current) {
      clearTimeout(stallingIntervalRef.current as any)
      stallingIntervalRef.current = null
    }

    // Abort previous LLM call if we're now moving to final answer or if new user input came
    abortControllerRef.current?.abort()
    isProcessingRef.current = true
    pendingUserInputRef.current = null

    // Create abort controller for the new request
    const controller = new AbortController()
    abortControllerRef.current = controller

    setStatus("thinking")
    setStatusMessage("Agent is working...")

    try {
      const currentWebllm = webllmRef.current
      
      // Use in-browser WebLLM
      if (!currentWebllm.isReady) {
        throw new Error("LLM not ready")
      }
      
      const MAX_HISTORY = isIOS ? 4 : 10
      const recentHistory = conversationHistory.slice(-MAX_HISTORY)
      
      // Use conversationHistory instead of messagesRef.current to avoid staleness
      const lastUserMsg = conversationHistory.filter(m => m.role === "user").pop()?.content || ""
      
    const langMap: Record<LanguageCode, string> = {
      en: "English", fr: "French", es: "Spanish"
    }
    const langName = langMap[language] || "English"

    const LOCALIZED_STALLING = {
      en: {
        greetings: "Hello! I'm happy to help you with that.",
        confirmations: ["Got it.", "I see.", "Okay.", "Sure.", "Understood.", "I've got that.", "Acknowledged.", "Right.", "I understand.", "Perfect."],
        safePhrases: [
          "I'm just watching the screen for you now.",
          "Perfect. My colleague is just checking those records.",
          "I see. I'm just pulling up that information for you now.",
          "I'm looking into that right now.",
          "One moment, I'm just verifying those records.",
          "I'm keeping an eye on the progress bar for you.",
          "Just waiting for the database to return those results.",
          "Checking the latest updates on our end.",
          "The search is still in progress, just a second.",
          "I can see the system is working on your request.",
          "My screen is currently updating with the latest information.",
          "Almost there, just waiting for the final confirmation from the backend.",
          "Hold on, I'm just verifying those records with my colleague.",
          "Looking at the records now, shouldn't be much longer.",
          "The information is being retrieved as we speak.",
          "I'm just waiting for the system to refresh.",
          "Verifying the logs as we speak.",
          "Just a moment, the records are loading.",
          "I'm on it, just waiting for the results to appear.",
          "The search is continuing.",
          "I'm watching the screen update right now."
        ],
        forbidden: ["ready to proceed", "let me know", "ask me", "here to assist", "help you with", "need help", "welcome", "whenever you are", "ready to help", "details", "underway"]
      },
      fr: {
        greetings: "Bonjour ! Je suis ravi de vous aider avec cela.",
        confirmations: ["Entendu.", "Je vois.", "D'accord.", "Bien sûr.", "Compris.", "J'ai bien noté.", "C'est noté.", "Parfait.", "Je comprends.", "Très bien."],
        safePhrases: [
          "Je surveille l'écran pour vous en ce moment.",
          "Parfait. Mon collègue vérifie ces dossiers.",
          "Je vois. Je récupère ces informations pour vous maintenant.",
          "Je m'en occupe tout de suite.",
          "Un instant, je vérifie ces registres.",
          "Je garde un œil sur la barre de progression pour vous.",
          "J'attends que la base de données renvoie ces résultats.",
          "Je vérifie les dernières mises à jour de notre côté.",
          "La recherche est toujours en cours, un instant.",
          "Je vois que le système travaille sur votre demande.",
          "Mon écran se met à jour avec les dernières informations.",
          "On y est presque, j'attends la confirmation finale du système.",
          "Un instant, je vérifie ces dossiers avec mon collègue.",
          "Je regarde les dossiers, ça ne devrait plus tarder.",
          "Les informations sont en cours de récupération.",
          "J'attends simplement que le système se rafraîchisse.",
          "Je vérifie les journaux en ce moment même.",
          "Un instant, les dossiers sont en cours de chargement.",
          "Je suis dessus, j'attends l'affichage des résultats.",
          "La recherche se poursuit.",
          "Je regarde l'écran se mettre à jour actuellement."
        ],
        forbidden: ["prêt à continuer", "faites-moi savoir", "demandez-moi", "aider", "bienvenue", "quand vous serez", "prêt à vous aider", "détails", "en cours", "ready to proceed", "let me know", "ask me", "here to assist", "help you with", "need help", "welcome", "whenever you are", "ready to help", "details", "underway"]
      },
      es: {
        greetings: "¡Hola! Estoy encantado de ayudarte con eso.",
        confirmations: ["Entendido.", "Ya veo.", "Vale.", "Claro.", "Comprendido.", "Lo tengo.", "Anotado.", "Bien.", "Comprendo.", "Perfecto."],
        safePhrases: [
          "Estoy revisando la pantalla para usted ahora mismo.",
          "Perfecto. Mi colega está revisando esos registros.",
          "Ya veo. Estoy obteniendo esa información para usted ahora.",
          "Estoy trabajando en ello ahora mismo.",
          "Un momento, estoy verificando esos registros.",
          "Estoy atento al progreso para usted.",
          "Solo espero que la base de datos devuelva esos resultados.",
          "Revisando las últimas actualizaciones por nuestra parte.",
          "La búsqueda sigue en curso, un segundo.",
          "Veo que el sistema está trabajando en su solicitud.",
          "Mi pantalla se está actualizando con la información más reciente.",
          "Casi listo, solo espero la confirmación final del sistema.",
          "Espere un momento, estoy verificando esos registros con mi colega.",
          "Estoy mirando los registros ahora, no debería tardar mucho.",
          "La información se está recuperando mientras hablamos.",
          "Solo estoy esperando que el sistema se actualice.",
          "Verificando los registros en este momento.",
          "Un momento, los registros se están cargando.",
          "Estoy en ello, solo espero que aparezcan los resultados.",
          "La búsqueda continúa.",
          "Estoy viendo cómo se actualiza la pantalla ahora mismo."
        ],
        forbidden: ["listo para continuar", "hágame saber", "pregúnteme", "ayudar", "bienvenido", "cuando esté listo", "detalles", "en curso", "ready to proceed", "let me know", "ask me", "here to assist", "help you with", "need help", "welcome", "whenever you are", "ready to help", "details", "underway"]
      }
    }

    const currentStalling = LOCALIZED_STALLING[language] || LOCALIZED_STALLING.en

    const prompt = isStalling 
        ? `You are a first level support agent on their very first day. 
        Sice you are still learning, you always need a long time to check the records for the user's request.
           YOUR ROLE: Provide exactly one short, warm statement while they wait.
           STRICT CONSTRAINTS:
           1. NO QUESTIONS. No question marks. No taking initiative.
           2. NEVER ASK OR REQUEST DETAILS. Just state that the search is in progress.
           3. NO CALLS TO ACTION, NO REQUESTING DETAILS, NO OFFERS OF HELP OR ASSISTANCE.
           4. NO REPETITION. Do not repeat the user's name, their email, or any specific words from their last message.
           5. PROHIBITED WORDS: "details", "underway", "détails", "detalles". 
           6. Exactly ONE short sentence. Respond ONLY in ${langName}.`
        : getSystemPrompt(language)

      const chatMessages = isStalling 
        ? [
            { role: "user" as const, content: `As all customers I'm in a rush, can you tell me how long you will take you to handle my request? DO NOT repeat my previous request literally, NEVER UNDER NO CIRCUMSTANCE ASK ME TO PROVIDE DETAILS, DO NOT SAY OF COURSE and DO NOT use any names. RESPOND ONLY IN ${langName}.` }
          ]
        : recentHistory.map(m => ({ role: m.role, content: m.content }))

      console.debug(`[Voice] Using WebLLM, stalling: ${isStalling}`)
      let assistantMessage = await currentWebllm.chat(
        chatMessages,
        prompt,
        { signal: controller.signal }
      )

      // Post-processing: Strict question-mark pruning, CTA, and "assistance" checks for stalling turns
      let msgLower = assistantMessage.toLowerCase()
      const isRedundantStall = currentStalling.forbidden.some(word => msgLower.includes(word))
      
      if (isStalling && (assistantMessage.includes("?") || assistantMessage.length < 5 || isRedundantStall)) {
        console.warn("[Voice] LLM attempted a CTA/Offer/Detail-ask during stalling - forcing safe phrase")
        assistantMessage = currentStalling.safePhrases[Math.floor(Math.random() * currentStalling.safePhrases.length)]
        msgLower = assistantMessage.toLowerCase()
      }

      // Add pre-written parts
      if (isStalling) {
        const hasSpoken = conversationHistory.some(m => m.role === "assistant")
        const isInitialStall = conversationHistory[conversationHistory.length - 1]?.role === "user"

        if (!hasSpoken) {
          assistantMessage = `${currentStalling.greetings} ${assistantMessage}`
        } else if (isInitialStall) {
          const randomConf = currentStalling.confirmations[Math.floor(Math.random() * currentStalling.confirmations.length)]
          assistantMessage = `${randomConf} ${assistantMessage}`
        }
      }

      setMessages(prev => [...prev, { role: "assistant", content: assistantMessage }])
      console.log("[LLM]", assistantMessage)

      // Speak the response (if TTS is loaded)
      // Use modelLoaded to allow speaking even if status is not 'ready'
      if (tts.modelLoaded) {
        setStatus("speaking")
        setStatusMessage("Speaking...")
        await tts.speak(assistantMessage, language)
      } else {
        console.warn("[Voice] TTS not ready, skipping audio playback")
      }

      if (isCallActiveRef.current) {
        if (isAITurnRef.current) {
          setStatus("thinking")
          setStatusMessage("Agent is working...")
        } else {
          setStatus("listening")
          setStatusMessage("Listening...")
        }
      }

    } catch (error: any) {
      // Check if this was an intentional abort (user interrupted)
      if (error.name === "AbortError") {
        console.debug("[Voice] Request aborted by user interruption")
        return
      } else {
        console.error("LLM error:", error)
        setStatusMessage(`LLM error: ${error}`)
      }
      if (isCallActiveRef.current) {
        if (isAITurnRef.current) {
          setStatus("thinking")
          setStatusMessage("Agent is working...")
        } else {
          setStatus("listening")
          setStatusMessage("Listening...")
        }
      }
    } finally {
      // Only unlock processing state if this call owns the current controller
      if (abortControllerRef.current === controller) {
        isProcessingRef.current = false
        abortControllerRef.current = null
      }

      // Schedule next stall if we are still waiting and call is active
      // User requested more time between stalling responses - increased to 30s
      // We check botReplyQueueRef.current AGAIN here to ensure no stall is scheduled 
      // if an answer just arrived during the speak phase.
      if (isStalling && isCallActiveRef.current && !botReplyQueueRef.current) {
        console.debug("[Voice] Stall phrase finished - scheduling next in 30s")
        if (stallingIntervalRef.current) clearTimeout(stallingIntervalRef.current as any)
        stallingIntervalRef.current = setTimeout(() => {
          // Double check before executing scheduled stall
          if (handleLLMResponseRef.current && !botReplyQueueRef.current && isCallActiveRef.current) {
            handleLLMResponseRef.current(messagesRef.current, true)
          }
        }, 30000) as any
      }

      // Check if a factual Zeebe answer arrived while we were stalling
      if (botReplyQueueRef.current) {
        const reply = botReplyQueueRef.current
        // No need to clear botReplyQueueRef.current here as handleBotReply(reply, true) 
        // will now handle that internally
        handleBotReply(reply, true)
      }
      
      // Process any pending user input (only if we've finished the turns)
      if (pendingUserInputRef.current && abortControllerRef.current === null && !isAITurnRef.current) {
        const pendingText = pendingUserInputRef.current
        pendingUserInputRef.current = null
        console.debug("[Voice] Processing delayed input:", pendingText)
        const userMessage: ChatMessage = { role: "user", content: pendingText }
        const newHistory = [...messagesRef.current, userMessage]
        setMessages(newHistory)
        relayToZeebe(pendingText)
        setTimeout(() => handleLLMResponse(newHistory, true), 0)
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
    setIsMicMuted(prev => !prev)
  }

  // Ensure mic hardware is released (recording icon hidden) when AI is turn-taking or manually muted
  useEffect(() => {
    if (!isCallActive) return

    const shouldHearUser = status !== "thinking" && status !== "speaking" && !isMicMuted
    
    if (shouldHearUser && !streamRef.current) {
      console.debug(`[Voice] Re-starting listener (status: ${status}, muted: ${isMicMuted})`)
      startListening()
    } else if (!shouldHearUser && streamRef.current) {
      console.debug(`[Voice] Releasing hardware (status: ${status}, muted: ${isMicMuted})`)
      stopListening()
    }
  }, [status, isMicMuted, isCallActive])

  // Start call
  const startCall = async () => {
    // Close any open menus
    setShowLanguageMenu(false)
    setShowVoiceMenu(false)

    const newUniqueId = crypto.randomUUID()
    console.log(`[Voice] Starting call with uniqueId: ${newUniqueId}`)
    uniqueIdRef.current = newUniqueId
    
    // Clear refs for new call
    lastSeenFactualReplyRef.current = null
    botReplyQueueRef.current = null
    isProcessingRef.current = false
    isAITurnRef.current = false
    
    // Connect to WebSocket for remote context (e.g., botReply from Zeebe)
    if (wsRef.current) {
      console.debug("[WS] Closing existing session.")
      wsRef.current.close()
    }
    
    const backendHost = window.location.hostname || "localhost"
    const wsUrl = `ws://${backendHost}:8000/ws/${newUniqueId}`
    console.log(`[WS] Connecting to ${wsUrl}`)
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => console.log(`[WS] Connection established with uniqueId: ${newUniqueId}`)
    ws.onerror = (e) => console.error("[WS] Connection failed:", e)
    ws.onclose = (e) => console.log(`[WS] Connection closed: Code=${e.code}`)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === "botReply") {
          console.log("[Zeebe-Worker] Received final context:", data.content)
          handleBotReply(data.content)
        }
      } catch (e) {
        console.error("[WS] Error parsing data:", e)
      }
    }

    if (workerRef.current) {
      console.debug("[Voice] Sending setUniqueId to worker")
      workerRef.current.postMessage({ type: "setUniqueId", uniqueId: newUniqueId })
    }
    
    setIsCallActive(true)
    setMessages([])
  }

  // End call
  const endCall = () => {
    // Abort any pending LLM request
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    isProcessingRef.current = false
    
    setIsCallActive(false)
    uniqueIdRef.current = "default"
    
    stopListening()
    tts.stop()
    setStatus("ready")
    setStatusMessage("Ready! Click mic to start a new call.")
  }

  // Check WebGPU support and auto-initialize on mount
  useEffect(() => {
    // Check WebGPU
    const checkWebGPU = async () => {
      if (typeof navigator !== "undefined" && "gpu" in navigator) {
        try {
          const adapter = await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter()
          setDebugInfo(prev => ({ ...prev, webgpu: adapter ? "available" : "no adapter" }))
        } catch {
          setDebugInfo(prev => ({ ...prev, webgpu: "error" }))
        }
      } else {
        setDebugInfo(prev => ({ ...prev, webgpu: "not supported" }))
      }
    }
    checkWebGPU()
    
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
  const waveformActive = isCallActive && !isMicMuted && (status === "listening" || status === "recording")
  const waveformProcessing = isCallActive && (status === "speaking" || status === "thinking" || status === "transcribing")

  const voices: { id: TTSVoice; name: string; desc: string }[] = [
    { id: "F1", name: "Female 1", desc: "Gentle" },
    { id: "F2", name: "Female 2", desc: "Bright, cheerful" },
    { id: "F3", name: "Female 3", desc: "Professional" },
    { id: "F4", name: "Female 4", desc: "Confident" },
    { id: "F5", name: "Female 5", desc: "Calm, steady" },
    { id: "M1", name: "Male 1", desc: "Lively, upbeat" },
    { id: "M2", name: "Male 2", desc: "Deep, calm" },
    { id: "M3", name: "Male 3", desc: "Authoritative" },
    { id: "M4", name: "Male 4", desc: "Soft, friendly" },
    { id: "M5", name: "Male 5", desc: "Warm" },
  ]

  return (
    <div className="h-screen bg-zinc-950 flex flex-col overflow-hidden">
      {/* Persistent Header */}
      <div className="pt-20 pb-10 text-center z-10 transition-all duration-700 ease-in-out" 
           style={{ transform: showHistory ? 'translateY(0)' : 'translateY(10vh)' }}>
        <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">AI Voice Chat</h1>
        {!isCallActive && messages.length === 0 && (
          <div className="space-y-4">
            <p className="text-zinc-400 text-lg animate-pulse">
              {status === "idle" 
                ? "Click Initialize to load the voice models"
                : status === "loading"
                ? statusMessage
                : "Click the phone to start a call"}
            </p>
            {status === "loading" && (tts.loadProgress > 0 || webllm.loadProgress > 0) && (
              <div className="mt-4 w-64 mx-auto space-y-3">
                {tts.loadProgress > 0 && tts.loadProgress < 100 && (
                  <div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-orange-500 transition-all duration-300 shadow-[0_0_10px_rgba(249,115,22,0.5)]"
                        style={{ width: `${tts.loadProgress}%` }}
                      />
                    </div>
                    <p className="text-[10px] uppercase tracking-wider text-zinc-500 mt-2 font-medium">Voices: {Math.round(tts.loadProgress)}%</p>
                  </div>
                )}
                {webllm.loadProgress > 0 && webllm.loadProgress < 100 && (
                  <div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500 transition-all duration-300 shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                        style={{ width: `${webllm.loadProgress}%` }}
                      />
                    </div>
                    <p className="text-[10px] uppercase tracking-wider text-zinc-500 mt-2 font-medium">Brain: {Math.round(webllm.loadProgress)}%</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Conversation area with auto-scroll */}
      <div className={`transition-all duration-700 ease-in-out overflow-hidden ${showHistory ? 'flex-1 opacity-100' : 'flex-none h-0 opacity-0 pointer-events-none'}`}>
        <Conversation className="h-full pb-32">
          <ConversationContent className="max-w-2xl mx-auto">
            {messages.length > 0 && (
              messages
                .filter(msg => !msg.content.startsWith("[CONTEXT]")) // Hide technical session updates
                .map((msg, i) => (
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
      </div>

      {/* Debug Panel */}
      {showDebugPanel && (
        <div className="fixed top-4 right-4 bg-zinc-900 border border-zinc-700 rounded-lg p-4 text-xs font-mono z-50 min-w-[200px]">
          <div className="flex justify-between items-center mb-2">
            <span className="text-zinc-400 font-semibold">Debug Info</span>
            <button onClick={() => setShowDebugPanel(false)} className="text-zinc-500 hover:text-white">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="space-y-1 text-zinc-300">
            <div>WebGPU: <span className={debugInfo.webgpu === "available" ? "text-green-400" : "text-yellow-400"}>{debugInfo.webgpu}</span></div>
            <div>iOS: <span className={isIOS ? "text-yellow-400" : "text-green-400"}>{isIOS ? "yes" : "no"}</span></div>
            <div>LLM Mode: <span className="text-blue-400">{llmMode}</span></div>
            <hr className="border-zinc-700 my-2" />
            <div>VAD: {debugInfo.vadLoaded ? <span className="text-green-400">✓</span> : <span className="text-zinc-500">○</span>}</div>
            <div>STT: {debugInfo.sttLoaded ? <span className="text-green-400">✓</span> : <span className="text-zinc-500">○</span>}</div>
            <div>TTS: {debugInfo.ttsLoaded ? <span className="text-green-400">✓</span> : <span className="text-zinc-500">○</span>}</div>
            <div>LLM: {debugInfo.llmLoaded ? <span className="text-green-400">✓</span> : <span className="text-zinc-500">○</span>}</div>
            <hr className="border-zinc-700 my-2" />
            <div className="flex gap-2">
              <button
                onClick={() => setLLMMode("webllm")}
                className={`px-2 py-1 rounded ${llmMode === "webllm" ? "bg-blue-600" : "bg-zinc-700"}`}
              >
                WebLLM
              </button>
              <button
                onClick={() => setLLMMode("webllm-small")}
                className={`px-2 py-1 rounded ${llmMode === "webllm-small" ? "bg-blue-600" : "bg-zinc-700"}`}
              >
                Small (0.5B)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Debug toggle button */}
      <button
        onClick={() => setShowDebugPanel(!showDebugPanel)}
        className="fixed top-4 right-4 p-2 bg-zinc-800 rounded-full text-zinc-400 hover:text-white z-40"
        title="Toggle debug panel"
      >
        <Settings className="h-4 w-4" />
      </button>

      {/* Fixed bottom bar */}
      <div className={`fixed left-0 right-0 p-4 transition-all duration-700 ease-in-out z-30 ${showHistory ? 'bottom-0' : 'top-1/2 -translate-y-1/2'}`}>
        <div className="max-w-2xl mx-auto relative">
          
          {/* Collapse/Expand Toggle - absolute positioned above the bar */}
          <div className="absolute -top-10 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-zinc-800/80 border border-zinc-700 flex items-center justify-center cursor-pointer hover:bg-zinc-700 transition-colors shadow-lg"
               onClick={() => setShowHistory(!showHistory)}
               title={showHistory ? "Collapse conversation" : "Expand conversation"}>
            {showHistory ? (
              <ChevronDown className="h-5 w-5 text-zinc-400" />
            ) : (
              <ChevronUp className="h-5 w-5 text-zinc-400" />
            )}
          </div>

          <div className="bg-zinc-800/90 backdrop-blur-xl rounded-2xl border border-zinc-700/50 p-3 shadow-2xl">
            {/* Text input / Status area */}
            {isCallActive ? (
              <div className="text-zinc-500 text-sm mb-3 px-2 flex items-center gap-2">
                {isMicMuted && (status === "listening" || status === "recording") ? (
                  <span className="text-red-500/80 font-medium">Muted</span>
                ) : (
                  <span>{status === "listening" ? "Listening..." : status === "recording" ? "Recording..." : status === "thinking" ? "Thinking..." : status === "speaking" ? "Speaking..." : "..."}</span>
                )}
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
                  className={waveformActive ? "text-orange-500" : waveformProcessing ? "text-orange-500" : "text-zinc-600"}
                />
              </div>

              {/* Buttons - flex-shrink-0 to prevent shrinking */}
              <div className="flex items-center gap-1 flex-shrink-0">
                
                {/* Language selector */}
                <div className="relative">
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={isCallActive}
                    onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                    className={`gap-1 ${
                      isCallActive
                        ? "text-zinc-600 cursor-not-allowed"
                        : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                    }`}
                    title={isCallActive ? "Can't change STT language during call" : "Change STT language"}
                  >
                    <span className="text-xs">{LANGUAGES.find(l => l.id === language)?.name}</span>
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                  {showLanguageMenu && !isCallActive && (
                    <>
                      {/* Click outside to close */}
                      <div className="fixed inset-0 z-10" onClick={() => setShowLanguageMenu(false)} />
                      <div className="absolute bottom-full mb-2 right-0 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl p-2 min-w-[120px] z-20">
                        {LANGUAGES.map((lang) => (
                          <button
                            key={lang.id}
                            onClick={() => {
                              setLanguage(lang.id)
                              setShowLanguageMenu(false)
                            }}
                            className={`w-full text-left px-3 py-2 rounded text-sm hover:bg-zinc-700 ${
                              language === lang.id ? "bg-zinc-700 text-white" : "text-zinc-300"
                            }`}
                          >
                            <div className="font-medium">{lang.name}</div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>

                {/* Mic mute - black/white icon */}
                {status !== "loading" && status !== "idle" && (
                  <Button
                    onClick={toggleMicMute}
                    size="icon"
                    variant="ghost"
                    disabled={!isCallActive || status === "thinking" || status === "speaking"}
                    className={`h-10 w-10 rounded-full ${
                      !isCallActive || status === "thinking" || status === "speaking"
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
                    disabled={isCallActive}
                    onClick={() => setShowVoiceMenu(!showVoiceMenu)}
                    className={`gap-1 ${
                      isCallActive
                        ? "text-zinc-600 cursor-not-allowed"
                        : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                    }`}
                    title={isCallActive ? "Can't change voice during call" : "Change persona"}
                  >
                    <span className="text-xs">{tts.voice}</span>
                    <ChevronDown className="h-3 w-3" />
                  </Button>
                  {showVoiceMenu && !isCallActive && (
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
