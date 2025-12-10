# AI Voice Chat - 100% In-Browser

A hands-free AI voice assistant that runs entirely in your browser. Speech recognition, LLM, and text-to-speech all run locally using WebGPU - no API keys, no server, no data leaves your device. Just talk naturally and the AI responds.

## Live Demo

Try it now: [HuggingFace Space](https://huggingface.co/spaces/RickRossTN/ai-voice-chat)

## What Makes This Different

**Everything runs in your browser:**
- **Speech-to-Text**: Whisper model via WebGPU/WASM
- **Voice Activity Detection**: Silero VAD detects when you're speaking
- **LLM**: Qwen 1.5B via WebLLM (easily swappable - see below)
- **Text-to-Speech**: Supertonic TTS with 10 natural voices

No audio leaves your device. No API keys needed. Just open and talk.

## Swap In Your Own LLM

**The built-in LLM is just a demo.** The real value is the voice pipeline - STT, VAD, and TTS all wired up and working. Rip out the tiny in-browser model and point it at any LLM you want:

- **Claude, GPT-4, Gemini** - via API routes
- **Ollama, LM Studio** - local inference servers  
- **Any OpenAI-compatible endpoint**

It's ~10 lines of code to swap. See [Using a Different LLM](#using-a-different-llm) below.

## Quick Start

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in Chrome or Edge.

## What Downloads When

| Asset | Size | When | Cached |
|-------|------|------|--------|
| Voice embeddings | ~500KB | Included in repo | ✓ Already local |
| Whisper STT model | ~150MB | First use | ✓ IndexedDB |
| Silero VAD model | ~2MB | First use | ✓ IndexedDB |
| Qwen 1.5B LLM | ~900MB | First use | ✓ IndexedDB |
| Supertonic TTS | ~50MB | First use | ✓ IndexedDB |

First load downloads ~1GB of models from HuggingFace CDN. After that, everything runs offline.

## Requirements

- **Browser**: Chrome 113+ or Edge 113+ (WebGPU required)
- **RAM**: ~4GB available for models
- **Microphone**: Required for voice input

Falls back to WASM if WebGPU unavailable (slower but works everywhere).

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                             │
│                                                             │
│  Microphone                                                 │
│       |                                                     │
│       v                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Silero   │ > │ Whisper  │ > │ WebLLM   │ > │Supertonic│ │
│  │ VAD      │   │ STT      │   │ (Qwen)   │   │ TTS      │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       |              |              |              |        │
│  Detects        Transcribes    Generates       Speaks      │
│  speech         to text        response        response    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Main voice chat UI
│   ├── layout.tsx            # App layout
│   └── globals.css           # Styles
├── components/ui/            # UI components
├── hooks/
│   ├── use-webllm.ts         # WebLLM integration
│   └── use-tts.ts            # TTS integration
└── lib/
    ├── tts.ts                # TTS pipeline
    └── splitter.ts           # Text chunking

public/
├── stt-worker-esm.js         # Whisper + VAD worker
├── vad-processor.js          # Audio worklet
└── voices/                   # TTS voice embeddings (bundled)
```

## Using a Different LLM

This demo uses WebLLM for fully local operation. To use an external LLM instead:

1. Create an API route (e.g., `src/app/api/chat/route.ts`)
2. In `page.tsx`, find `handleLLMResponse()` and replace the WebLLM call:

```typescript
// Instead of webllm.chat(), call your API:
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ messages: conversationHistory })
});
const data = await response.json();
return data.response;
```

## Tech Stack

- **Framework**: Next.js 16, React 19
- **STT**: Whisper via @huggingface/transformers
- **VAD**: Silero VAD via ONNX Runtime
- **LLM**: Qwen 1.5B via @mlc-ai/web-llm
- **TTS**: Supertonic via @huggingface/transformers
- **Styling**: Tailwind CSS v4

## Voice Options

10 voices bundled (5 female, 5 male):
- F1: Calm, steady
- F2: Bright, cheerful
- F3: Professional
- F4: Confident
- F5: Gentle
- M1: Lively, upbeat
- M2: Deep, calm
- M3: Authoritative
- M4: Soft, friendly
- M5: Warm

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [Whisper](https://github.com/openai/whisper) - OpenAI
- [Silero VAD](https://github.com/snakers4/silero-vad) - Silero Team
- [WebLLM](https://github.com/mlc-ai/web-llm) - MLC AI
- [Transformers.js](https://github.com/huggingface/transformers.js) - Hugging Face
- [Supertonic TTS](https://huggingface.co/onnx-community/Supertonic-TTS-ONNX) - Supertone
