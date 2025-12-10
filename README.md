# Voice Chat - 100% In-Browser

A fully browser-based voice assistant. Speech recognition, LLM, and text-to-speech all run locally in your browser using WebGPU - no API keys, no server, no data leaves your device.

## Live Demo

Try it now: [HuggingFace Space](https://huggingface.co/spaces/RickRossTN/voice-chat)

## What Makes This Different

**Everything runs in your browser:**
- **Speech-to-Text**: Whisper model via WebGPU/WASM
- **Voice Activity Detection**: Silero VAD detects when you're speaking
- **LLM**: Qwen 1.5B runs directly in the browser via WebLLM
- **Text-to-Speech**: Supertonic TTS with 10 natural voices

No audio leaves your device. No API keys needed. Just open and talk.

## Quick Start

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in Chrome or Edge.

First load downloads ~1GB of models (cached in browser for future visits).

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
│   ├── conversation.tsx      # Chat display
│   ├── message.tsx           # Message bubbles
│   ├── live-waveform.tsx     # Audio visualizer
│   └── button.tsx            # UI primitives
├── hooks/
│   ├── use-webllm.ts         # WebLLM integration
│   └── use-tts.ts            # TTS integration
└── lib/
    ├── tts.ts                # TTS pipeline
    └── splitter.ts           # Text chunking

public/
├── stt-worker-esm.js         # Whisper + VAD worker
├── vad-processor.js          # Audio worklet
└── voices/                   # TTS voice embeddings (F1-F5, M1-M5)
```

## Adding Your Own LLM

This demo uses an in-browser LLM for fully local operation. To connect an external LLM:

1. Create an API route at `/api/chat`
2. Modify `handleLLMResponse()` in `page.tsx` to call your API
3. See comments in `page.tsx` for example code

## Tech Stack

- **Framework**: Next.js 16, React 19
- **STT**: @huggingface/transformers (Whisper)
- **VAD**: Silero VAD via ONNX Runtime
- **LLM**: @mlc-ai/web-llm (Qwen 1.5B)
- **TTS**: Supertonic via @huggingface/transformers
- **Styling**: Tailwind CSS v4

## Voice Options

10 voices included (5 female, 5 male):
- F1: Calm, steady
- F2: Bright, cheerful
- F3: Professional announcer
- F4: Confident, expressive
- F5: Gentle, soothing
- M1: Lively, upbeat
- M2: Deep, calm
- M3: Authoritative
- M4: Soft, friendly
- M5: Warm, storyteller

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [Whisper](https://github.com/openai/whisper) - OpenAI
- [Silero VAD](https://github.com/snakers4/silero-vad) - Silero Team
- [WebLLM](https://github.com/mlc-ai/web-llm) - MLC AI
- [Transformers.js](https://github.com/huggingface/transformers.js) - Hugging Face
- [Supertonic TTS](https://github.com/supertone-inc/supertonic-py) - Supertone
