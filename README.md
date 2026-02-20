# Enterprise AI Voice Assistant

A high-performance, hybrid-edge voice assistant leveraging **WebGPU** for real-time client-side AI and **Python/Camunda** for backend business orchestration.

## 🏗️ Hybrid Architecture Summary

This system utilizes a hybrid-edge approach—leveraging the user's local hardware for latency-sensitive AI tasks while maintaining a secure Python-based bridge to enterprise workflows.

### 1. Native npm Frontend (Next.js)
- **Framework:** Next.js 16 + React 19 + Tailwind CSS 4.
- **Environment:** High-performance **npm-based** setup located in `voice-chat/frontend`.
- **Requirements:** Requires **Node.js v20+** for engine compatibility with Next.js 16 and Turbopack.

### 2. In-Browser Multilingual AI (WebGPU)
All heavy AI inference occurs directly in your browser using **WebGPU** acceleration:
- **STT (Speech-to-Text):** **Transformers.js (v3)** running `Whisper-tiny` in a dedicated Web Worker with **VAD (Voice Activity Detection)** for efficient recording.
- **LLM (Large Language Model):** **WebLLM** running `Qwen2.5-1.5B` (Desktop) or `Qwen2.5-0.5B` (**iOS/Mobile alternative**) for optimized memory usage. Supports real-time interruption via `interruptGenerate()`.
- **TTS (Text-to-Speech):** **Supertonic-TTS-2** (ONNX) providing high-quality synthesis with **10 distinct personas**. "Gentle" (F1) is the default.

### 3. Python Backend & Camunda Integration
Located in `voice-chat/backend`, the **Python 3.13** server coordinates business logic:
- **FastAPI Core:** Real-time communication via **WebSockets**.
- **Camunda/Zeebe Client:** Native integration using `pyzeebe` to trigger and monitor enterprise workflows.
- **Enterprise Security:** Built-in **OAuth2** token management and **PyJWT** decoding for authenticated API transactions.
- **HF Proxy:** Caching proxy for model shards and voice assets to bypass CORS and improve performance.

### 4. Globalized Capabilities (EN, FR, ES)
- **Native Multi-Language:** Transcription and synthesis for **English, French, and Spanish**.
- **Localized Stalling:** A "Junior Receptionist" behavior engine provides 21+ localized safe phrases (e.g., *"One moment, I'm just verifying those records..."*) to mask backend processing latency.

---

## 🚀 Quick Start

### Frontend (npm)
```bash
cd voice-chat/frontend
npm install
npm run dev
```
*Note: Ensure Node.js is v20+.*

### Backend (Python)
```bash
cd voice-chat/backend
# (Optional) Create venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## 🛠️ Tech Stack
- **STT/TTS:** Transformers.js + ONNX Runtime Web
- **LLM:** WebLLM + WebGPU
- **Backend:** FastAPI + PyZeebe + PyJWT
- **Styling:** Tailwind CSS v4

---

## 📜 License
MIT License - see [LICENSE](LICENSE)

## 🤝 Credits
- [Next.js](https://nextjs.org/)
- [Camunda/Zeebe](https://camunda.com/)
- [WebLLM](https://webllm.mlc-ai.org/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [Supertonic TTS](https://huggingface.co/onnx-community/Supertonic-TTS-ONNX)
