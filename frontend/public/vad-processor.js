/**
 * VAD Audio Worklet Processor
 * Captures audio chunks and sends them to the main thread for VAD processing
 */

const MIN_CHUNK_SIZE = 512
let globalPointer = 0
const globalBuffer = new Float32Array(MIN_CHUNK_SIZE)

class VADProcessor extends AudioWorkletProcessor {
  process(inputs, _outputs, _parameters) {
    const buffer = inputs[0][0]
    if (!buffer) return true // Keep alive even if no input

    if (buffer.length >= MIN_CHUNK_SIZE) {
      // Buffer is large enough, send directly
      this.port.postMessage({ buffer: new Float32Array(buffer) })
    } else {
      const remaining = MIN_CHUNK_SIZE - globalPointer
      if (buffer.length >= remaining) {
        // Fill remaining space and send
        globalBuffer.set(buffer.subarray(0, remaining), globalPointer)
        this.port.postMessage({ buffer: new Float32Array(globalBuffer) })

        // Reset and store overflow
        globalBuffer.fill(0)
        globalBuffer.set(buffer.subarray(remaining), 0)
        globalPointer = buffer.length - remaining
      } else {
        // Accumulate
        globalBuffer.set(buffer, globalPointer)
        globalPointer += buffer.length
      }
    }

    return true
  }
}

registerProcessor("vad-processor", VADProcessor)
