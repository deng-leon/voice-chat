import type { Metadata } from "next"
import "./globals.css"
import Script from "next/script"

export const metadata: Metadata = {
  title: "Voice Chat",
  description: "Real-time AI voice chat running on your device",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        {/* Suppress noisy library warnings before any modules load */}
        <Script id="suppress-onnx" strategy="beforeInteractive">{`
          (function() {
            const originalLog = console.log;
            const originalWarn = console.warn;
            const originalError = console.error;
            const originalInfo = console.info;
            const originalDebug = console.debug;

            const suppress = (...args) => {
              const msg = args.map(arg => {
                try { return String(arg); } catch(e) { return ""; }
              }).join(' ');
              
              // Strip ANSI escape codes (like colors) and convert to lower case
              const cleanMsg = msg.replace(/\\x1B\\[[0-9;]*[mK]/g, '').toLowerCase();
              
              return (
                cleanMsg.includes('onnxruntime') || 
                cleanMsg.includes('verifyeachnodeisassignedtoanep') ||
                cleanMsg.includes('session_state.cc') ||
                cleanMsg.includes('execution providers') ||
                cleanMsg.includes('content-length') ||
                cleanMsg.includes('unknown model class') ||
                msg.includes('[W:onnxruntime')
              );
            };

            const interceptor = (original) => function(...args) {
              if (suppress(...args)) return;
              original.apply(console, args);
            };

            console.log = interceptor(originalLog);
            console.warn = interceptor(originalWarn);
            console.error = interceptor(originalError);
            console.info = interceptor(originalInfo);
            console.debug = interceptor(originalDebug);
          })();
        `}</Script>
      </head>
      <body>{children}</body>
    </html>
  )
}
