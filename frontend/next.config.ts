import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export for HuggingFace Spaces
  output: "export",
  // Disable dev indicators that show ONNX warnings
  devIndicators: false,
  // Turbopack is default in Next.js 16
  turbopack: {},
};

export default nextConfig;
