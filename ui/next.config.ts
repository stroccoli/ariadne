import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: produces out/ directory served by FastAPI in production.
  // In development, `npm run dev` still works normally.
  output: "export",
};

export default nextConfig;
