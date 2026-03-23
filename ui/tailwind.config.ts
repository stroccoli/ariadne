import type { Config } from "tailwindcss";
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gold: {
          DEFAULT: "#D4A24C",
          light: "#E8B85E",
          muted: "rgba(212, 162, 76, 0.12)",
        },
        surface: {
          DEFAULT: "#141B2D",
          input: "#1C2438",
        },
        accent: "#6366F1",
        confidence: {
          high: "#34D399",
          mid: "#FBBF24",
          low: "#F87171",
        },
        border: "#1E293B",
        "bg-primary": "#0B0F1A",
      },
      fontFamily: {
        sans: ["var(--font-inter)", ...defaultTheme.fontFamily.sans],
        mono: ["var(--font-jetbrains)", ...defaultTheme.fontFamily.mono],
        display: ["var(--font-cinzel)", "serif"],
      },
      textColor: {
        primary: "#E8ECF4",
        secondary: "#6B7A94",
        muted: "#3D4B63",
      },
    },
  },
  plugins: [],
};

export default config;
