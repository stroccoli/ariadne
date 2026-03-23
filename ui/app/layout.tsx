import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Cinzel } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

const cinzel = Cinzel({
  subsets: ["latin"],
  variable: "--font-cinzel",
  weight: ["700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Ariadne — AI Incident Analyzer",
  description:
    "From raw logs to root cause. Paste your incident logs, get a structured diagnosis with confidence scores and recommended actions.",
  openGraph: {
    title: "Ariadne — AI Incident Analyzer",
    description:
      "From raw logs to root cause. AI-powered incident analysis for engineering teams.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} ${cinzel.variable}`}
    >
      <body className="min-h-screen bg-bg-primary font-sans text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
