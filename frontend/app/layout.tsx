import type React from "react"
import { AuthProvider } from "../context/AuthContext"
import ProvidersWithToken from "./ProvidersWithToken"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Toaster } from "@/components/ui/toaster"
import "./globals.css"

export const metadata: Metadata = {
  title: "Система прогнозирования",
  description: "Корпоративная платформа анализа и прогнозирования данных",
    generator: 'v0.dev'
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <head>
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body>
        <AuthProvider>
          <ProvidersWithToken>
            {children}
            <Toaster />
          </ProvidersWithToken>
        </AuthProvider>
      </body>
    </html>
  );
}
