import type React from "react"
import { ExcelProvider } from "../context/ExcelContext"
import { ConfigProvider } from "../context/ConfigContext"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import "./globals.css"

export const metadata: Metadata = {
  title: "Система прогнозирования",
  description: "Корпоративная платформа анализа и прогнозирования данных",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
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
        <ConfigProvider>
          <ExcelProvider>
            {children}
          </ExcelProvider>
        </ConfigProvider>
      </body>
    </html>
  )
}
