"use client"
import React, { createContext, useContext, useState } from "react"

// Типы для данных Excel
export type ExcelContextType = {
  excelBuffer: ArrayBuffer | null
  setExcelBuffer: (buffer: ArrayBuffer | null) => void
  models: string[]
  setModels: (models: string[]) => void
  articles: string[]
  setArticles: (articles: string[]) => void
  parsedJson: any[]
  setParsedJson: (json: any[]) => void
  finalPredictionJson: any[]
  setFinalPredictionJson: (json: any[]) => void
  exchangeRatesJson: any[]
  setExchangeRatesJson: (json: any[]) => void
  // Добавляем сохранение состояния анализа
  selectedArticle: string
  setSelectedArticle: (article: string) => void
  selectedModels: string[]
  setSelectedModels: (models: string[]) => void
}

const ExcelContext = createContext<ExcelContextType | undefined>(undefined)

export function useExcelContext() {
  const ctx = useContext(ExcelContext)
  if (!ctx) throw new Error("useExcelContext must be used within ExcelProvider")
  return ctx
}

export function ExcelProvider({ children }: { children: React.ReactNode }) {
  const [excelBuffer, setExcelBuffer] = useState<ArrayBuffer | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [articles, setArticles] = useState<string[]>([])
  const [parsedJson, setParsedJson] = useState<any[]>([])
  const [finalPredictionJson, setFinalPredictionJson] = useState<any[]>([])
  const [exchangeRatesJson, setExchangeRatesJson] = useState<any[]>([])
  // Добавляем состояние для анализа
  const [selectedArticle, setSelectedArticle] = useState<string>("")
  const [selectedModels, setSelectedModels] = useState<string[]>([])

  return (
    <ExcelContext.Provider value={{ 
      excelBuffer, setExcelBuffer, 
      models, setModels, 
      articles, setArticles, 
      parsedJson, setParsedJson,
      finalPredictionJson, setFinalPredictionJson,
  exchangeRatesJson, setExchangeRatesJson,
      selectedArticle, setSelectedArticle,
      selectedModels, setSelectedModels
    }}>
      {children}
    </ExcelContext.Provider>
  )
}
