"use client"

import React, { useState } from "react"
import { motion } from "framer-motion"
import { Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { downloadTaxExcel } from "./utils/downloadTaxExcel"
import { useAuth } from "../context/AuthContext"

export function TaxExportPage() {
  const { session } = useAuth();
  const accessToken = session?.access_token;
  const [loadingExcel, setLoadingExcel] = useState(false)

  const handleExcelDownload = async () => {
    setLoadingExcel(true)
    try {
      const blob = await downloadTaxExcel(accessToken)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const currentDate = new Date().toISOString().split('T')[0]
      a.download = `tax_forecast_${currentDate}.zip`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Ошибка скачивания файла')
    } finally {
      setLoadingExcel(false)
    }
  }

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Выгрузка</h1>
        <p className="text-gray-600 text-lg">Скачивание результатов прогноза налогов</p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Карточка скачивания прогноза */}
        <Card className="shadow-xl border-gray-200 bg-white flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-900">
              <Download className="w-5 h-5 text-green-600" />
              Скачать прогноз
            </CardTitle>
            <CardDescription>
              Скачайте архив с файлами прогноза налогов в формате Excel.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-end">
            <Button
              onClick={handleExcelDownload}
              className="w-full bg-green-600 hover:bg-green-700 text-white py-4 text-base font-semibold shadow-lg"
              disabled={loadingExcel}
            >
              {loadingExcel ? (
                <>
                  <svg className="animate-spin w-4 h-4 mr-2 text-white" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  Загрузка...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Скачать архив
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
