"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Download, BarChart3, TrendingUp, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, XAxis, YAxis, ResponsiveContainer, Legend } from "recharts"

// Mock data for charts
const mockForecastData = [
  { date: "2024-01", forecast: 1200, actual: 1150, error: 4.2 },
  { date: "2024-02", forecast: 1300, actual: 1280, error: 1.5 },
  { date: "2024-03", forecast: 1250, actual: 1320, error: -5.6 },
  { date: "2024-04", forecast: 1400, actual: 1380, error: 1.4 },
  { date: "2024-05", forecast: 1350, actual: 1290, error: 4.4 },
  { date: "2024-06", forecast: 1450, actual: 1420, error: 2.1 },
  { date: "2024-07", forecast: 1500, actual: 1480, error: 1.3 },
  { date: "2024-08", forecast: 1380, actual: 1410, error: -2.2 },
  { date: "2024-09", forecast: 1420, actual: 1390, error: 2.1 },
  { date: "2024-10", forecast: 1480, actual: 1520, error: -2.7 },
  { date: "2024-11", forecast: 1550, actual: 1530, error: 1.3 },
  { date: "2024-12", forecast: 1600, actual: 1580, error: 1.3 },
]

const mockArticles = [
  "Статья 1 - Основные материалы",
  "Статья 2 - Вспомогательные материалы",
  "Статья 3 - Энергоресурсы",
  "Статья 4 - Транспортные расходы",
  "Статья 5 - Прочие расходы",
]

const chartConfig = {
  forecast: {
    label: "Прогноз",
    color: "hsl(var(--chart-1))",
  },
  actual: {
    label: "Факт",
    color: "hsl(var(--chart-2))",
  },
  error: {
    label: "Ошибка %",
    color: "hsl(var(--chart-3))",
  },
}

function ForecastChart({ data }: { data: any[] }) {
  return (
    <ChartContainer config={chartConfig} className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="var(--color-forecast)"
            strokeWidth={3}
            dot={{ fill: "var(--color-forecast)", strokeWidth: 2, r: 4 }}
            name="Прогноз"
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="var(--color-actual)"
            strokeWidth={3}
            dot={{ fill: "var(--color-actual)", strokeWidth: 2, r: 4 }}
            name="Факт"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

function ErrorChart({ data }: { data: any[] }) {
  return (
    <ChartContainer config={chartConfig} className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          <Line
            type="monotone"
            dataKey="error"
            stroke="var(--color-error)"
            strokeWidth={3}
            dot={{ fill: "var(--color-error)", strokeWidth: 2, r: 4 }}
            name="Ошибка %"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

export function AnalysisPage() {
  const [selectedArticle, setSelectedArticle] = useState(mockArticles[0])

  const handleDownloadExcel = async () => {
    try {
      // Mock Excel download
      const mockData = "mock excel data for forecast"
      const blob = new Blob([mockData], {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `forecast_${selectedArticle.replace(/\s+/g, "_")}.xlsx`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert("Не удалось скачать Excel файл")
    }
  }

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Анализ прогнозов</h1>
        <p className="text-gray-600 text-lg">Анализ точности прогнозирования и сравнение с фактическими данными</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
        {/* Controls */}
        <Card className="shadow-lg border-gray-200 bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-900">
              <BarChart3 className="w-5 h-5 text-blue-600" />
              Параметры анализа
            </CardTitle>
            <CardDescription>Выберите статью ЧОК для анализа и скачайте отчет в Excel</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label className="text-gray-800 font-semibold">Статья ЧОК</Label>
                <Select value={selectedArticle} onValueChange={setSelectedArticle}>
                  <SelectTrigger className="border-gray-300 focus:border-blue-500 focus:ring-blue-500">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {mockArticles.map((article) => (
                      <SelectItem key={article} value={article}>
                        {article}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-end">
                <Button
                  onClick={handleDownloadExcel}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Скачать прогноз Excel
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Separator className="bg-gray-200" />

        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Forecast vs Actual Chart */}
          <Card className="shadow-lg border-gray-200 bg-white">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-gray-900">
                <TrendingUp className="w-5 h-5 text-blue-600" />
                Прогноз vs Факт
              </CardTitle>
              <CardDescription>
                Сравнение прогнозируемых и фактических значений по датам для {selectedArticle}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ForecastChart data={mockForecastData} />
            </CardContent>
          </Card>

          {/* Error Chart */}
          <Card className="shadow-lg border-gray-200 bg-white">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-gray-900">
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                Ошибка прогноза
              </CardTitle>
              <CardDescription>Процентная ошибка прогнозирования по датам для {selectedArticle}</CardDescription>
            </CardHeader>
            <CardContent>
              <ErrorChart data={mockForecastData} />
            </CardContent>
          </Card>
        </div>

        {/* Statistics Summary */}
        <Card className="shadow-lg border-gray-200 bg-white">
          <CardHeader>
            <CardTitle className="text-gray-900">Статистика точности</CardTitle>
            <CardDescription>Основные метрики качества прогнозирования для {selectedArticle}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-2xl font-bold text-blue-700">2.1%</div>
                <div className="text-sm text-blue-600 font-medium">Средняя ошибка</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                <div className="text-2xl font-bold text-green-700">5.6%</div>
                <div className="text-sm text-green-600 font-medium">Макс. ошибка</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                <div className="text-2xl font-bold text-purple-700">97.9%</div>
                <div className="text-sm text-purple-600 font-medium">Точность</div>
              </div>
              <div className="text-center p-4 bg-amber-50 rounded-lg border border-amber-200">
                <div className="text-2xl font-bold text-amber-700">12</div>
                <div className="text-sm text-amber-600 font-medium">Периодов</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </main>
  )
}
