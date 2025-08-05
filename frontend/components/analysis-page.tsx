"use client"

import { useState } from "react"
import { useExcelContext } from "../context/ExcelContext"
import { motion } from "framer-motion"
import { Download, BarChart3, TrendingUp, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, XAxis, YAxis, ResponsiveContainer, Legend } from "recharts"
import { downloadExcel } from "./utils/downloadExcel"
import { fetchExcelAndParseModels } from "./utils/fetchExcelAndParseModels"
import { parseYamlConfig } from "./utils/parseYaml"
import { fetchExcelDataForChart } from "./utils/fetchExcelDataForChart"
import * as XLSX from "xlsx"
import { useEffect } from "react"

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

function excelSerialToDate(serial: number): string {
  // Excel date serial to JS Date (days since 1899-12-30)
  const excelEpoch = new Date(Date.UTC(1899, 11, 30))
  const msPerDay = 24 * 60 * 60 * 1000
  const date = new Date(excelEpoch.getTime() + serial * msPerDay)
  // Формат YYYY-MM-DD
  return date.toISOString().slice(0, 10)
}

function ForecastChart({ data }: { data: any[] }) {
  // преобразуем даты
  const chartData = data.map(d => ({ ...d, date: typeof d.date === 'number' ? excelSerialToDate(d.date) : d.date }))
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          <Line
            type="linear"
            dataKey="forecast"
            stroke="var(--color-forecast)"
            strokeWidth={3}
            dot={{ fill: "var(--color-forecast)", strokeWidth: 2, r: 4 }}
            name="Прогноз"
          />
          <Line
            type="linear"
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
  // ошибка в % (может быть положительной и отрицательной)
  const chartData = data.map(d => ({
    ...d,
    date: typeof d.date === 'number' ? excelSerialToDate(d.date) : d.date,
    percentError: d.forecast != null && d.actual != null && d.actual !== 0 ? ((d.forecast - d.actual) / d.actual * 100) : null
  }))
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          <Line
            type="linear"
            dataKey="percentError"
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

function AbsoluteDiffChart({ data }: { data: any[] }) {
  // абсолютная разница (может быть положительной и отрицательной)
  const chartData = data.map(d => ({
    ...d,
    date: typeof d.date === 'number' ? excelSerialToDate(d.date) : d.date,
    absDiff: d.forecast != null && d.actual != null ? (d.forecast - d.actual) : null
  }))
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          <Line
            type="linear"
            dataKey="absDiff"
            stroke="var(--color-forecast)"
            strokeWidth={3}
            dot={{ fill: "var(--color-forecast)", strokeWidth: 2, r: 4 }}
            name="Разница"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

export function AnalysisPage() {
  const [chartData, setChartData] = useState<any[]>([])
  const [chartLoading, setChartLoading] = useState(false)
  const [selectedArticle, setSelectedArticle] = useState<string>("")
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const {
    excelBuffer,
    setExcelBuffer,
    models,
    setModels,
    articles,
    setArticles,
    parsedJson,
    setParsedJson,
  } = useExcelContext()

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        // Если данные уже есть в контексте, не загружаем заново
        if (excelBuffer && models.length > 0 && articles.length > 0 && parsedJson.length > 0) {
          setSelectedArticle(articles[0] || "");
          setSelectedModel(models[0] || "");
          setLoading(false);
          return;
        }
        // Загружаем Excel-файл только если его нет
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
        const url = backendUrl.replace(/\/$/, '') + '/export_excel/';
        const response = await fetch(url);
        if (!response.ok) throw new Error('Ошибка скачивания файла');
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        setExcelBuffer(arrayBuffer);
        // Получаем модели через utils
        const filteredModels = await fetchExcelAndParseModels(arrayBuffer);
        setModels(filteredModels);
        // fetch articles from config_refined.yaml
        const yamlRes = await fetch("/config_refined.yaml");
        const yamlText = await yamlRes.text();
        const config = parseYamlConfig(yamlText);
        const articleList = config && config["Статья"] ? Object.keys(config["Статья"]) : [];
        setArticles(articleList);
        setSelectedArticle(articleList[0] || "");
        setSelectedModel(filteredModels[0] || "");
        // Кэшируем распаршенный json
        const workbook = XLSX.read(arrayBuffer, { type: "array" });
        const sheet = workbook.Sheets["data"] || workbook.Sheets[workbook.SheetNames[0]];
        const json = XLSX.utils.sheet_to_json(sheet);
        setParsedJson(json);
      } catch (err) {
        setArticles([]);
        setModels([]);
        setParsedJson([]);
      }
      setLoading(false);
    }
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedArticle || !selectedModel || !parsedJson || !parsedJson.length) return;
    setChartLoading(true);
    try {
      let fixedArticleName = selectedArticle;
      if (selectedArticle.trim().toLowerCase() === 'торговая дз') {
        fixedArticleName = selectedArticle + '_USD';
      }
      const filtered = parsedJson.filter((row: any) => {
        const val = row["Статья"];
        return val && val.trim().toLowerCase() === fixedArticleName.trim().toLowerCase();
      });
      const chartData = filtered.map((row: any) => ({
        date: row["Дата"],
        actual: row["Fact"],
        forecast: row[`predict_${selectedModel}`],
        error: row[`predict_${selectedModel}`] && row["Fact"] ? Number((((row[`predict_${selectedModel}`] - row["Fact"]) / row["Fact"]) * 100).toFixed(2)) : null
      }));
      setChartData(chartData);
    } catch (err) {
      setChartData([]);
    }
    setTimeout(() => setChartLoading(false), 300);
  }, [selectedArticle, selectedModel, parsedJson]);

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Анализ прогнозов</h1>
        <p className="text-gray-600 text-lg">Анализ точности прогнозирования и сравнение с фактическими данными</p>
      </motion.div>

      {/* Карточка скачивания прогноза сразу под заголовком */}
      <Card className="shadow-lg border-gray-200 bg-white mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-900">
            <Download className="w-5 h-5 text-green-600" />
            Скачать прогноз
          </CardTitle>
          <CardDescription>
            Скачайте полный файл прогноза в формате Excel. Включает все статьи и модели.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            variant="outline"
            className="mt-2"
            onClick={async () => {
              try {
                const blob = await downloadExcel();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'Прогноз.xlsx';
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
              } catch (err) {
                alert('Ошибка скачивания файла');
              }
            }}
          >
            <Download className="w-4 h-4 mr-2" /> Скачать результаты (Excel)
          </Button>
        </CardContent>
      </Card>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
        {/* Controls */}
        <Card className="shadow-lg border-gray-200 bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-900">
              <BarChart3 className="w-5 h-5 text-blue-600" />
              Параметры анализа
            </CardTitle>
            <CardDescription>Выберите статью ЧОК и модель для анализа. Excel подгружается автоматически.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {loading ? (
              <div className="text-center py-8 text-lg text-gray-500">Загрузка данных...</div>
            ) : (
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <Label className="text-gray-800 font-semibold">Статья ЧОК</Label>
                  <Select value={selectedArticle} onValueChange={setSelectedArticle}>
                    <SelectTrigger className="border-gray-300 focus:border-blue-500 focus:ring-blue-500">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {articles.map((article) => (
                        <SelectItem key={article} value={article}>
                          {article}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-3">
                  <Label className="text-gray-800 font-semibold">Модель</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="border-gray-300 focus:border-blue-500 focus:ring-blue-500">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {models.map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Separator className="bg-gray-200" />

        {/* Charts and stats only after model selected */}
        {!loading && selectedModel && selectedArticle && (
          chartLoading ? (
            <div className="text-center py-8 text-lg text-gray-500">Загрузка графиков...</div>
          ) : (
            <>
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
                  <ForecastChart data={chartData} />
                </CardContent>
              </Card>

              <Card className="shadow-lg border-gray-200 bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-gray-900">
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                    Ошибка %
                  </CardTitle>
                  <CardDescription>Ошибка прогноза в процентах по датам для {selectedArticle}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ErrorChart data={chartData} />
                </CardContent>
              </Card>

              <Card className="shadow-lg border-gray-200 bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-gray-900">
                    <BarChart3 className="w-5 h-5 text-purple-600" />
                    Разница
                  </CardTitle>
                  <CardDescription>Абсолютная разница между прогнозом и фактом по датам для {selectedArticle}</CardDescription>
                </CardHeader>
                <CardContent>
                  <AbsoluteDiffChart data={chartData} />
                </CardContent>
              </Card>

              {/* Statistics Summary */}
              <Card className="shadow-lg border-gray-200 bg-white">
                <CardHeader>
                  <CardTitle className="text-gray-900">Статистика точности</CardTitle>
                  <CardDescription>Основные метрики качества прогнозирования для {selectedArticle}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-4 gap-6">
                    <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="text-2xl font-bold text-blue-700">{chartData.length ? `${(chartData.reduce((acc, d) => acc + (d.error ?? 0), 0) / chartData.length).toFixed(2)}%` : '-'}</div>
                      <div className="text-sm text-blue-600 font-medium">Средняя ошибка</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                      <div className="text-2xl font-bold text-green-700">{chartData.length ? `${Math.max(...chartData.map(d => Math.abs(d.error ?? 0))).toFixed(2)}%` : '-'}</div>
                      <div className="text-sm text-green-600 font-medium">Макс. ошибка</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <div className="text-2xl font-bold text-purple-700">{chartData.length ? `${(100 - chartData.reduce((acc, d) => acc + Math.abs(d.error ?? 0), 0) / chartData.length).toFixed(2)}%` : '-'}</div>
                      <div className="text-sm text-purple-600 font-medium">Точность</div>
                    </div>
                    <div className="text-center p-4 bg-amber-50 rounded-lg border border-amber-200">
                      <div className="text-2xl font-bold text-amber-700">{chartData.length}</div>
                      <div className="text-sm text-amber-600 font-medium">Периодов</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )
        )}
      </motion.div>
    </main>
  )
}
