"use client"

import React, { useState } from "react"
import { useExcelContext } from "../context/ExcelContext"
import { motion } from "framer-motion"
import { BarChart3, TrendingUp, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, XAxis, YAxis, ResponsiveContainer, Legend } from "recharts"
import { fetchExcelAndParseModels } from "./utils/fetchExcelAndParseModels"
import { parseYamlConfig } from "./utils/parseYaml"
import { fetchExcelDataForChart } from "./utils/fetchExcelDataForChart"
import * as XLSX from "xlsx"
import { useEffect } from "react"
import { useConfig } from "../context/ConfigContext"

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

function ForecastChart({ data, mainModelLower }: { data: any[], mainModelLower?: string }) {
  // data: [{ model, data: [...] }, ...]
  // Merge all models' data by date
  const dates = Array.from(new Set(data.flatMap(d => d.data.map((row: any) => typeof row.date === 'number' ? excelSerialToDate(row.date) : row.date))));
  // Build chartData: [{ date, [model1_forecast], [model1_actual], ... }]
  const chartData = dates.map(date => {
    const row: any = { date };
    data.forEach(({ model, data: modelData }: any) => {
      const found = modelData.find((r: any) => (typeof r.date === 'number' ? excelSerialToDate(r.date) : r.date) === date);
      row[`${model}_forecast`] = found ? found.forecast : null;
      row[`${model}_actual`] = found ? found.actual : null;
    });
    return row;
  });
  // Цвета для моделей
  const modelColors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          {/* Одна линия факта (берём первую модель, т.к. actual одинаковый) */}
          <Line
            type="linear"
            dataKey={`${data[0]?.model}_actual`}
            stroke="var(--color-actual)"
            strokeWidth={3}
            dot={{ fill: "var(--color-actual)", strokeWidth: 2, r: 4 }}
            name="Факт"
            strokeDasharray={mainModelLower ? undefined : undefined} // всегда сплошная
          />
          {/* Каждая модель своим цветом */}
          {data.map(({ model }, idx) => (
            <Line
              key={model + "-forecast"}
              type="linear"
              dataKey={`${model}_forecast`}
              stroke={modelColors[idx % modelColors.length]}
              strokeWidth={3}
              dot={{ fill: modelColors[idx % modelColors.length], strokeWidth: 2, r: 4 }}
              name={`Прогноз (${model})`}
              strokeDasharray={mainModelLower && model.toLowerCase() === mainModelLower ? undefined : "6 4"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

function ErrorChart({ data, mainModelLower }: { data: any[], mainModelLower?: string }) {
  // data: [{ model, data: [...] }, ...]
  const dates = Array.from(new Set(data.flatMap(d => d.data.map((row: any) => typeof row.date === 'number' ? excelSerialToDate(row.date) : row.date))));
  const modelColors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  const chartData = dates.map(date => {
    const row: any = { date };
    data.forEach(({ model, data: modelData }: any) => {
      const found = modelData.find((r: any) => (typeof r.date === 'number' ? excelSerialToDate(r.date) : r.date) === date);
      // Берем готовое значение ошибки из Excel
      row[`${model}_error`] = found ? found.errorPercent : null;
    });
    return row;
  });
  
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          {data.map(({ model }, idx) => (
            <Line
              key={model + "-error"}
              type="linear"
              dataKey={`${model}_error`}
              stroke={modelColors[idx % modelColors.length]}
              strokeWidth={3}
              dot={{ fill: modelColors[idx % modelColors.length], strokeWidth: 2, r: 4 }}
              name={`Ошибка % (${model})`}
              strokeDasharray={mainModelLower && model.toLowerCase() === mainModelLower ? undefined : "6 4"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

function AbsoluteDiffChart({ data, mainModelLower }: { data: any[], mainModelLower?: string }) {
  // data: [{ model, data: [...] }, ...]
  const dates = Array.from(new Set(data.flatMap(d => d.data.map((row: any) => typeof row.date === 'number' ? excelSerialToDate(row.date) : row.date))));
  const modelColors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  const chartData = dates.map(date => {
    const row: any = { date };
    data.forEach(({ model, data: modelData }: any) => {
      const found = modelData.find((r: any) => (typeof r.date === 'number' ? excelSerialToDate(r.date) : r.date) === date);
      // Берем готовое значение разности из Excel
      row[`${model}_diff`] = found ? found.difference : null;
    });
    return row;
  });
  
  return (
    <ChartContainer config={chartConfig} className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} className="text-gray-600" />
          <YAxis tickLine={false} axisLine={false} className="text-gray-600" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Legend />
          {data.map(({ model }, idx) => (
            <Line
              key={model + "-diff"}
              type="linear"
              dataKey={`${model}_diff`}
              stroke={modelColors[idx % modelColors.length]}
              strokeWidth={3}
              dot={{ fill: modelColors[idx % modelColors.length], strokeWidth: 2, r: 4 }}
              name={`Разница (${model})`}
              strokeDasharray={mainModelLower && model.toLowerCase() === mainModelLower ? undefined : "6 4"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

export function AnalysisPage() {
  const { config, loading: configLoading } = useConfig();
  const [chartData, setChartData] = useState<any[]>([])
  const [chartLoading, setChartLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showFullTable, setShowFullTable] = useState(false)
  const {
    excelBuffer,
    setExcelBuffer,
    models,
    setModels,
    articles,
    setArticles,
    parsedJson,
    setParsedJson,
    // Используем состояние из контекста
    selectedArticle,
    setSelectedArticle,
    selectedModels,
    setSelectedModels,
  } = useExcelContext()

  // Получаем главную модель для выбранной статьи
  const mainModel = config && config.model_article && selectedArticle ? config.model_article[selectedArticle] : null;
  const mainModelLower = mainModel ? mainModel.toLowerCase() : null;

  // Логгирование для отладки определения главной модели
  useEffect(() => {
  }, [selectedArticle, mainModel, config, models]);

  // При смене статьи выбираем только главную модель, но только если это реально смена статьи
  useEffect(() => {
    if (!selectedArticle || models.length === 0) return;
    
    // Выбираем главную модель только если еще ничего не выбрано
    if (selectedModels.length === 0) {
      if (mainModelLower) {
        const found = models.find((m: string) => m.toLowerCase() === mainModelLower);
        if (found) {
          setSelectedModels([found]);
          return;
        }
      }
      // Если главная модель не найдена, выбираем первую модель
      if (models.length > 0) {
        setSelectedModels([models[0]]);
      }
    }
  }, [mainModelLower, models, selectedArticle, selectedModels.length]);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        // Если данные уже есть в контексте, не загружаем заново
        if (excelBuffer && models.length > 0 && articles.length > 0 && parsedJson.length > 0) {
          // Если статья еще не выбрана, выбираем первую
          if (!selectedArticle && articles.length > 0) {
            setSelectedArticle(articles[0]);
          }
          // Не перезаписываем выбранные модели - они уже сохранены в контексте
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
        // get articles from config context
        const articleList = config && config["Статья"] ? Object.keys(config["Статья"]) : [];
        setArticles(articleList);
        // Инициализируем выбор только если еще не выбрано
        if (!selectedArticle && articleList.length > 0) {
          setSelectedArticle(articleList[0]);
        }
        // Не устанавливаем модели здесь - это будет делать useEffect с главной моделью
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
    if (!configLoading) fetchData();
  }, [configLoading, config]);

  useEffect(() => {
    if (!selectedArticle || !selectedModels.length || !parsedJson || !parsedJson.length) return;
    setChartLoading(true);
    try {
      // Для "Торговая дз" ищем в обеих формах: с "_USD" и без
      const filtered = parsedJson.filter((row: any) => {
        const val = row["Статья"];
        if (!val) return false;
        
        const rowArticleName = val.trim().toLowerCase();
        const selectedArticleLower = selectedArticle.trim().toLowerCase();
        
        if (selectedArticleLower === 'торговая дз') {
          // Для "Торговая дз" проверяем оба варианта
          return rowArticleName === 'торговая дз' || rowArticleName === 'торговая дз_usd';
        } else {
          // Для остальных статей - точное совпадение
          return rowArticleName === selectedArticleLower;
        }
      });
      // Для каждой модели — свой массив данных
      const allChartData = selectedModels.map(model => ({
        model,
        data: filtered.map((row: any) => ({
          date: row["Дата"],
          actual: row["Fact"],
          forecast: row[`predict_${model}`],
          // Берем готовые значения из Excel (обратите внимание на пробелы в названиях)
          errorPercent: row[`predict_${model} отклонение  %`], // два пробела перед %
          difference: row[`predict_${model} разница`],
          // Оставляем старый расчет как fallback для совместимости
          error: row[`predict_${model} отклонение  %`] || (row[`predict_${model}`] && row["Fact"] ? Number((((row[`predict_${model}`] - row["Fact"]) / row["Fact"]) * 100).toFixed(2)) : null)
        }))
      }));
      
      setChartData(allChartData);
    } catch (err) {
      setChartData([]);
    }
    setTimeout(() => setChartLoading(false), 300);
  }, [selectedArticle, selectedModels, parsedJson]);

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Анализ прогнозов</h1>
        <p className="text-gray-600 text-lg">Анализ точности прогнозирования и сравнение с фактическими данными</p>
      </motion.div>

      {/* Карточка параметров анализа */}
      <Card className="shadow-lg border-gray-200 bg-white mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-900">
            <BarChart3 className="w-5 h-5 text-blue-600" />
            Параметры анализа
          </CardTitle>
          <CardDescription>Выберите статью ЧОК и модели для анализа. Excel подгружается автоматически.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {loading ? (
            <div className="text-center py-8 text-lg text-gray-500">Загрузка данных...</div>
          ) : (
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label className="text-gray-800 font-semibold">Статья ЧОК</Label>
                <Select 
                  value={selectedArticle} 
                  onValueChange={(newArticle) => {
                    setSelectedArticle(newArticle);
                    // При смене статьи сбрасываем выбор моделей, чтобы useEffect выбрал главную модель
                    setSelectedModels([]);
                  }}
                >
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
                <Label className="text-gray-800 font-semibold">Модели</Label>
                <div className="flex flex-wrap gap-2">
                  {models.map((model) => (
                    <label key={model} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model)}
                        onChange={e => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, model]);
                          } else {
                            setSelectedModels(selectedModels.filter(m => m !== model));
                          }
                        }}
                      />
                      <span
                        style={{ fontWeight: mainModelLower && model.toLowerCase() === mainModelLower ? 'bold' : 'normal' }}
                      >{model}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

        {/* Charts and stats only after article and at least one model selected */}
        {!loading && selectedModels.length > 0 && selectedArticle && (
          <div className="w-full min-h-[1200px]">
            {chartLoading ? (
              <div className="text-center py-8 text-lg text-gray-500">Загрузка графиков...</div>
            ) : (
              <>
                {/* Таблица данных по выбранной статье */}
                <Card className="shadow-lg border-gray-200 bg-white mb-8">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-gray-900">Данные по статье: {selectedArticle}</CardTitle>
                        <CardDescription>Детальные данные с отклонениями и разницей для всех выбранных моделей</CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        onClick={() => setShowFullTable(!showFullTable)}
                        className="flex items-center gap-2"
                      >
                        {showFullTable ? 'Свернуть' : `Показать все (${chartData.length > 0 ? chartData[0].data.length : 0})`}
                        <motion.div
                          animate={{ rotate: showFullTable ? 180 : 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <TrendingUp className="w-4 h-4" />
                        </motion.div>
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex">
                      {/* Фиксированные колонки: Дата и Факт */}
                      <div className="flex-none">
                        <table className="border-collapse text-sm">
                          <thead>
                            <tr className="border-b border-gray-200 bg-gray-50">
                              <th className="text-left py-3 px-3 font-semibold text-gray-900 border-r border-gray-300 min-w-[100px]">Дата</th>
                              <th className="text-center py-3 px-3 font-semibold text-gray-900 border-r border-gray-400 min-w-[100px]">Факт</th>
                            </tr>
                          </thead>
                          <tbody>
                            {chartData.length > 0 && chartData[0].data
                              .slice(0, showFullTable ? undefined : 10)
                              .map((row: any, index: number) => (
                              <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                                <td className="py-3 px-3 font-medium text-gray-900 border-r border-gray-300">
                                  {typeof row.date === 'number' ? excelSerialToDate(row.date) : row.date}
                                </td>
                                <td className="py-3 px-3 text-center font-medium text-gray-700 border-r border-gray-400">
                                  {row.actual !== null && row.actual !== undefined ? Number(row.actual).toLocaleString() : '-'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      
                      {/* Прокручиваемые колонки: данные моделей */}
                      <div className="flex-1 overflow-x-auto">
                        <table className="border-collapse text-sm w-full">
                          <thead>
                            <tr className="border-b border-gray-200 bg-gray-50">
                              {selectedModels.map((model, modelIndex) => (
                                <React.Fragment key={model}>
                                  <th className={`text-center py-3 px-3 font-semibold text-gray-700 ${modelIndex < selectedModels.length - 1 ? 'border-r border-gray-300' : ''}`} style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '120px' }}>
                                    Прогноз ({model})
                                    {mainModelLower && model.toLowerCase() === mainModelLower && (
                                      <span className="ml-1 px-2 py-1 text-xs font-medium text-blue-600 bg-blue-100 rounded-full">
                                        целевая
                                      </span>
                                    )}
                                  </th>
                                  <th className="text-center py-3 px-3 font-semibold text-gray-700" style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '100px' }}>Отклонение %</th>
                                  <th className={`text-center py-3 px-3 font-semibold text-gray-700 ${modelIndex < selectedModels.length - 1 ? 'border-r-2 border-gray-500' : ''}`} style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '100px' }}>Разница</th>
                                </React.Fragment>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {chartData.length > 0 && chartData[0].data
                              .slice(0, showFullTable ? undefined : 10)
                              .map((row: any, index: number) => (
                              <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                                {selectedModels.map((model, modelIndex) => {
                                  const modelData = chartData.find(d => d.model === model);
                                  const modelRow = modelData?.data[index];
                                  const isMainModel = mainModelLower && model.toLowerCase() === mainModelLower;
                                  
                                  return (
                                    <React.Fragment key={model}>
                                      <td className={`py-3 px-3 text-center font-medium ${isMainModel ? 'text-blue-700 bg-blue-50' : 'text-gray-700'} ${modelIndex < selectedModels.length - 1 ? 'border-r border-gray-300' : ''}`}>
                                        {modelRow?.forecast !== null && modelRow?.forecast !== undefined ? Number(modelRow.forecast).toLocaleString() : '-'}
                                      </td>
                                      <td className={`py-3 px-3 text-center font-medium ${isMainModel ? 'text-blue-700 bg-blue-50' : 'text-gray-700'}`}>
                                        {modelRow?.errorPercent !== null && modelRow?.errorPercent !== undefined ? `${Number(modelRow.errorPercent).toFixed(2)}%` : '-'}
                                      </td>
                                      <td className={`py-3 px-3 text-center font-medium ${isMainModel ? 'text-blue-700 bg-blue-50' : 'text-gray-700'} ${modelIndex < selectedModels.length - 1 ? 'border-r-2 border-gray-500' : ''}`}>
                                        {modelRow?.difference !== null && modelRow?.difference !== undefined ? Number(modelRow.difference).toLocaleString() : '-'}
                                      </td>
                                    </React.Fragment>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    {!showFullTable && chartData.length > 0 && chartData[0].data.length > 10 && (
                      <div className="mt-4 text-center">
                        <p className="text-sm text-gray-500 mb-3">
                          Показано {Math.min(10, chartData[0].data.length)} из {chartData[0].data.length} записей
                        </p>
                        <Button
                          variant="ghost"
                          onClick={() => setShowFullTable(true)}
                          className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                        >
                          Показать все записи
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="shadow-lg border-gray-200 bg-white mb-8">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-gray-900">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                      {`Прогноз vs Факт`}
                    </CardTitle>
                    <CardDescription>
                      Сравнение прогнозируемых и фактических значений по датам для {selectedArticle} и всех выбранных моделей
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ForecastChart data={chartData} mainModelLower={mainModelLower} />
                  </CardContent>
                </Card>
                <Card className="shadow-lg border-gray-200 bg-white mb-8">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-gray-900">
                      <AlertTriangle className="w-5 h-5 text-amber-600" />
                      {`Ошибка %`}
                    </CardTitle>
                    <CardDescription>Ошибка прогноза в процентах по датам для {selectedArticle} и всех выбранных моделей</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ErrorChart data={chartData} mainModelLower={mainModelLower} />
                  </CardContent>
                </Card>
                <Card className="shadow-lg border-gray-200 bg-white mb-8">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-gray-900">
                      <BarChart3 className="w-5 h-5 text-purple-600" />
                      {`Разница`}
                    </CardTitle>
                    <CardDescription>Абсолютная разница между прогнозом и фактом по датам для {selectedArticle} и всех выбранных моделей</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <AbsoluteDiffChart data={chartData} mainModelLower={mainModelLower} />
                  </CardContent>
                </Card>
                {/* Statistics Summary for all selected models */}
                <Card className="shadow-lg border-gray-200 bg-white mb-8">
                  <CardHeader>
                    <CardTitle className="text-gray-900">Статистика точности</CardTitle>
                    <CardDescription>Основные метрики качества прогнозирования для {selectedArticle} по всем выбранным моделям</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left py-3 px-4 font-semibold text-gray-900">Модель</th>
                            <th className="text-center py-3 px-4 font-semibold text-gray-700">Средняя ошибка</th>
                            <th className="text-center py-3 px-4 font-semibold text-gray-700">Макс. ошибка</th>
                            <th className="text-center py-3 px-4 font-semibold text-gray-700">Точность</th>
                            <th className="text-center py-3 px-4 font-semibold text-gray-700">Мин. ошибка</th>
                          </tr>
                        </thead>
                        <tbody>
                          {chartData.map(({ model, data }) => (
                            <tr key={model} className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${mainModelLower && model.toLowerCase() === mainModelLower ? 'bg-blue-50' : ''}`}>
                              <td className="py-4 px-4">
                                <div className="flex items-center">
                                  <span className={`font-medium ${mainModelLower && model.toLowerCase() === mainModelLower ? 'text-blue-900' : 'text-gray-900'}`}>{model}</span>
                                  {mainModelLower && model.toLowerCase() === mainModelLower && (
                                    <span className="ml-2 px-2 py-1 text-xs font-medium text-blue-600 bg-blue-100 rounded-full">
                                      целевая
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td className="py-4 px-4 text-center">
                                <span className={`text-sm font-bold ${mainModelLower && model.toLowerCase() === mainModelLower ? 'text-blue-700' : 'text-gray-700'}`}>
                                  {data.length ? `${(data.reduce((acc: any, d: any) => acc + (d.errorPercent ?? d.error ?? 0), 0) / data.length).toFixed(2)}%` : '-'}
                                </span>
                              </td>
                              <td className="py-4 px-4 text-center">
                                <span className={`text-sm font-bold ${mainModelLower && model.toLowerCase() === mainModelLower ? 'text-blue-700' : 'text-gray-700'}`}>
                                  {data.length ? `${Math.max(...data.map((d: any) => Math.abs(d.errorPercent ?? d.error ?? 0))).toFixed(2)}%` : '-'}
                                </span>
                              </td>
                              <td className="py-4 px-4 text-center">
                                <span className={`text-sm font-bold ${mainModelLower && model.toLowerCase() === mainModelLower ? 'text-blue-700' : 'text-gray-700'}`}>
                                  {data.length ? `${(100 - data.reduce((acc: any, d: any) => acc + Math.abs(d.errorPercent ?? d.error ?? 0), 0) / data.length).toFixed(2)}%` : '-'}
                                </span>
                              </td>
                              <td className="py-4 px-4 text-center">
                                <span className={`text-sm font-bold ${mainModelLower && model.toLowerCase() === mainModelLower ? 'text-blue-700' : 'text-gray-700'}`}>
                                  {data.length ? `${Math.min(...data.map((d: any) => Math.abs(d.errorPercent ?? d.error ?? 0))).toFixed(2)}%` : '-'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
    </main>
  )
}
