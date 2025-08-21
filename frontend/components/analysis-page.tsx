"use client"
import { AnalysisGraphs } from "./analyze/graphs";
import { ArticleDataTable } from "./analyze/ArticleDataTable";
import { ArticleStatsTable } from "./ArticleStatsTable";
import React, { useState } from "react"
import { useExcelContext } from "../context/ExcelContext"
import { motion } from "framer-motion"
import { BarChart3, TrendingUp, AlertTriangle, Table } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, XAxis, YAxis, ResponsiveContainer, Legend } from "recharts"
import dynamic from "next/dynamic";
const BulletChartCard = dynamic(() => import("./BulletChartCard").then(mod => mod.BulletChartCard), { ssr: false });
import { fetchExcelAndParseModels } from "./utils/fetchExcelAndParseModels"
import { parseYamlConfig } from "./utils/parseYaml"
import { fetchExcelDataForChart } from "./utils/fetchExcelDataForChart"
import * as XLSX from "xlsx"
import { useEffect } from "react"
import { useConfig } from "../context/ConfigContext"
import { useAuth } from "../context/AuthContext"

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

function ForecastChart({ data, mainModelLower, forecastType = 'original' }: { data: any[], mainModelLower?: string, forecastType?: 'original' | 'corrected' }) {
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
  
  const forecastLabel = forecastType === 'corrected' ? 'Скорректированный прогноз' : 'Прогноз';
  
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
              name={`${forecastLabel} (${model})`}
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
  const { session } = useAuth();
  const accessToken = session?.access_token;
  const [chartData, setChartData] = useState<any[]>([])
  const [chartLoading, setChartLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showFullTable, setShowFullTable] = useState(false)
  const [forecastType, setForecastType] = useState<'original' | 'corrected'>('original')
  const [currency, setCurrency] = useState<'RUB' | 'USD'>('RUB')
  const [pipeline, setPipeline] = useState<'base' | 'base+'>('base')
  const {
    excelBuffer,
    setExcelBuffer,
    models,
    setModels,
    articles,
    setArticles,
    parsedJson,
    setParsedJson,
    finalPredictionJson,
    setFinalPredictionJson,
    exchangeRatesJson,
    setExchangeRatesJson,
    // Используем состояние из контекста
    selectedArticle,
    setSelectedArticle,
    selectedModels,
    setSelectedModels,
  } = useExcelContext()


  // Получаем главную модель для выбранной статьи и pipeline
  let mainModel = null;
  if (config && config.model_article && selectedArticle && pipeline) {
    const info = config.model_article[selectedArticle];
    if (info && typeof info === 'object' && info.pipeline === pipeline) {
      mainModel = info.model;
    }
  }
  const mainModelLower = mainModel ? mainModel.toLowerCase() : null;



  // Логгирование для отладки определения главной модели
  useEffect(() => {
  }, [selectedArticle, mainModel, config, models]);

  // При смене статьи выбираем только главную модель, но только если это реально смена статьи
  useEffect(() => {
    if (!selectedArticle || models.length === 0) return;
    
    // Если выбран скорректированный прогноз, принудительно выбираем только целевую модель
    if (forecastType === 'corrected') {
      if (mainModelLower) {
        const found = models.find((m: string) => m.toLowerCase() === mainModelLower);
        if (found) {
          setSelectedModels([found]);
          return;
        }
      }
      // Если целевая модель не найдена, но выбран скорректированный прогноз, выбираем первую модель
      if (models.length > 0) {
        setSelectedModels([models[0]]);
      }
      return;
    }
    
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
  }, [mainModelLower, models, selectedArticle, selectedModels.length, forecastType]);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
    // Если данные уже есть в контексте, не загружаем заново
  if (excelBuffer && models.length > 0 && articles.length > 0 && parsedJson.length > 0 && finalPredictionJson && finalPredictionJson.length > 0 && exchangeRatesJson && exchangeRatesJson.length > 0) {
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
        const response = await fetch(url, {
          headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
        });
        if (!response.ok) throw new Error('Ошибка скачивания файла');
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        setExcelBuffer(arrayBuffer);
        // Получаем модели через utils
        const filteredModels = await fetchExcelAndParseModels(arrayBuffer, accessToken);
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
  // читаем лист final_prediction для корректировок
  const finalSheet = workbook.Sheets["final_prediction"];
  const finalJson = finalSheet ? XLSX.utils.sheet_to_json(finalSheet) : [];
  setFinalPredictionJson(finalJson as any[]);
  // читаем лист с курсами валют
  const ratesSheet = workbook.Sheets["Курсы валют"];
  const ratesJson = ratesSheet ? XLSX.utils.sheet_to_json(ratesSheet) : [];
  setExchangeRatesJson(ratesJson as any[]);
      } catch (err) {
        setArticles([]);
        setModels([]);
  setParsedJson([]);
  setFinalPredictionJson([]);
  setExchangeRatesJson([]);
      }
      setLoading(false);
    }
    if (!configLoading) fetchData();
  }, [configLoading, config]);

  useEffect(() => {
    if (!selectedArticle || !selectedModels.length || !parsedJson || !parsedJson.length) return;
    setChartLoading(true);
    try {
      const usdArticles = config?.["Статьи для предикта в USD"] || [];
      const selectedArticleLower = selectedArticle.trim().toLowerCase();
      const isUsdArticle = usdArticles.map((a: string) => a.trim().toLowerCase()).includes(selectedArticleLower);

      const filtered = parsedJson.filter((row: any) => {
        const val = row["Статья"];
        if (!val) return false;
        const rowArticleName = val.trim().toLowerCase();
        const rowPipeline = (row["pipeline"] ?? '').toLowerCase();
        if (rowPipeline !== pipeline) return false;
        if (isUsdArticle) {
          return rowArticleName === selectedArticleLower || rowArticleName === selectedArticleLower + '_usd';
        } else {
          return rowArticleName === selectedArticleLower;
        }
      });

      const toDateKey = (d: any) => (typeof d === 'number' ? excelSerialToDate(d) : (typeof d === 'string' ? d.slice(0, 10) : d));
      const ratesIndex = new Map<string, number>();
      if (Array.isArray(exchangeRatesJson)) {
        for (const r of exchangeRatesJson as any[]) {
          const dk = toDateKey(r["Дата"] ?? r["date"] ?? r["Date"]);
          if (!dk) continue;
          const rateVal = Number(r["Курс"] ?? r["rate"] ?? r["Rate"] ?? r["exchange_rate"]);
          if (isFinite(rateVal) && rateVal > 0) ratesIndex.set(dk, rateVal);
        }
      }
      const ratesList = Array.from(ratesIndex.values());
      const avgRate = ratesList.length ? (ratesList.reduce((a, b) => a + b, 0) / ratesList.length) : 1;
      const resolveRate = (dk?: string | null) => {
        if (!dk) return avgRate;
        const r = ratesIndex.get(dk);
        if (!(r && isFinite(r) && r > 0)) return avgRate;
        return r;
      };
      const convert = (value: number | null | undefined, from: 'RUB' | 'USD', to: 'RUB' | 'USD', rate?: number) => {
        if (value === null || value === undefined) return value as any;
        if (from === to) return value;
        const r = rate && isFinite(rate) && rate > 0 ? rate : 1;
        return from === 'RUB' ? value / r : value * r;
      };
      const getRowBaseCurrency = (rowArticleName: string): 'RUB' | 'USD' => {
        const name = rowArticleName.trim().toLowerCase();
        if (name.endsWith('_usd')) return 'USD';
        return isUsdArticle ? 'USD' : 'RUB';
      };

      const allChartData = selectedModels.map(model => ({
        model,
        data: filtered.map((row: any) => {
          const articleStr = (row["Статья"] ?? '').toString();
          const baseCur = getRowBaseCurrency(articleStr);
          const dk = toDateKey(row["Дата"]);
          const rate = resolveRate(dk);

          // Берём значения как есть из Excel
          const fact = row["Fact"];
          const forecast = row[`predict_${model}`];
          const diff = row[`predict_${model} разница`];
          // Для поиска поля 'отклонение  %' с пробелами, как в Excel
          const errorKey = Object.keys(row).find(
            k => k.trim().toLowerCase() === `predict_${model}`.trim().toLowerCase() + ' отклонение %'
          );
          const error = errorKey ? row[errorKey] : undefined;

          // Корректировки: ищем по дате, статье и модели в final_prediction
          let adjustments = 0;
          if (Array.isArray(finalPredictionJson)) {
            const dateKey = (row["Дата"] ?? '').toString();
            const articleKey = (row["Статья"] ?? '').toString().trim().toLowerCase();
            const modelKey = model.toString().trim().toLowerCase();
            const found = finalPredictionJson.find((adjRow: any) => {
              const adjDate = (adjRow["Дата"] ?? adjRow["date"] ?? adjRow["Date"] ?? '').toString();
              const adjArticle = (adjRow["Статья"] ?? adjRow["article"] ?? '').toString().trim().toLowerCase();
              const adjModel = (adjRow["Модель"] ?? adjRow["model"] ?? '').toString().trim().toLowerCase();
              return adjDate === dateKey && adjArticle === articleKey && adjModel === modelKey;
            });
            if (found) {
              adjustments = found["Корректировка"] ?? found["adjustment"] ?? 0;
            } else {
              adjustments = row["Корректировка"] ?? row["adjustments"] ?? row[`adjustments_${model}`] ?? 0;
            }
          } else {
            adjustments = row["Корректировка"] ?? row["adjustments"] ?? row[`adjustments_${model}`] ?? 0;
          }

          // Конвертируем только числовые значения (не проценты)
          const factConv = convert(fact, baseCur, currency, rate);
          const forecastConv = convert(forecast, baseCur, currency, rate);
          const diffConv = convert(diff, baseCur, currency, rate);
          const adjustmentsConv = convert(adjustments, baseCur, currency, rate);

          return {
            date: row["Дата"],
            actual: factConv,
            forecast: forecastConv,
            difference: diffConv,
            errorPercent: error, // проценты не конвертируем!
            adjustments: adjustmentsConv,
          };
        })
      }));

      setChartData(allChartData);
    } catch (err) {
      setChartData([]);
    }
    setTimeout(() => setChartLoading(false), 300);
  }, [selectedArticle, selectedModels, parsedJson, currency, exchangeRatesJson, config, pipeline]);


  return (
    <main className="flex-1 p-8">
      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[60vh]">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 border-solid mb-6" />
          <div className="text-lg text-gray-500">Загрузка данных...</div>
        </div>
      ) : (
        <>
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Анализ прогнозов</h1>
            <p className="text-gray-600 text-lg">Анализ точности прогнозирования и сравнение с фактическими данными</p>
            {/* Здесь был Bullet график. Теперь он в разделе "Общая аналитика". */}
          </motion.div>

          {/* Карточка параметров анализа */}
          <Card className="shadow-lg border-gray-200 bg-white mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-gray-900">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                Параметры анализа
              </CardTitle>
              <CardDescription>Выберите статью ЧОК и модели для анализа.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-6">
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
                      <SelectValue placeholder="Выберите статью ЧОК" />
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
                
                <div className="grid grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="forecastType"
                        value="original"
                        checked={forecastType === 'original'}
                        onChange={(e) => {
                          setForecastType(e.target.value as 'original' | 'corrected');
                        }}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">Исходный прогноз</span>
                    </label>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="forecastType"
                        value="corrected"
                        checked={forecastType === 'corrected'}
                        onChange={(e) => {
                          setForecastType(e.target.value as 'original' | 'corrected');
                          if (e.target.value === 'corrected') {
                            // При переключении на скорректированный прогноз сбрасываем выбор моделей
                            // useEffect автоматически выберет целевую модель
                            setSelectedModels([]);
                          }
                        }}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">Скорректированный прогноз</span>
                    </label>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="currency"
                        value="RUB"
                        checked={currency === 'RUB'}
                        onChange={(e) => setCurrency(e.target.value as 'RUB' | 'USD')}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">RUB</span>
                    </label>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="currency"
                        value="USD"
                        checked={currency === 'USD'}
                        onChange={(e) => setCurrency(e.target.value as 'RUB' | 'USD')}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">USD</span>
                    </label>
                  </div>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="pipeline"
                        value="base"
                        checked={pipeline === 'base'}
                        onChange={(e) => setPipeline(e.target.value as 'base' | 'base+')}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">base</span>
                    </label>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="pipeline"
                        value="base+"
                        checked={pipeline === 'base+'}
                        onChange={(e) => setPipeline(e.target.value as 'base' | 'base+')}
                        className="w-4 h-4 accent-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">base+</span>
                    </label>
                  </div>
                </div>
              </div>
              
              <div className="space-y-3">
                <Label className="text-gray-800 font-semibold">Модели</Label>
                <div className="flex flex-wrap gap-2">
                  {models.map((model) => {
                    const isMainModel = mainModelLower && model.toLowerCase() === mainModelLower;
                    const isDisabled = forecastType === 'corrected' && !isMainModel;
                    return (
                      <label key={model} className={`flex items-center gap-2 ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
                        <input
                          type="checkbox"
                          checked={selectedModels.includes(model)}
                          disabled={isDisabled}
                          onChange={e => {
                            if (isDisabled) return;
                            if (e.target.checked) {
                              setSelectedModels([...selectedModels, model]);
                            } else {
                              setSelectedModels(selectedModels.filter(m => m !== model));
                            }
                          }}
                          className="accent-blue-600 focus:ring-blue-500"
                        />
                        <span
                          style={{ fontWeight: isMainModel ? 'bold' : 'normal' }}
                        >
                          {model}
                          {isMainModel && (
                            <span className="ml-1 text-xs text-blue-600">(целевая)</span>
                          )}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>
  </CardContent>
  </Card>

          {/* Charts and stats only after article and at least one model selected */}
          {selectedModels.length > 0 && selectedArticle && (
            <div className="w-full min-h-[1200px]">
              {chartLoading ? (
                <div className="text-center py-8 text-lg text-gray-500">Загрузка графиков...</div>
              ) : (
                <>
                  <ArticleDataTable
                    selectedArticle={selectedArticle}
                    showFullTable={showFullTable}
                    setShowFullTable={setShowFullTable}
                    chartData={chartData}
                    forecastType={forecastType}
                    selectedModels={selectedModels}
                    mainModelLower={mainModelLower}
                    excelSerialToDate={excelSerialToDate}
                  />

                  {/* Агрегированная информация по статье */}
                  <Card className="shadow-lg border-gray-200 bg-white mb-8">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="flex items-center gap-2 text-gray-900">
                            <Table className="w-5 h-5 text-green-600" />
                            Агрегированная информация
                          </CardTitle>
                          <CardDescription>
                            Статистические метрики по выбранной статье за все годы и периоды
                          </CardDescription>
                        </div>
                        <Button
                          variant="outline"
                          onClick={async () => {
                            if (!selectedArticle) return;
                            const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
                            const url = backendUrl.replace(/\/$/, '') + `/download_article_excel/?article=${encodeURIComponent(selectedArticle)}&pipeline=${encodeURIComponent(pipeline)}`;
                            try {
                              const response = await fetch(url, {
                                headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
                              });
                              if (!response.ok) throw new Error('Ошибка скачивания файла');
                              const blob = await response.blob();
                              const a = document.createElement('a');
                              const currentDate = new Date().toISOString().split('T')[0];
                              a.href = window.URL.createObjectURL(blob);
                              a.download = `article_${selectedArticle}_${pipeline}_${currentDate}.xlsx`;
                              a.click();
                              window.URL.revokeObjectURL(a.href);
                            } catch (err) {
                              alert('Ошибка скачивания файла');
                            }
                          }}
                          className="flex items-center gap-2"
                        >
                          Скачать Excel
                        </Button>
                      </div>
                    </CardHeader>
                    {/* Таблица статистики */}
                    {selectedArticle && (
                      <ArticleStatsTable
                        article={selectedArticle}
                        currency={currency}
                        exchangeRatesJson={exchangeRatesJson}
                        parsedJson={parsedJson}
                        config={config}
                        pipeline={pipeline}
                      />
                    )}
                  </Card>

                  <AnalysisGraphs
                    chartData={chartData}
                    mainModelLower={mainModelLower}
                    forecastType={forecastType}
                    selectedArticle={selectedArticle}
                    ForecastChart={ForecastChart}
                    ErrorChart={ErrorChart}
                    AbsoluteDiffChart={AbsoluteDiffChart}
                  />
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
        </>
      )}
    </main>
  )
}
