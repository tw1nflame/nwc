"use client"

import React, { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  X,
  Play,
  Square,
  Download,
  Settings,
  FileText,
  Calendar,
  Database,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2,
  TrendingUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Separator } from "@/components/ui/separator"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

import { parseYamlConfig } from "./utils/parseYaml"
import { useConfig } from "../context/ConfigContext"
import { sendTrainRequest } from "./utils/api"
import { fetchTrainStatus } from "./utils/trainStatus"
import { stopTrainTask } from "./utils/stopTrain"
import { useAuth } from "../context/AuthContext"
import { clearTrainStatus } from "./utils/clearStatus"



function MonthYearPicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedYear, setSelectedYear] = useState(new Date(value).getFullYear())
  const [selectedMonth, setSelectedMonth] = useState(new Date(value).getMonth())

  const months = [
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
  ]

  const currentYear = new Date().getFullYear()
  const years = Array.from({ length: 10 }, (_, i) => currentYear + i - 2)

  const handleApply = () => {
    const month = (selectedMonth + 1).toString().padStart(2, "0")
    onChange(`${selectedYear}-${month}-01`)
    setIsOpen(false)
  }

  const formatDisplayValue = (dateString: string) => {
    const date = new Date(dateString)
    return `${months[date.getMonth()]} ${date.getFullYear()}`
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="w-full justify-start text-left font-normal border-gray-300 focus:border-blue-500 bg-transparent"
        >
          <Calendar className="mr-2 h-4 w-4 text-gray-500" />
          {formatDisplayValue(value)}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-0" align="start">
        <div className="p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Выберите месяц и год</h4>
          </div>

          <div className="space-y-4">
            <div>
              <Label className="text-sm font-medium text-gray-700 mb-2 block">Год</Label>
              <Select
                value={selectedYear.toString()}
                onValueChange={(value) => setSelectedYear(Number.parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {years.map((year) => (
                    <SelectItem key={year} value={year.toString()}>
                      {year}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-sm font-medium text-gray-700 mb-2 block">Месяц</Label>
              <div className="grid grid-cols-3 gap-2">
                {months.map((month, index) => (
                  <Button
                    key={month}
                    variant={selectedMonth === index ? "default" : "outline"}
                    size="sm"
                    className={`text-xs ${
                      selectedMonth === index ? "bg-blue-600 hover:bg-blue-700 text-white" : "hover:bg-gray-100"
                    }`}
                    onClick={() => setSelectedMonth(index)}
                  >
                    {month.slice(0, 3)}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex gap-2 pt-2 border-t">
            <Button variant="outline" size="sm" onClick={() => setIsOpen(false)} className="flex-1">
              Отмена
            </Button>
            <Button size="sm" onClick={handleApply} className="flex-1 bg-blue-600 hover:bg-blue-700">
              Применить
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function FileInput({
  label,
  accept,
  file,
  onFileChange,
  onRemove,
  icon: Icon = FileText,
}: {
  label: string
  accept: string
  file: File | null
  onFileChange: (file: File | null) => void
  onRemove: () => void
  icon?: React.ComponentType<{ className?: string }>
}) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [inputKey, setInputKey] = useState(0)

  const handleRemove = () => {
    setInputKey((k) => k + 1)
    onRemove()
  }

  return (
    <div className="space-y-3">
      <Label className="text-gray-800 font-semibold flex items-center gap-2">
        <Icon className="w-4 h-4 text-gray-600" />
        {label}
      </Label>

      <div className="space-y-3">
        <Button
          type="button"
          onClick={() => inputRef.current?.click()}
          variant="outline"
          className="w-full bg-gray-50 border-gray-300 text-gray-700 hover:bg-gray-100 border-dashed border-2 h-12"
        >
          <Upload className="w-4 h-4 mr-2" />
          {file ? "Выбрать другой файл" : "Выберите файл"}
        </Button>

        <AnimatePresence>
          {file && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex items-center justify-between p-3 bg-amber-50 border border-amber-200 rounded-lg"
            >
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-amber-600" />
                <span className="text-sm font-medium text-amber-900">{file.name}</span>
                <Badge variant="secondary" className="text-xs bg-amber-100 text-amber-800">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </Badge>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleRemove}
                className="text-red-500 hover:text-red-700 hover:bg-red-50"
              >
                <X className="w-4 h-4" />
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <input
        key={inputKey}
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFileChange(e.target.files?.[0] || null)}
      />
    </div>
  )
}

function StatusIndicator({ trainStatus, onClearStatus, accessToken }: { trainStatus: any; onClearStatus?: () => void; accessToken?: string }) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case "running":
      case "pending":
        return {
          color: "blue",
          icon: null,
          text: "Обучение запущено...",
          bgClass: "bg-blue-50 border-blue-200 text-blue-800",
          iconClass: "",
        }
      case "done":
        return {
          color: "green",
          icon: CheckCircle,
          text: "Прогноз готов. Данные сохранены в БД и готовы к выгрузке!",
          subtext: "Статус автоматически очистится при запуске нового обучения",
          bgClass: "bg-emerald-50 border-emerald-200 text-emerald-800",
          iconClass: "text-emerald-600",
        }
      case "error":
        return {
          color: "red",
          icon: AlertCircle,
          text: "Ошибка при обучении",
          bgClass: "bg-red-50 border-red-200 text-red-800",
          iconClass: "text-red-600",
        }
      case "revoked":
        return {
          color: "yellow",
          icon: AlertCircle,
          text: "Обучение было остановлено",
          bgClass: "bg-amber-50 border-amber-200 text-amber-800",
          iconClass: "text-amber-600",
        }
      default:
        return {
          color: "gray",
          icon: Clock,
          text: "Ожидание запуска",
          bgClass: "bg-slate-50 border-slate-200 text-slate-700",
          iconClass: "text-slate-500",
        }
    }
  }

  const config = getStatusConfig(trainStatus?.status || "idle")
  const Icon = config.icon

  const handleClearStatus = async () => {
    if (!accessToken) return;
    await clearTrainStatus(accessToken)
    onClearStatus && onClearStatus()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Основной статус */}
      <div className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${config.bgClass}`}>
        <div className="flex items-center gap-3 flex-1">
          {Icon && <Icon className={`w-5 h-5 ${config.iconClass}`} />}
          <div className="flex-1">
            <span className="font-medium">{config.text}</span>
            {config.subtext && (
              <p className="text-xs mt-1 opacity-75">{config.subtext}</p>
            )}
            {/* Показываем детали завершенного обучения */}
            {trainStatus?.status === "done" && (trainStatus?.pipeline || trainStatus?.date) && (
              <div className="text-xs mt-2 space-y-0.5 opacity-90">
                {trainStatus?.pipeline && <p>Алгоритм: {trainStatus.pipeline}</p>}
                {trainStatus?.date && <p>Дата прогноза: {new Date(trainStatus.date).toLocaleDateString()}</p>}
                {trainStatus?.completed_at && <p>Завершено: {new Date(trainStatus.completed_at).toLocaleString()}</p>}
              </div>
            )}
          </div>
        </div>
        {trainStatus?.status === "done" && (
          <Button
            size="sm"
            onClick={handleClearStatus}
            variant="ghost"
            className="ml-auto text-emerald-700 hover:text-emerald-900 hover:bg-emerald-100 px-3"
          >
            Скрыть
          </Button>
        )}
      </div>

      {/* Детальный прогресс (показываем только во время обучения) */}
      {(trainStatus?.status === "running" || trainStatus?.status === "pending") && (
        <DetailedProgressIndicator trainStatus={trainStatus} />
      )}
    </motion.div>
  )
}

function DetailedProgressIndicator({ trainStatus }: { trainStatus: any }) {
  const {
    current_article = "",
    total_articles = 0,
    processed_articles = 0,
    percentage = 0
  } = trainStatus || {}

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      className="bg-white border border-gray-200 rounded-lg p-4 space-y-4"
    >
      {/* Прогресс-бар */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-gray-700">Общий прогресс</span>
          <span className="text-blue-600 font-semibold">{percentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <motion.div
            className="bg-blue-600 h-2.5 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>Обработано: {processed_articles} из {total_articles}</span>
        </div>
      </div>

      {/* Текущая статья */}
      {current_article && (
        <div className="flex items-center gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex-shrink-0">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            </div>
          </div>
          <div className="flex-grow min-w-0">
            <p className="text-sm font-medium text-blue-900">Обрабатывается статья:</p>
            <p className="text-sm text-blue-700 truncate" title={current_article}>
              {current_article}
            </p>
          </div>
        </div>
      )}
    </motion.div>
  )
}

function PredictForm({
  config,
  onSubmit,
  trainStatus,
  onStop,
  onClearStatus,
  accessToken,
}: {
  config: any
  onSubmit?: (data: any) => void
  trainStatus: any
  onStop?: () => void
  onClearStatus?: () => void
  accessToken?: string
}) {
  const [pipeline, setPipeline] = useState("BASE")
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [date, setDate] = useState("2025-01-01")
  const [dataFile, setDataFile] = useState<File | null>(null)

  useEffect(() => {
    if (config && config["Статья"]) {
      setSelectedItems(Object.keys(config["Статья"]))
    }
  }, [config])

  const allItems = config && config["Статья"] ? Object.keys(config["Статья"]) : []
  const availableItems = allItems.filter((item) => !selectedItems.includes(item))

  const handleAddItem = (value: string) => {
    if (value && !selectedItems.includes(value)) {
      setSelectedItems([...selectedItems, value])
    }
  }

  const handleRemove = (item: string) => {
    setSelectedItems(selectedItems.filter((i) => i !== item))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!dataFile) return
    onSubmit && onSubmit({ pipeline, selectedItems, date, dataFile })
  }


  const handleStop = async () => {
    await stopTrainTask(accessToken)
    onStop && onStop()
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full mx-auto">
      <Card className="shadow-xl border-gray-200 bg-white">
        <CardContent className="p-8">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Algorithm Selection */}
            <div className="space-y-4">
              <Label className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <Settings className="w-5 h-5 text-gray-600" />
                Алгоритм прогноза
              </Label>
              <RadioGroup value={pipeline} onValueChange={setPipeline} className="flex gap-8">
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="BASE" id="base" className="border-gray-400 text-blue-600" />
                  <Label htmlFor="base" className="font-medium text-gray-700">
                    BASE
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="BASE+" id="base-plus" className="border-gray-400 text-blue-600" />
                  <Label htmlFor="base-plus" className="font-medium text-gray-700">
                    BASE+
                  </Label>
                </div>
              </RadioGroup>
            </div>

            <Separator className="bg-gray-200" />

            {/* Articles Selection */}
            <div className="space-y-4">
              <Label className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <FileText className="w-5 h-5 text-gray-600" />
                Статьи для прогноза
              </Label>

              <div className="flex flex-wrap gap-2 min-h-[40px] p-3 border border-gray-200 rounded-lg bg-gray-50">
                <AnimatePresence>
                  {selectedItems.map((item) => (
                    <motion.div
                      key={item}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                    >
                      <Badge
                        variant="secondary"
                        className="bg-indigo-100 text-indigo-800 hover:bg-indigo-200 flex items-center gap-1 px-3 py-1"
                      >
                        {item}
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-4 w-4 p-0 hover:bg-red-100 text-indigo-600 hover:text-red-600"
                          onClick={() => handleRemove(item)}
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      </Badge>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>

              <Select onValueChange={handleAddItem} value="">
                <SelectTrigger className="border-gray-300 focus:border-blue-500 focus:ring-blue-500">
                  <SelectValue placeholder="Добавить статью..." />
                </SelectTrigger>
                <SelectContent>
                  {availableItems.map((key) => (
                    <SelectItem key={key} value={key}>
                      {key}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Separator className="bg-gray-200" />

            {/* Date Selection */}
            <div className="space-y-4">
              <Label className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-gray-600" />
                Месяц и год предикта
              </Label>
              <div className="max-w-xs">
                <MonthYearPicker value={date} onChange={setDate} />
              </div>
            </div>

            <Separator className="bg-gray-200" />

            {/* File Upload */}
            <div className="max-w-md">
              <FileInput
                label="Файл с данными (ЧОК исторические)"
                accept=".xlsm,.xlsx"
                file={dataFile}
                onFileChange={setDataFile}
                onRemove={() => setDataFile(null)}
                icon={Database}
              />
            </div>

            <Separator className="bg-gray-200" />

            {/* Submit Button, Status, Stop, Download Excel */}
            <div className="space-y-6">
              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-[26px] text-lg font-semibold shadow-lg"
                disabled={!dataFile || trainStatus?.status === "running"}
              >
                <Play className="w-5 h-5 mr-2" />
                Запустить расчёт
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Отдельная карточка для статуса и управления */}
      <Card className="shadow-xl border-gray-200 bg-white mt-6">
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              <h3 className="text-lg font-semibold text-gray-800">Статус прогноза</h3>
            </div>

            {/* Status Display */}
            <StatusIndicator trainStatus={trainStatus} onClearStatus={onClearStatus} accessToken={accessToken} />

            {/* Stop Button */}
            <AnimatePresence>
              {trainStatus?.status === "running" && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <Button
                    type="button"
                    onClick={handleStop}
                    variant="destructive"
                    className="w-full bg-red-600 hover:bg-red-700 py-[26px] flex items-center justify-center text-lg font-semibold"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Остановить обучение
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default function TrainingPage() {
  const { session } = useAuth();
  const accessToken = session?.access_token;
  // Логируем accessToken для отладки
  const { config, loading: configLoading } = useConfig();
  const [trainStatus, setTrainStatus] = useState<any>({ status: "idle" })
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    let ignore = false
    async function checkStatus() {
      const status = await fetchTrainStatus(accessToken)
      // Логируем ответ от train_status
      if (!ignore) {
        setTrainStatus(status && status.status ? status : { status: "idle" })
        // Запускаем polling если обучение активно (running или pending)
        if (status.status === "running" || status.status === "pending") {
          setPolling(true)
        } else {
          setPolling(false)
        }
      }
    }
    checkStatus()
    return () => {
      ignore = true
    }
  }, [accessToken])

  useEffect(() => {
    if (!polling) return
    const interval = setInterval(async () => {
      const status = await fetchTrainStatus(accessToken)
      // Логируем ответ от train_status при polling
      setTrainStatus(status && status.status ? status : { status: "idle" })
      // Останавливаем polling если обучение не активно
      if (status.status !== "running" && status.status !== "pending") {
        setPolling(false)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [polling, accessToken])

  const handleTrainSubmit = async ({ pipeline, selectedItems, date, dataFile }: any) => {
    await sendTrainRequest({ pipeline, selectedItems, date, dataFile, accessToken })
    setPolling(true)
  }

  const handleStop = async () => {
    // После остановки задачи через API, обновляем статус через polling
    // API сохранит глобальный статус "revoked", который получим при следующем запросе
    setPolling(false)
    // Немедленно обновляем статус
    const status = await fetchTrainStatus(accessToken)
    setTrainStatus(status && status.status ? status : { status: "idle" })
  }

  const handleClearStatus = async () => {
    // Вызываем API для очистки глобального статуса
    await clearTrainStatus(accessToken)
    // Обновляем локальный статус
    setTrainStatus({ status: "idle" })
    setPolling(false)
  }

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Обучение модели</h1>
        <p className="text-gray-600 text-lg">Настройка и запуск процесса обучения модели прогнозирования</p>
      </motion.div>

  {config && <PredictForm config={config} onSubmit={handleTrainSubmit} trainStatus={trainStatus} onStop={handleStop} onClearStatus={handleClearStatus} accessToken={accessToken} />}
    </main>
  )
}
