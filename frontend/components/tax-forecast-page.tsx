"use client"

import React, { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  FileText,
  Calendar,
  Check,
  ChevronsUpDown,
  X,
  TrendingUp,
  Play,
  Square,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { cn } from "@/lib/utils"
import { Separator } from "@/components/ui/separator"

import { useConfig } from "../context/ConfigContext"
import { useAuth } from "../context/AuthContext"

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
  files,
  onFilesChange,
  multiple = false,
}: {
  label: string
  accept: string
  files: File[]
  onFilesChange: (files: File[]) => void
  multiple?: boolean
}) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      if (multiple) {
        onFilesChange([...files, ...newFiles])
      } else {
        onFilesChange(newFiles)
      }
    }
    // Reset input value to allow selecting the same file again
    if (inputRef.current) {
      inputRef.current.value = ""
    }
  }

  const removeFile = (index: number) => {
    const newFiles = [...files]
    newFiles.splice(index, 1)
    onFilesChange(newFiles)
  }

  return (
    <div className="space-y-3">
      <Label className="text-gray-800 font-semibold flex items-center gap-2">
        <FileText className="w-4 h-4 text-gray-600" />
        {label}
      </Label>

      <div className="space-y-3">
        <input
          type="file"
          ref={inputRef}
          className="hidden"
          accept={accept}
          multiple={multiple}
          onChange={handleFileChange}
        />
        <Button
          type="button"
          onClick={() => inputRef.current?.click()}
          variant="outline"
          className="w-full bg-gray-50 border-gray-300 text-gray-700 hover:bg-gray-100 border-dashed border-2 h-12"
        >
          <Upload className="w-4 h-4 mr-2" />
          {files.length > 0 && !multiple ? "Выбрать другой файл" : "Выберите файл"}
        </Button>

        <AnimatePresence>
          {files.length > 0 && (
            <div className="space-y-2">
              {files.map((file, index) => (
                <motion.div
                  key={`${file.name}-${index}`}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="flex items-center justify-between p-3 bg-amber-50 border border-amber-200 rounded-lg"
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <FileText className="w-4 h-4 text-amber-600 flex-shrink-0" />
                    <span className="text-sm font-medium text-amber-900 truncate">{file.name}</span>
                    <Badge variant="secondary" className="text-xs bg-amber-100 text-amber-800 flex-shrink-0">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </Badge>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    className="text-amber-700 hover:text-amber-900 hover:bg-amber-100 h-8 w-8 p-0"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </motion.div>
              ))}
            </div>
          )}
        </AnimatePresence>
        {files.length > 0 && multiple && (
             <div className="text-sm text-gray-500">Загружено файлов: {files.length}</div>
        )}
      </div>
    </div>
  )
}

function StatusIndicator({ status, onClearStatus }: { status: any; onClearStatus?: () => void }) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case "running":
      case "pending":
        return {
          color: "blue",
          icon: null,
          text: "Прогноз выполняется...",
          bgClass: "bg-blue-50 border-blue-200 text-blue-800",
          iconClass: "",
        }
      case "done":
        return {
          color: "green",
          icon: CheckCircle,
          text: "Прогноз готов!",
          subtext: "Результаты сохранены и доступны для скачивания",
          bgClass: "bg-emerald-50 border-emerald-200 text-emerald-800",
          iconClass: "text-emerald-600",
        }
      case "error":
        return {
          color: "red",
          icon: AlertCircle,
          text: "Ошибка при прогнозировании",
          bgClass: "bg-red-50 border-red-200 text-red-800",
          iconClass: "text-red-600",
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

  const config = getStatusConfig(status?.state || "idle")
  const Icon = config.icon

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
            {status?.message && (
                <p className="text-xs mt-1 opacity-90">{status.message}</p>
            )}
          </div>
        </div>
        {status?.state === "done" && (
          <Button
            size="sm"
            onClick={onClearStatus}
            variant="ghost"
            className="ml-auto text-emerald-700 hover:text-emerald-900 hover:bg-emerald-100 px-3"
          >
            Скрыть
          </Button>
        )}
      </div>

      {/* Детальный прогресс (показываем только во время обучения) */}
      {(status?.state === "running" || status?.state === "pending") && (
        <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
            <div className="flex items-center gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <Loader2 className="w-4 h-4 text-white animate-spin" />
                    </div>
                </div>
                <div className="flex-grow min-w-0">
                    <p className="text-sm font-medium text-blue-900">Выполняется:</p>
                    <p className="text-sm text-blue-700 truncate">
                    {status?.currentTask || "Инициализация..."}
                    </p>
                </div>
            </div>
        </div>
      )}
    </motion.div>
  )
}

export function TaxForecastPage() {
  const { session } = useAuth()
  const { config, loading: configLoading } = useConfig()
  const [historyFile, setHistoryFile] = useState<File[]>([])
  const [forecastDate, setForecastDate] = useState<string>(new Date().toISOString().split('T')[0])
  const [selectedPairs, setSelectedPairs] = useState<string[]>([])
  const [status, setStatus] = useState<any>({ state: "idle" })
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Flatten options for multiselect
  const selectionOptions: string[] = []
  if (config && config['Налоги']) {
      Object.entries(config['Налоги']).forEach(([group, companies]: [string, any]) => {
          if (Array.isArray(companies)) {
              companies.forEach((company: string) => {
                  selectionOptions.push(`${group} | ${company}`)
              })
          }
      })
  }

  // Initialize with all selected by default as per streamlit code
  // default=selection_options
  React.useEffect(() => {
      if (selectionOptions.length > 0 && selectedPairs.length === 0) {
          setSelectedPairs(selectionOptions)
      }
  }, [config])

  // Cleanup on unmount
  React.useEffect(() => {
      return () => {
          if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
          }
      }
  }, [])

  // Restore active task on load
  React.useEffect(() => {
      // If we already have a task ID, don't check again to avoid overwriting status
      if (currentTaskId) return

      const checkActiveTask = async () => {
          if (!session?.access_token) return

          try {
              const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/taxes/active-task`, {
                  headers: {
                      "Authorization": `Bearer ${session.access_token}`
                  }
              })
              
              if (response.ok) {
                  const data = await response.json()
                  if (data.task_id) {
                      setCurrentTaskId(data.task_id)
                      setStatus({ state: "running", currentTask: "Восстановление сессии..." })
                      pollStatus(data.task_id)
                  }
              }
          } catch (error) {
              console.error("Failed to check active task:", error)
          }
      }

      checkActiveTask()
  }, [session, currentTaskId])

  const toggleSelection = (option: string) => {
      if (selectedPairs.includes(option)) {
          setSelectedPairs(selectedPairs.filter(item => item !== option))
      } else {
          setSelectedPairs([...selectedPairs, option])
      }
  }

  const handleAddItem = (value: string) => {
    if (value && !selectedPairs.includes(value)) {
      setSelectedPairs([...selectedPairs, value])
    }
  }

  const handleRemove = (item: string) => {
    setSelectedPairs(selectedPairs.filter((i) => i !== item))
  }

  const handleStartForecast = async () => {
      if (historyFile.length === 0) {
          setStatus({ state: "error", message: "Пожалуйста, загрузите файл с историческими данными." })
          return
      }
      
      setStatus({ state: "running", currentTask: "Запуск прогноза..." })
      
      try {
          const formData = new FormData()
          formData.append("history_file", historyFile[0])
          formData.append("forecast_date", forecastDate)
          formData.append("selected_groups", JSON.stringify(selectedPairs))
          
          const token = session?.access_token
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/taxes/forecast`, {
              method: "POST",
              headers: {
                  "Authorization": `Bearer ${token}`
              },
              body: formData
          })
          
          if (!response.ok) {
              const errorData = await response.json()
              throw new Error(errorData.detail || "Failed to start forecast")
          }
          
          const data = await response.json()
          const taskId = data.task_id
          setCurrentTaskId(taskId)
          
          pollStatus(taskId)
          
      } catch (error: any) {
          console.error(error)
          setStatus({ state: "error", message: error.message || "Ошибка при запуске прогноза" })
      }
  }

  const pollStatus = async (taskId: string) => {
      const token = session?.access_token
      
      // Clear any existing interval
      if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
      }

      pollingIntervalRef.current = setInterval(async () => {
          try {
              const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/taxes/status/${taskId}`, {
                  headers: {
                      "Authorization": `Bearer ${token}`
                  }
              })
              
              if (!response.ok) return
              
              const data = await response.json()
              
              if (data.status === "SUCCESS") {
                  if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
                  setStatus({ state: "done", message: "Прогноз успешно завершен!" })
                  setCurrentTaskId(null)
                  // Trigger download
                  window.location.href = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/taxes/download/${taskId}`
              } else if (data.status === "FAILURE") {
                  if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
                  setStatus({ state: "error", message: `Ошибка: ${data.error}` })
                  setCurrentTaskId(null)
              } else if (data.status === "REVOKED") {
                  if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
                  setStatus({ state: "idle", message: "Прогноз остановлен" })
                  setCurrentTaskId(null)
              } else if (data.status === "PROGRESS") {
                  if (data.meta && data.meta.status) {
                      setStatus({ state: "running", currentTask: data.meta.status })
                  }
              }
          } catch (e) {
              console.error(e)
          }
      }, 2000)
  }

  const handleStopForecast = async () => {
      if (!currentTaskId) return

      // Clear polling interval immediately
      if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
      }

      try {
          const token = session?.access_token
          await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/taxes/stop/${currentTaskId}`, {
              method: "POST",
              headers: {
                  "Authorization": `Bearer ${token}`
              }
          })
          setStatus({ state: "idle", message: "Прогноз остановлен" })
          setCurrentTaskId(null)
      } catch (error) {
          console.error("Failed to stop forecast:", error)
      }
  }

  const handleClearStatus = () => {
      setStatus({ state: "idle" })
  }

  const availableItems = selectionOptions.filter((item) => !selectedPairs.includes(item))

  return (
    <div className="h-full flex flex-col bg-gray-50/50 overflow-hidden">
      <div className="flex-none p-8 pb-4">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-2">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Прогноз налогов</h1>
          <p className="text-gray-600 text-lg">Настройка параметров и запуск прогнозирования налогов</p>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto p-8 pt-0">
        <div className="max-w-5xl mx-auto space-y-8 pb-12">
            
            <Card className="border-none shadow-sm bg-white/80 backdrop-blur">
                <CardHeader>
                    <CardTitle className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                        <Check className="w-5 h-5 text-blue-600" />
                        Выбор компаний для прогноза
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <Label className="text-gray-800 font-semibold block">Выберите группы и компании</Label>
                            {selectedPairs.length > 0 && (
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setSelectedPairs([])}
                                    className="text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                                >
                                    <X className="w-4 h-4 mr-1" />
                                    Очистить всё
                                </Button>
                            )}
                        </div>
                        
                        <div className="flex flex-wrap gap-2 min-h-[40px] p-3 border border-gray-200 rounded-lg bg-gray-50">
                            <AnimatePresence>
                                {selectedPairs.map((item) => (
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
                                <SelectValue placeholder="Добавить пару (Налог | Компания)..." />
                            </SelectTrigger>
                            <SelectContent>
                                {availableItems.map((key) => (
                                    <SelectItem key={key} value={key}>
                                        {key}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <p className="text-sm text-gray-500">Выберите одну или несколько пар (Налог | Компания) для расчета</p>
                    </div>
                </CardContent>
            </Card>

            <Separator />

            <Card className="border-none shadow-sm bg-white/80 backdrop-blur">
                <CardHeader>
                    <CardTitle className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                        <Calendar className="w-5 h-5 text-blue-600" />
                        Параметры прогноза
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="max-w-md">
                        <Label className="text-gray-800 font-semibold mb-2 block">Выберите дату (месяц) для начала прогноза</Label>
                        <MonthYearPicker value={forecastDate} onChange={setForecastDate} />
                    </div>
                </CardContent>
            </Card>

            <Card className="border-none shadow-sm bg-white/80 backdrop-blur">
                <CardHeader>
                    <CardTitle className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                        <FileText className="w-5 h-5 text-blue-600" />
                        Исторические данные
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div>
                        <FileInput
                            label="Загрузите файл с историческими данными"
                            accept=".xlsx,.xls"
                            files={historyFile}
                            onFilesChange={setHistoryFile}
                            multiple={false}
                        />
                    </div>
                </CardContent>
            </Card>

            <Separator />

            <div className="space-y-6">
              <Button
                size="lg"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-[26px] text-lg font-semibold shadow-lg"
                onClick={handleStartForecast}
                disabled={status.state === "running" || historyFile.length === 0}
              >
                <Play className="w-5 h-5 mr-2" />
                Начать прогноз
              </Button>
            </div>

            <Card className="shadow-xl border-gray-200 bg-white mt-6">
                <CardContent className="p-6">
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                            <TrendingUp className="w-5 h-5 text-blue-600" />
                            <h3 className="text-lg font-semibold text-gray-800">Статус прогноза</h3>
                        </div>

                        <StatusIndicator status={status} onClearStatus={handleClearStatus} />
                        
                        <AnimatePresence>
                            {status.state === "running" && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                >
                                    <Button
                                        type="button"
                                        onClick={handleStopForecast}
                                        variant="destructive"
                                        className="w-full bg-red-600 hover:bg-red-700 py-[26px] flex items-center justify-center text-lg font-semibold"
                                    >
                                        <Square className="w-4 h-4 mr-2" />
                                        Остановить прогноз
                                    </Button>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </CardContent>
            </Card>

        </div>
      </div>
    </div>
  )
}
