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
  ChevronDown,
  ChevronUp,
  FileText,
  Calendar,
  Database,
  Activity,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { parseYamlConfig } from "./utils/parseYaml"
import { sendTrainRequest } from "./utils/api"
import { downloadExcel } from "./utils/downloadExcel"
import { fetchTrainStatus } from "./utils/trainStatus"
import { stopTrainTask } from "./utils/stopTrain"

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

function Sidebar({ onConfigLoad, onLogDownload }: { onConfigLoad?: () => void; onLogDownload?: () => void }) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [isAdmin, setIsAdmin] = useState(false)
  const [open, setOpen] = useState(false)

  const handleLogin = () => {
    if (username === "admin" && password === "admin") {
      setIsAdmin(true)
      onConfigLoad && onConfigLoad()
    } else {
      setIsAdmin(false)
    }
  }

  return (
    <motion.aside
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      className="w-80 bg-white border-r border-gray-200 min-h-screen flex flex-col shadow-lg"
    >
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Прогнозирование</h2>
            <p className="text-sm text-gray-500">Система анализа</p>
          </div>
        </div>

        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700 font-semibold"
            >
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Технический доступ
              </div>
              {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>

          <CollapsibleContent className="space-y-4 mt-4">
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
              <div>
                <Label htmlFor="username" className="text-gray-700 font-medium">
                  Имя пользователя
                </Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <div>
                <Label htmlFor="password" className="text-gray-700 font-medium">
                  Пароль
                </Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <Button onClick={handleLogin} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold">
                Войти
              </Button>

              <AnimatePresence>
                {isAdmin && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2 pt-4 border-t border-gray-200"
                  >
                    <Button
                      onClick={onConfigLoad}
                      variant="outline"
                      className="w-full bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100"
                    >
                      <Database className="w-4 h-4 mr-2" />
                      Загрузить конфиг
                    </Button>
                    <Button
                      onClick={onLogDownload}
                      variant="outline"
                      className="w-full bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Скачать логи
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </motion.aside>
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
  // Добавим state для key, чтобы сбрасывать input
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

function StatusIndicator({ trainStatus }: { trainStatus: any }) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case "running":
      case "pending":
        return {
          color: "blue",
          icon: Loader2,
          text: "Обучение запущено...",
          bgClass: "bg-blue-50 border-blue-200 text-blue-800",
          iconClass: "text-blue-600 animate-spin",
        }
      case "done":
        return {
          color: "green",
          icon: CheckCircle,
          text: "Обучение завершено!",
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${config.bgClass} font-medium`}
    >
      <Icon className={`w-5 h-5 ${config.iconClass}`} />
      <span>{config.text}</span>
      {trainStatus?.result_file && trainStatus?.status === "done" && (
        <Badge variant="outline" className="ml-auto bg-emerald-100 text-emerald-800 border-emerald-300">
          {trainStatus.result_file}
        </Badge>
      )}
    </motion.div>
  )
}

function PredictForm({
  config,
  onSubmit,
  trainStatus,
}: {
  config: any
  onSubmit?: (data: any) => void
  trainStatus: any
}) {
  const [pipeline, setPipeline] = useState("BASE")
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [date, setDate] = useState("2025-01-01")
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [prevResultsFile, setPrevResultsFile] = useState<File | null>(null)

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
    if (!dataFile || !prevResultsFile) return
    onSubmit && onSubmit({ pipeline, selectedItems, date, dataFile, prevResultsFile })
  }

  const handleStop = async () => {
    await stopTrainTask()
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

            {/* File Uploads */}
            <div className="grid md:grid-cols-2 gap-8">
              <FileInput
                label="Файл с данными (ЧОК исторические)"
                accept=".xlsm,.xlsx"
                file={dataFile}
                onFileChange={setDataFile}
                onRemove={() => setDataFile(null)}
                icon={Database}
              />

              <FileInput
                label="Файл с предыдущими прогнозами"
                accept=".xlsm,.xlsx"
                file={prevResultsFile}
                onFileChange={setPrevResultsFile}
                onRemove={() => setPrevResultsFile(null)}
                icon={FileText}
              />
            </div>

            <Separator className="bg-gray-200" />

            {/* Submit Button, Status, Stop, Download Excel */}
            <div className="space-y-6">
              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-[26px] text-lg font-semibold shadow-lg"
                disabled={!dataFile || !prevResultsFile || trainStatus?.status === "running"}
              >
                <Play className="w-5 h-5 mr-2" />
                Запустить расчёт
              </Button>

              {/* Status Display */}
              <StatusIndicator trainStatus={trainStatus} />

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

              {/* Download Excel Button */}
              <Button
                type="button"
                onClick={async () => {
                  try {
                    const blob = await downloadExcel();
                    const url = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = "export_BASEPLUS.xlsx";
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(url);
                  } catch (err) {
                    alert("Не удалось скачать Excel файл");
                  }
                }}
                variant="outline"
                className="w-full bg-green-50 border-green-200 text-green-700 hover:bg-green-100 flex items-center justify-center text-lg font-semibold"
              >
                <Download className="w-5 h-5 mr-2" />
                Скачать Excel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default function App() {
  const [config, setConfig] = useState(null)
  const [trainStatus, setTrainStatus] = useState({ status: "idle" })
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    fetch("/config_refined.yaml")
      .then((res) => res.text())
      .then((text) => setConfig(parseYamlConfig(text)))
  }, [])

  useEffect(() => {
    let ignore = false
    async function checkStatus() {
      const status = await fetchTrainStatus()
      if (!ignore) {
        setTrainStatus(status && status.status ? status : { status: "idle" })
        if (status.status === "running") setPolling(true)
        else setPolling(false)
      }
    }
    checkStatus()
    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    if (!polling) return
    const interval = setInterval(async () => {
      const status = await fetchTrainStatus()
      setTrainStatus(status && status.status ? status : { status: "idle" })
      if (status.status !== "running") {
        setPolling(false)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [polling])

  const handleConfigLoad = async () => {
    const resp = await fetch("/config_refined.yaml")
    const text = await resp.text()
    setConfig(parseYamlConfig(text))
  }

  const handleLogDownload = () => {
    window.open("/logs", "_blank")
  }

  const handleTrainSubmit = async ({ pipeline, selectedItems, date, dataFile, prevResultsFile }: any) => {
    setTrainStatus({ status: "running" })
    setPolling(true)
    const resp = await sendTrainRequest({ pipeline, selectedItems, date, dataFile, prevResultsFile })
    // Можно обработать resp.task_id если нужно
  }

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <Sidebar onConfigLoad={handleConfigLoad} onLogDownload={handleLogDownload} />

      <main className="flex-1 p-8">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Система прогнозирования</h1>
          <p className="text-gray-600 text-lg">Корпоративная платформа анализа и прогнозирования данных</p>
        </motion.div>

        {config && <PredictForm config={config} onSubmit={handleTrainSubmit} trainStatus={trainStatus} />}
      </main>
    </div>
  )
}
