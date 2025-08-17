"use client"

import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  X,
  Download,
  FileText,
  Database,
  CheckCircle,
  AlertCircle,
  File,
  FileSpreadsheet,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { downloadExcel } from "./utils/downloadExcel"
import { uploadAdjustments } from "./utils/uploadAdjustments"
import { useAuth } from "../context/AuthContext"

function FileInput({
  label,
  accept,
  file,
  onFileChange,
  onRemove,
  icon: Icon = FileText,
  description,
}: {
  label: string
  accept: string
  file: File | null
  onFileChange: (file: File | null) => void
  onRemove: () => void
  icon?: React.ComponentType<{ className?: string }>
  description?: string
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
      
      {description && (
        <p className="text-sm text-gray-600">{description}</p>
      )}

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
                <FileSpreadsheet className="w-4 h-4 text-amber-600" />
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

function UploadStatus({ status, message }: { status: "idle" | "uploading" | "success" | "error"; message?: string }) {
  if (status === "idle") return null

  const config = {
    uploading: {
      icon: Database,
      text: "Загрузка файла...",
      bgClass: "bg-blue-50 border-blue-200 text-blue-800",
      iconClass: "text-blue-600 animate-pulse",
    },
    success: {
      icon: CheckCircle,
      text: "Файл успешно загружен!",
      bgClass: "bg-emerald-50 border-emerald-200 text-emerald-800",
      iconClass: "text-emerald-600",
    },
    error: {
      icon: AlertCircle,
      text: message || "Ошибка загрузки файла",
      bgClass: "bg-red-50 border-red-200 text-red-800",
      iconClass: "text-red-600",
    },
  }[status]

  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${config.bgClass} font-medium`}
    >
      <Icon className={`w-5 h-5 ${config.iconClass}`} />
      <span>{config.text}</span>
    </motion.div>
  )
}

export function ExportPage() {
  const { session } = useAuth();
  const accessToken = session?.access_token;
  const [correctionFile, setCorrectionFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle")
  const [uploadMessage, setUploadMessage] = useState("")

  const handleCorrectionUpload = async () => {
    if (!correctionFile) return
    setUploadStatus("uploading")
    setUploadMessage("")
    try {
      const result = await uploadAdjustments(correctionFile, 'Дата', accessToken) as any
      setUploadStatus("success")
      setUploadMessage(`Успешно обработано ${result.processed_adjustments} корректировок`)
      setTimeout(() => {
        setCorrectionFile(null)
        setUploadStatus("idle")
      }, 3000)
    } catch (error) {
      setUploadStatus("error")
      setUploadMessage(error instanceof Error ? error.message : "Неизвестная ошибка")
    }
  }
  const handleExcelDownload = async () => {
    try {
      const blob = await downloadExcel(accessToken)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const currentDate = new Date().toISOString().split('T')[0]
      a.download = `predict_${currentDate}.xlsx`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Ошибка скачивания файла')
    }
  }

  return (
    <main className="flex-1 p-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Выгрузка и корректировки</h1>
        <p className="text-gray-600 text-lg">Скачивание результатов прогноза и загрузка файлов корректировок</p>
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
              Скачайте полный файл прогноза в формате Excel. Включает все статьи и модели с результатами расчета.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 flex flex-col justify-end">
            <Button
              onClick={handleExcelDownload}
              className="w-full bg-green-600 hover:bg-green-700 text-white py-4 text-base font-semibold shadow-lg"
            >
              <Download className="w-4 h-4 mr-2" />
              Скачать Excel файл
            </Button>
          </CardContent>
        </Card>

        {/* Карточка загрузки корректировок */}
        <Card className="shadow-xl border-gray-200 bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-900">
              <Upload className="w-5 h-5 text-blue-600" />
              Загрузка корректировок
            </CardTitle>
            <CardDescription>
              Загрузите файл с корректировками для обновления прогнозных данных. Поддерживаются файлы Excel (.xlsx, .xlsm).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <FileInput
                label="Файл корректировок"
                accept=".xlsx,.xlsm"
                file={correctionFile}
                onFileChange={setCorrectionFile}
                onRemove={() => setCorrectionFile(null)}
                icon={FileSpreadsheet}
                description="Выберите Excel файл с корректировками прогнозных значений"
              />
            </div>

            <Button
              onClick={handleCorrectionUpload}
              disabled={!correctionFile || uploadStatus === "uploading"}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white py-4 text-base font-semibold shadow-lg"
            >
              {uploadStatus === "uploading" ? (
                <>
                  <Database className="w-4 h-4 mr-2 animate-pulse" />
                  Загрузка...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Загрузить корректировки
                </>
              )}
            </Button>

            <UploadStatus status={uploadStatus} message={uploadMessage} />
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
