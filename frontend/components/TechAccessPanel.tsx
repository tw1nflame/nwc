"use client";

import React, { useState } from "react";
import { checkSecretWord, downloadLogsArchive, uploadOldForecast, stopAndClearTraining } from "./utils/techAccessApi";

export const TechAccessPanel: React.FC = () => {
  const [secret, setSecret] = useState("");
  const [checked, setChecked] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logLoading, setLogLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [stopLoading, setStopLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [stopMsg, setStopMsg] = useState<string | null>(null);

  const checkSecret = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await checkSecretWord(secret);
      if (data.ok) {
        setChecked(true);
      } else {
        setError("Секретное слово неверно");
      }
    } catch (e) {
      setError("Ошибка запроса");
    } finally {
      setLoading(false);
    }
  };

  const downloadLogs = async () => {
    setLogLoading(true);
    setError(null);
    try {
      const blob = await downloadLogsArchive(secret);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "logs.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setError("Ошибка скачивания логов");
    } finally {
      setLogLoading(false);
    }
  };

  const uploadExcel = async () => {
    if (!file) return;
    setUploadLoading(true);
    setUploadMsg(null);
    setError(null);
    try {
      const data = await uploadOldForecast(secret, file);
      if (data.ok) {
        setUploadMsg("Файл успешно загружен в БД");
      } else {
        setError(data.detail || "Ошибка загрузки");
      }
    } catch (e) {
      setError("Ошибка загрузки файла");
    } finally {
      setUploadLoading(false);
    }
  };

  const stopAndClear = async () => {
    setStopLoading(true);
    setStopMsg(null);
    setError(null);
    try {
      const data = await stopAndClearTraining(secret);
      if (data.ok) {
        setStopMsg(data.message || "Задача остановлена и Redis очищен");
      } else {
        setError(data.detail || "Ошибка остановки");
      }
    } catch (e) {
      setError("Ошибка остановки задачи");
    } finally {
      setStopLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded bg-gray-50 mt-4">
      <div className="font-semibold mb-2">Скачать логи</div>
      {!checked ? (
        <div className="flex flex-col gap-2">
          <input
            type="password"
            className="border rounded px-2 py-1"
            placeholder="Секретное слово"
            value={secret}
            onChange={e => setSecret(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") checkSecret(); }}
            disabled={loading}
          />
          <button
            className="bg-blue-600 text-white rounded px-3 py-1 disabled:opacity-60"
            onClick={checkSecret}
            disabled={loading || !secret}
          >
            {loading ? "Проверка..." : "Войти"}
          </button>
          {error && <div className="text-red-600 text-sm">{error}</div>}
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <button
            className="bg-green-600 text-white rounded px-3 py-1 disabled:opacity-60"
            onClick={downloadLogs}
            disabled={logLoading}
          >
            {logLoading ? "Скачивание..." : "Скачать логи (архив)"}
          </button>
          
          <div className="font-semibold mb-2 mt-4">Экстренная остановка</div>
          <button
            className="bg-red-600 text-white rounded px-3 py-1 disabled:opacity-60"
            onClick={stopAndClear}
            disabled={stopLoading}
          >
            {stopLoading ? "Остановка..." : "Остановить задачу и очистить Redis"}
          </button>
          {stopMsg && <div className="text-green-700 text-sm mt-1">{stopMsg}</div>}
          
          <div className="font-semibold mb-2 mt-4">Загрузить старый прогноз</div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <input
                type="file"
                accept=".xlsx"
                onChange={e => setFile(e.target.files?.[0] || null)}
                className="mb-0 w-32 shrink-0"
                disabled={uploadLoading}
                style={{ maxWidth: 130 }}
              />
              {file && (
                <span
                  className="truncate text-xs text-gray-700 max-w-[120px] inline-block align-middle"
                  title={file.name}
                  style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                >
                  {file.name}
                </span>
              )}
            </div>
            <div style={{ display: 'block', width: '100%' }}>
              <button
                className="bg-blue-700 text-white rounded px-3 py-1 disabled:opacity-60 w-full border border-blue-700"
                onClick={uploadExcel}
                disabled={uploadLoading || !file}
                style={{ marginTop: 2, minHeight: 36, background: '#1d4ed8', opacity: (uploadLoading || !file) ? 0.6 : 1, pointerEvents: 'auto', visibility: 'visible', width: '100%' }}
              >
                {uploadLoading ? "Загрузка..." : "Загрузить старый прогноз"}
              </button>
              {uploadMsg && <div className="text-green-700 text-sm mt-1">{uploadMsg}</div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
