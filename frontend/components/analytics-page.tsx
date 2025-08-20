"use client";
import { useAuth } from "../context/AuthContext";
import { useAnalyticsContext } from "../context/AnalyticsContext";
import dynamic from "next/dynamic";
import React from "react";
const BulletChartCard = dynamic(() => import("./BulletChartCard").then(mod => mod.BulletChartCard), { ssr: false });




export function AnalyticsPage() {
  const {
    bulletChartData,
    setBulletChartData,
    exchangeRates,
    setExchangeRates,
    loading,
    setLoading,
    startMonth,
    setStartMonth,
    endMonth,
    setEndMonth,
    currency,
    setCurrency
  } = useAnalyticsContext();
  const { session } = useAuth();
  React.useEffect(() => {
    if (bulletChartData.length > 0 && exchangeRates.length > 0) {
      setLoading(false);
      return;
    }
    setLoading(true);
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    const url = backendUrl.replace(/\/$/, '') + '/bullet_chart_data/';
    const accessToken = session?.access_token;
    fetch(url, {
      headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
    })
      .then(res => res.json())
      .then(data => {
        setBulletChartData(data.data || []);
        setExchangeRates(data.exchange_rates || []);
        setLoading(false);
      })
      .catch(() => {
        setBulletChartData([]);
        setExchangeRates([]);
        setLoading(false);
      });
  }, [session, setBulletChartData, setExchangeRates, setLoading]);

  // Получить все уникальные месяцы из данных (формат YYYY-MM)
  const allMonths = React.useMemo(() => {
    const months = bulletChartData.map(d => (d.date ? d.date.slice(0, 7) : null)).filter(Boolean);
    return Array.from(new Set(months)).sort();
  }, [bulletChartData]);

  // Установить период по умолчанию: с самого раннего по самый поздний месяц
  React.useEffect(() => {
    if (allMonths.length > 0) {
      setStartMonth(allMonths[0]);
      setEndMonth(allMonths[allMonths.length - 1]);
    }
  }, [allMonths, setStartMonth, setEndMonth]);

  // Фильтрация данных по выбранному периоду
  const filteredData = React.useMemo(() => {
    if (!startMonth && !endMonth) return bulletChartData;
    return bulletChartData.filter(d => {
      const m = d.date ? d.date.slice(0, 7) : null;
      if (!m) return false;
      if (startMonth && m < startMonth) return false;
      if (endMonth && m > endMonth) return false;
      return true;
    });
  }, [bulletChartData, startMonth, endMonth]);

  return (
    <main className="flex-1 p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Общая аналитика</h1>
        <p className="text-gray-600 text-lg">Аналитика по результатам прогнозирования и обучению моделей</p>
      </div>
      <div className="mb-6">
        <div className="bg-white border border-gray-200 rounded-lg p-4 flex flex-col md:flex-row gap-4 items-center shadow-sm">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-800">Параметры анализа:</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="font-medium">Период с</label>
            <select
              className="border rounded px-2 py-1"
              value={startMonth || ''}
              onChange={e => setStartMonth(e.target.value || null)}
            >
              <option value="">-</option>
              {allMonths.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="font-medium">по</label>
            <select
              className="border rounded px-2 py-1"
              value={endMonth || ''}
              onChange={e => setEndMonth(e.target.value || null)}
            >
              <option value="">-</option>
              {allMonths.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-4">
            <label className="font-medium">Валюта</label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="currency"
                value="RUB"
                checked={currency === 'RUB'}
                onChange={() => setCurrency('RUB')}
                className="accent-blue-600"
              />
              <span>RUB</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="currency"
                value="USD"
                checked={currency === 'USD'}
                onChange={() => setCurrency('USD')}
                className="accent-blue-600"
              />
              <span>USD</span>
            </label>
          </div>
        </div>
      </div>
      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[40vh]">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 border-solid mb-6" />
          <div className="text-lg text-gray-500">Загрузка данных...</div>
        </div>
      ) : (
  <BulletChartCard data={filteredData} currency={currency} exchangeRates={exchangeRates} />
      )}
    </main>
  );
}
