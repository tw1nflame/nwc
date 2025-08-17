import React, { useEffect, useState } from "react";
import { CardContent } from "@/components/ui/card";
import { fetchArticleStatsExcel } from "./utils/fetchArticleStatsExcel";
import { useAuth } from "../context/AuthContext";


interface ArticleStatsTableProps {
  article: string;
  currency: 'RUB' | 'USD';
  exchangeRatesJson: any[];
  parsedJson: any[];
  config: any;
}


export const ArticleStatsTable: React.FC<ArticleStatsTableProps> = ({ article, currency, exchangeRatesJson, parsedJson, config }) => {
  const { session } = useAuth();
  const accessToken = session?.access_token;
  const [rawStats, setRawStats] = useState<any[][] | null>(null);
  const [rawColumns, setRawColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- Копируем логику валют из AnalysisPage ---
  function excelSerialToDate(serial: number): string {
    const excelEpoch = new Date(Date.UTC(1899, 11, 30))
    const msPerDay = 24 * 60 * 60 * 1000
    const date = new Date(excelEpoch.getTime() + serial * msPerDay)
    return date.toISOString().slice(0, 10)
  }
  // Логика выбора целевой модели и валюты как в ArticleDataTable
  const usdArticles = config?.["Статьи для предикта в USD"] || [];
  const modelArticle = config?.model_article || {};
  const selectedArticleLower = article.trim().toLowerCase();
  // В ArticleDataTable mainModelLower = modelArticle[selectedArticle]?.toLowerCase()
  const mainModel = modelArticle[article] || null;
  const mainModelLower = mainModel ? mainModel.toLowerCase() : null;
  const isUsdArticle = usdArticles.map((a: string) => a.trim().toLowerCase()).includes(selectedArticleLower);
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

  // Загружаем файл статистики только при смене статьи
  useEffect(() => {
    if (!article) return;
    setLoading(true);
    setError(null);
    fetchArticleStatsExcel(article, accessToken)
      .then(({ columns, stats }) => {
        setRawColumns(columns);
        setRawStats(stats);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [article, accessToken]);

  // Фильтрация столбцов под выбранную валюту: показываем только (RUB) или (USD) + проценты
  const { columns, stats } = React.useMemo(() => {
    if (!rawStats || !rawColumns.length) return { columns: [], stats: null as any };
    const want = currency;
    const selectedIdx: number[] = [];
    const selectedCols: string[] = [];
    rawColumns.forEach((col, idx) => {
      const isPct = /отклонение\s*%/i.test(col);
      const isRub = /\(\s*RUB\s*\)$/.test(col);
      const isUsd = /\(\s*USD\s*\)$/.test(col);
      if (isPct) {
        selectedIdx.push(idx);
        selectedCols.push(col);
      } else if (want === 'RUB' && isRub) {
        selectedIdx.push(idx);
        selectedCols.push(col);
      } else if (want === 'USD' && isUsd) {
        selectedIdx.push(idx);
        selectedCols.push(col);
      }
    });
    const filteredStats: any[][] = (rawStats as any[]).map((row: any[]) => {
      const out: any[] = [row[0]]; // metric name
      (selectedIdx as number[]).forEach((idx: number) => {
        out.push(row[idx + 1]); // +1 потому что первый столбец — метрика
      });
      return out;
    });
    return { columns: selectedCols, stats: filteredStats };
  }, [rawStats, rawColumns, currency]);

  if (loading) return <div className="py-8 text-center">Загрузка статистики...</div>;
  if (error) return <div className="py-8 text-center text-red-500">Ошибка: {error}</div>;
  if (!stats || !columns.length) return null;

  // Функция для проверки, относится ли столбец к целевой модели
  const isTargetColumn = (col: string) => {
    if (!mainModelLower) return false;
    // Проверяем, относится ли столбец к целевой модели (по названию модели)
    // Например: predict_<model>, predict_<model> разница, predict_<model> отклонение % и т.д.
    const patterns = [
      `predict_${mainModelLower}`,
      `predict_${mainModelLower} разница`,
      `predict_${mainModelLower} отклонение %`,
      `${mainModelLower}`,
      `${mainModelLower} разница`,
      `${mainModelLower} отклонение %`,
    ];
    const colNorm = col.replace(/\s+/g, ' ').toLowerCase();
    return patterns.some((pat) => colNorm.includes(pat));
  };

  return (
    <CardContent>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="py-2 px-3 border-b border-gray-200 text-left bg-gray-50 min-w-[210px]">Метрика</th>
              {columns.map((col) => (
                <th key={col} className="py-2 px-3 border-b border-gray-200 text-center bg-gray-50">
                  {col}
                  {isTargetColumn(col) && (
                    <span className="ml-1 px-2 py-1 text-xs font-medium text-blue-600 bg-blue-100 rounded-full align-middle">целевая</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stats.map((row: any[], i: number) => (
              <tr key={i} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                {row.map((cell: any, j: number) => {
                  // Первый столбец — метрика, остальные — значения по колонкам
                  const isTarget = j > 0 && isTargetColumn(columns[j - 1]);
                  return (
                    <td
                      key={j}
                      className={
                        j === 0
                          ? "py-2 px-3 font-medium text-gray-900 min-w-[210px]"
                          : isTarget
                            ? "py-2 px-3 text-center text-blue-700 font-semibold"
                            : "py-2 px-3 text-center text-gray-700"
                      }
                    >
                      {cell === null || cell === undefined || Number.isNaN(cell)
                        ? '-'
                        : typeof cell === 'number'
                          ? cell.toLocaleString(undefined, { maximumFractionDigits: 3 })
                          : cell}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </CardContent>
  );
};
