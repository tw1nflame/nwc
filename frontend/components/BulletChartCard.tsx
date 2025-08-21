"use client"
import React, { useEffect, useState, useRef } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import Plot from "react-plotly.js";
import { useConfig } from "../context/ConfigContext";


export interface BulletChartProps {
  data: Array<{
    article: string;
    deviation: number;
    difference: number;
    model?: string;
    date?: string;
    pipeline?: 'base' | 'base+';
  }>;
  loading?: boolean;
  currency?: 'RUB' | 'USD';
  exchangeRates?: any[];
  pipeline?: 'base' | 'base+';
}

export const BulletChartCard: React.FC<BulletChartProps> = ({ data, loading, currency = 'RUB', exchangeRates = [], pipeline = 'base' }) => {
  const { config } = useConfig();
  const usdArticles: string[] = config?.['Статьи для предикта в USD'] || [];
  // Получаем соответствие статья -> pipeline/model
  const modelArticle = config?.model_article || {};
  // Получаем лимиты из config для выбранной валюты
  const bulletLimits = config?.bullet_chart_limits?.[currency] || {};
  const diffMin = Number(bulletLimits.diff_min ?? -10);
  const diffMax = Number(bulletLimits.diff_max ?? 10);
  const devMin = Number(bulletLimits.deviation_min ?? -5);
  const devMax = Number(bulletLimits.deviation_max ?? 5);
  // Фильтрация и группировка данных + конвертация разницы в нужную валюту
  // Для алерта по отсутствию курса
  const alertShownRef = useRef(false);
  // Средний курс по exchangeRates (по всем датам, где есть курс)
  const meanRate = React.useMemo(() => {
    const rates = exchangeRates
      .map((r: any) => r['Курс'] ?? r['exchange_rate'])
      .filter((v: any) => typeof v === 'number' && !isNaN(v) && v > 0);
    if (!rates.length) return 1;
    return rates.reduce((a, b) => a + b, 0) / rates.length;
  }, [exchangeRates]);

  function convertDifference(diff: number, date: string | undefined, article: string) {
    const isUsdArticle = usdArticles.includes(article);
    if (!exchangeRates || !date) return diff;
    const rateRow = exchangeRates.find((r: any) => r['Дата'] === date);
    // поддержка разных ключей: 'Курс' (старый бэкенд), 'exchange_rate' (новый)
    let rate = rateRow ? (rateRow['Курс'] ?? rateRow['exchange_rate']) : null;
    if (!rate || typeof rate !== 'number' || isNaN(rate) || rate <= 0) {
      if (!alertShownRef.current) {
        console.log(`Курс отсутствует для даты ${date} и потенциально других дат. Для них курс заменён на среднее значение.`);
        alertShownRef.current = true;
      }
      rate = meanRate;
    }
    if (currency === 'RUB') {
      // USD-статьи переводим в рубли, рублевые не меняем
      if (isUsdArticle) return diff * rate;
      return diff;
    }
    if (currency === 'USD') {
      // Рублевые статьи переводим в доллары, долларовые не меняем
      if (!isUsdArticle) return diff / rate;
      return diff;
    }
    return diff;
  }
  // Для каждой статьи определяем pipeline и модель из config
  const filteredData = Array.isArray(data)
    ? data.filter(
        d => {
          const meta = modelArticle[d.article];
          return (
            typeof d.deviation === 'number' && !isNaN(d.deviation) &&
            typeof d.difference === 'number' && !isNaN(d.difference) &&
            meta && d.pipeline === meta.pipeline
          );
        }
      ).map(d => {
        const meta = modelArticle[d.article];
        return {
          ...d,
          model: meta?.model || d.model,
          difference: convertDifference(d.difference, d.date, d.article),
          deviation: d.deviation
        };
      })
    : [];

  const articles = Array.from(new Set(filteredData.map(d => d.article)));


  // MultiSelect: выбранные статьи
  const [selectedArticles, setSelectedArticles] = useState<string[]>(articles);
  // Сброс выбора если список статей изменился
  useEffect(() => {
    setSelectedArticles(articles);
  }, [JSON.stringify(articles)]);
  // удалено логгирование

  // Привязываем цвет к статье по её позиции в общем списке статей
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363', '#bdb76b'
  ];
  // Маппинг: статья -> цвет (позиция в полном списке articles)
  const articleColorMap: Record<string, string> = {};
  articles.forEach((article, idx) => {
    articleColorMap[article] = colors[idx % colors.length];
  });

  // Формируем traces для plotly
  const traces = articles
    .filter(article => selectedArticles.includes(article))
    .map((article) => {
      const points = filteredData.filter(d => d.article === article);
      return {
        x: points.map(d => d.deviation),
        y: points.map(d => d.difference),
        mode: 'markers',
        type: 'scatter',
        name: article,
        marker: { color: articleColorMap[article], size: 12, symbol: 'circle' },
  text: points.map(d => d.model || ''),
  customdata: points.map(d => d.date || ''),
  hovertemplate: `Статья: ${article}<br>Отклонение: %{x:.2f}%<br>Разница: %{y} ${currency === 'RUB' ? 'млн. руб' : 'млн. долл'}<br>Дата: %{customdata}<br>Модель: %{text}<extra></extra>`
      };
    });

  // Линии для осей
  const shapes = [
    // Горизонтальные линии (ось разницы)
    {
      type: 'line',
      xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: diffMin, y1: diffMin,
      line: { color: '#e74c3c', width: 2, dash: 'dot' },
    },
    {
      type: 'line',
      xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: diffMax, y1: diffMax,
      line: { color: '#e74c3c', width: 2, dash: 'dot' },
    },
    // Вертикальные линии (ось отклонения)
    {
      type: 'line',
      xref: 'x', yref: 'paper',
      x0: devMin, x1: devMin, y0: 0, y1: 1,
      line: { color: '#f39c12', width: 2, dash: 'dot' },
    },
    {
      type: 'line',
      xref: 'x', yref: 'paper',
      x0: devMax, x1: devMax, y0: 0, y1: 1,
      line: { color: '#f39c12', width: 2, dash: 'dot' },
    },
  ];

  return (
    <Card className="shadow-lg border-gray-200 bg-white mb-8">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-gray-900">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          График разниц и отклонений
        </CardTitle>
        <CardDescription>
          Отклонения по оси X и разница по оси Y для целевой модели по каждой статье.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col items-center justify-center min-h-[220px]">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500 border-solid mb-4" />
            <div className="text-base text-gray-500">Загрузка графика...</div>
          </div>
        ) : (
          <div>
            <div style={{ width: '100%', height: 520 }}>
              <Plot
                data={traces}
                layout={{
                  autosize: true,
                  height: 500,
                  margin: { t: 30, r: 30, l: 40, b: 40 },
                  xaxis: { title: { text: 'Отклонение (%)' }, zeroline: false },
                  yaxis: { title: { text: `Разница (${currency === 'RUB' ? 'млн. руб' : 'млн. долл'})` }, zeroline: false },
                  showlegend: false,
                  hovermode: 'closest',
                  font: { family: 'inherit', size: 14 },
                  plot_bgcolor: '#fff',
                  paper_bgcolor: '#fff',
                  shapes,
                  dragmode: 'zoom',
                }}
                config={{ responsive: true, displayModeBar: true, legendClick: false, legendDoubleClick: false }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
            <div className="flex flex-wrap gap-4 mt-4 p-2" style={{maxWidth: '100%'}}>
              {articles.map(article => (
                <label
                  key={article}
                  className="flex items-center gap-2 cursor-pointer whitespace-nowrap"
                  onDoubleClick={e => {
                    setSelectedArticles([article]);
                  }}
                >
                  <Checkbox
                    checked={selectedArticles.includes(article)}
                    onCheckedChange={checked => {
                      setSelectedArticles(prev =>
                        checked
                          ? [...prev, article]
                          : prev.filter(a => a !== article)
                      );
                    }}
                  />
                  <span>{article}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
