"use client"
import React, { useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import Plot from "react-plotly.js";

export interface BulletChartProps {
  data: Array<{
    article: string;
    deviation: number;
    difference: number;
    model?: string;
  }>;
  onLoaded?: () => void;
}
export const BulletChartCard: React.FC<BulletChartProps> = ({ data, onLoaded }) => {
  useEffect(() => {
    if (onLoaded) {
      // Даем небольшой таймаут для плавности, иначе может "мигнуть"
      const t = setTimeout(() => onLoaded(), 100);
      return () => clearTimeout(t);
    }
  }, [onLoaded]);
  // Фильтрация и группировка данных
  const filteredData = Array.isArray(data)
    ? data.filter(
        d => typeof d.deviation === 'number' && !isNaN(d.deviation) && typeof d.difference === 'number' && !isNaN(d.difference)
      )
    : [];

  const articles = Array.from(new Set(filteredData.map(d => d.article)));
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];

  // Формируем traces для plotly
  const traces = articles.map((article, idx) => {
    const points = filteredData.filter(d => d.article === article);
    return {
      x: points.map(d => d.deviation),
      y: points.map(d => d.difference),
      mode: 'markers',
      type: 'scatter',
      name: article,
      marker: { color: colors[idx % colors.length], size: 12, symbol: 'circle' },
      text: points.map(d => d.model || ''),
      hovertemplate: `Статья: ${article}<br>Отклонение: %{x:.2f}%<br>Разница: %{y}<br>Модель: %{text}<extra></extra>`
    };
  });

  return (
    <Card className="shadow-lg border-gray-200 bg-white mb-8">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-gray-900">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          Bullet график
        </CardTitle>
        <CardDescription>
          Отклонения по оси X и разница по оси Y для целевой модели по каждой статье. Данные подгружаются автоматически.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div style={{ width: '100%', height: 520 }}>
          <Plot
            data={traces}
            layout={{
              autosize: true,
              height: 500,
              margin: { t: 30, r: 30, l: 40, b: 40 },
              xaxis: { title: 'Отклонение', zeroline: false },
              yaxis: { title: 'Разница', zeroline: false },
              legend: { orientation: 'h', y: -0.2 },
              hovermode: 'closest',
              font: { family: 'inherit', size: 14 },
              plot_bgcolor: '#fff',
              paper_bgcolor: '#fff',
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </CardContent>
    </Card>
  );
};
