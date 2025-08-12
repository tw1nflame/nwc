import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { BarChart3, TrendingUp, AlertTriangle } from "lucide-react";

export function AnalysisGraphs({
  chartData,
  mainModelLower,
  forecastType,
  selectedArticle,
  ForecastChart,
  ErrorChart,
  AbsoluteDiffChart
}) {
  return (
    <>
      <Card className="shadow-lg border-gray-200 bg-white mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-900">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            {forecastType === 'corrected' ? 'Скорректированный прогноз vs Факт' : 'Прогноз vs Факт'}
          </CardTitle>
          <CardDescription>
            Сравнение {forecastType === 'corrected' ? 'скорректированных (прогноз + корректировки)' : 'прогнозируемых'} и фактических значений по датам для {selectedArticle} и всех выбранных моделей
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ForecastChart data={chartData} mainModelLower={mainModelLower} forecastType={forecastType} />
        </CardContent>
      </Card>
      <Card className="shadow-lg border-gray-200 bg-white mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-900">
            <AlertTriangle className="w-5 h-5 text-amber-600" />
            Ошибка %
          </CardTitle>
          <CardDescription>
            Ошибка {forecastType === 'corrected' ? 'скорректированного прогноза' : 'прогноза'} в процентах по датам для {selectedArticle} и всех выбранных моделей
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ErrorChart data={chartData} mainModelLower={mainModelLower} />
        </CardContent>
      </Card>
      <Card className="shadow-lg border-gray-200 bg-white mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-900">
            <BarChart3 className="w-5 h-5 text-purple-600" />
            Разница
          </CardTitle>
          <CardDescription>
            Разница между {forecastType === 'corrected' ? 'скорректированным прогнозом' : 'прогнозом'} и фактом по датам для {selectedArticle} и всех выбранных моделей
          </CardDescription>
        </CardHeader>
        <CardContent>
          <AbsoluteDiffChart data={chartData} mainModelLower={mainModelLower} />
        </CardContent>
      </Card>
    </>
  );
}
