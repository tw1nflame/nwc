import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TrendingUp } from "lucide-react";
import { motion } from "framer-motion";

// Вынесенная таблица "Данные по статье"
export function ArticleDataTable({
  selectedArticle,
  showFullTable,
  setShowFullTable,
  chartData,
  forecastType,
  selectedModels,
  mainModelLower,
  excelSerialToDate
}: {
  selectedArticle: string;
  showFullTable: boolean;
  setShowFullTable: (v: boolean) => void;
  chartData: any[];
  forecastType: 'original' | 'corrected';
  selectedModels: string[];
  mainModelLower?: string;
  excelSerialToDate: (serial: number) => string;
}) {
  return (
    <Card className="shadow-lg border-gray-200 bg-white mb-8">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-gray-900">
              <Table className="w-5 h-5 text-green-600" />
              Данные по статье: {selectedArticle}
            </CardTitle>
            <CardDescription>
              Детальные данные с отклонениями и разницей для всех выбранных моделей
            </CardDescription>
          </div>
          <Button
            variant="outline"
            onClick={() => setShowFullTable(!showFullTable)}
            className="flex items-center gap-2"
          >
            {showFullTable ? 'Свернуть' : `Показать все (${chartData.length > 0 ? chartData[0].data.length : 0})`}
            <motion.div
              animate={{ rotate: showFullTable ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <TrendingUp className="w-4 h-4" />
            </motion.div>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex">
          {/* Фиксированные колонки: Дата и Факт */}
          <div className="flex-none">
            <table className="border-collapse text-sm">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  <th className="text-left py-3 px-3 font-semibold text-gray-900 border-r border-gray-300 min-w-[100px] h-[72px] align-middle">Дата</th>
                  <th className="text-center py-3 px-3 font-semibold text-gray-900 border-r border-gray-400 min-w-[100px] h-[72px] align-middle">Факт</th>
                  {forecastType === 'corrected' && (
                    <>
                      <th className="text-center py-3 px-3 font-semibold text-orange-700 border-r border-gray-400 min-w-[100px] h-[72px] align-middle">Корректировка</th>
                      <th className="text-center py-3 px-3 font-semibold text-orange-700 border-r border-gray-400 min-w-[120px] h-[72px] align-middle">Финальный прогноз</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {chartData.length > 0 && chartData[0].data
                  .slice(0, showFullTable ? undefined : 10)
                  .map((row: any, index: number) => (
                    <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                      <td className="py-3 px-3 font-medium text-gray-900 border-r border-gray-300">
                        {typeof row.date === 'number' ? excelSerialToDate(row.date) : row.date}
                      </td>
                      <td className="py-3 px-3 text-center font-medium text-gray-700 border-r border-gray-400">
                        {row.actual !== null && row.actual !== undefined ? Number(row.actual).toLocaleString() : '-'}
                      </td>
                      {forecastType === 'corrected' && (
                        <>
                          <td className="py-3 px-3 text-center font-medium text-orange-700 border-r border-gray-400">
                            {row.adjustments !== null && row.adjustments !== undefined && row.adjustments !== 0
                              ? Number(row.adjustments).toLocaleString()
                              : '0'}
                          </td>
                          <td className="py-3 px-3 text-center font-medium text-orange-700 border-r border-gray-400">
                            {row.forecast !== null && row.forecast !== undefined
                              ? Number(Number(row.forecast) + Number(row.adjustments || 0)).toLocaleString()
                              : '-'}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          {/* Прокручиваемые колонки: данные моделей */}
          <div className="flex-1 overflow-x-auto">
            <table className="border-collapse text-sm w-full">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  {selectedModels.map((model, modelIndex) => (
                    <React.Fragment key={model}>
                      <th className={`text-center py-3 px-3 font-semibold text-gray-700 h-[72px] align-middle ${modelIndex === 0 ? 'border-l-2 border-gray-500' : ''}`} style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '120px' }}>
                        Прогноз ({model})
                        {mainModelLower && model.toLowerCase() === mainModelLower && (
                          <span className="ml-1 px-2 py-1 text-xs font-medium text-blue-600 bg-blue-100 rounded-full">
                            целевая
                          </span>
                        )}
                      </th>
                      <th className="text-center py-3 px-3 font-semibold text-gray-700 h-[72px] align-middle" style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '100px' }}>Отклонение %</th>
                      <th className={`text-center py-3 px-3 font-semibold text-gray-700 h-[72px] align-middle ${modelIndex < selectedModels.length - 1 ? 'border-r-2 border-gray-500' : ''}`} style={{ width: `${100 / (selectedModels.length * 3)}%`, minWidth: '100px' }}>Разница</th>
                    </React.Fragment>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chartData.length > 0 && chartData[0].data
                  .slice(0, showFullTable ? undefined : 10)
                  .map((row: any, index: number) => (
                    <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                      {selectedModels.map((model, modelIndex) => {
                        const modelData = chartData.find(d => d.model === model);
                        const modelRow = modelData?.data[index];
                        const isMainModel = mainModelLower && model.toLowerCase() === mainModelLower;
                        return (
                          <React.Fragment key={model}>
                            <td className={`py-3 px-3 text-center font-medium text-gray-700 ${modelIndex === 0 ? 'border-l-2 border-gray-500' : ''}`}>
                              {modelRow?.forecast !== null && modelRow?.forecast !== undefined ? Number(modelRow.forecast).toLocaleString() : '-'}
                            </td>
                            <td className={`py-3 px-3 text-center font-medium text-gray-700`}>
                              {modelRow?.errorPercent !== null && modelRow?.errorPercent !== undefined && modelRow?.errorPercent !== ''
                                ? `${parseFloat(modelRow.errorPercent).toFixed(2)}%`
                                : '-'}
                            </td>
                            <td className={`py-3 px-3 text-center font-medium text-gray-700 ${modelIndex < selectedModels.length - 1 ? 'border-r-2 border-gray-500' : ''}`}>
                              {modelRow?.difference !== null && modelRow?.difference !== undefined ? Number(modelRow.difference).toLocaleString() : '-'}
                            </td>
                          </React.Fragment>
                        );
                      })}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
        {!showFullTable && chartData.length > 0 && chartData[0].data.length > 10 && (
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-500 mb-3">
              Показано {Math.min(10, chartData[0].data.length)} из {chartData[0].data.length} записей
            </p>
            <Button
              variant="ghost"
              onClick={() => setShowFullTable(true)}
              className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
            >
              Показать все записи
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
