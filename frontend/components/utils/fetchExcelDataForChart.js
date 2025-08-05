import * as XLSX from "xlsx"

export async function fetchExcelDataForChart(modelName, articleName) {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const url = backendUrl.replace(/\/$/, '') + '/export_excel/';
  let response;
  try {
    response = await fetch(url);
    if (!response.ok) throw new Error('Ошибка скачивания файла');
    const blob = await response.blob();
    const arrayBuffer = await blob.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: "array" });
    console.log('[fetchExcelDataForChart] Sheet names:', workbook.SheetNames);
    const sheet = workbook.Sheets["data"];
    if (!sheet) throw new Error('Нет листа data');
    const json = XLSX.utils.sheet_to_json(sheet);
    console.log('[fetchExcelDataForChart] Header row:', Object.keys(json[0] || {}));
    console.log('[fetchExcelDataForChart] First 3 rows:', json.slice(0, 3));
    // Print unique articles
    const uniqueArticles = [...new Set(json.map(row => row["Статья"]))];
    console.log('[fetchExcelDataForChart] Уникальные статьи:', uniqueArticles);
    // Temporary fix: always append _USD for Торговая ДЗ
    let fixedArticleName = articleName;
    if (articleName.trim().toLowerCase() === 'торговая дз') {
      fixedArticleName = articleName + '_USD';
    }
    // Filter by article (case/whitespace-insensitive)
    const filtered = json.filter(row => {
      const val = row["Статья"];
      return val && val.trim().toLowerCase() === fixedArticleName.trim().toLowerCase();
    });
    console.log('[fetchExcelDataForChart] Filtered rows for article', fixedArticleName, ':', filtered.length);
    if (filtered.length > 0) {
      console.log('[fetchExcelDataForChart] First filtered row:', filtered[0]);
    }
    // Build chart data
    const chartData = filtered.map(row => ({
      date: row["Дата"],
      actual: row["Fact"],
      forecast: row[`predict_${modelName}`],
      error: row[`predict_${modelName}`] && row["Fact"] ? Number((((row[`predict_${modelName}`] - row["Fact"]) / row["Fact"]) * 100).toFixed(2)) : null
    }))
    console.log('[fetchExcelDataForChart] chartData:', chartData);
    return chartData
  } catch (err) {
    console.error('[fetchExcelDataForChart] Error:', err);
    throw err;
  }
}
