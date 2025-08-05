import * as XLSX from "xlsx"

// Accepts arrayBuffer, if not provided fetches from backend
export async function fetchExcelAndParseModels(arrayBuffer) {
  if (!arrayBuffer) {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    const url = backendUrl.replace(/\/$/, '') + '/export_excel/';
    const response = await fetch(url);
    if (!response.ok) throw new Error('Ошибка скачивания файла');
    const blob = await response.blob();Context 
    arrayBuffer = await blob.arrayBuffer();
  }
  const workbook = XLSX.read(arrayBuffer, { type: "array" });
  // Find first sheet with predict_ columns
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  const json = XLSX.utils.sheet_to_json(sheet, { header: 1 });
  const headers = json[0] || [];
  const models = headers
    .filter((h) => typeof h === "string" && h.startsWith("predict_"))
    .map((h) => h.replace("predict_", ""));
  return models;
}
