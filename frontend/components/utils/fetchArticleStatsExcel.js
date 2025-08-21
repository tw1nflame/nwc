import * as XLSX from "xlsx";

// Простое кеширование и дедупликация запросов по статье
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 минут
const cache = new Map(); // article -> { ts, data }
const inflight = new Map(); // article -> Promise

export async function fetchArticleStatsExcel(article, accessToken, pipeline) {
  const now = Date.now();
  // Кешировать по article+pipeline
  const cacheKey = `${article}__${pipeline || ''}`;
  const cached = cache.get(cacheKey);
  if (cached && now - cached.ts < CACHE_TTL_MS) {
    return cached.data;
  }
  if (inflight.has(cacheKey)) {
    return inflight.get(cacheKey);
  }
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  let url = backendUrl.replace(/\/$/, '') + `/export_article_stats/?article=${encodeURIComponent(article)}`;
  if (pipeline) {
    url += `&pipeline=${encodeURIComponent(pipeline)}`;
  }
  const promise = (async () => {
    let response;
    response = await fetch(url, {
      headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined,
      cache: 'no-store'
    });
    try {
      if (!response.ok) throw new Error('Ошибка скачивания файла');
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const workbook = XLSX.read(arrayBuffer, { type: "array" });
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      const json = XLSX.utils.sheet_to_json(sheet, { header: 1 });
      if (!json.length) throw new Error("Пустой файл статистики");
      const data = {
        columns: json[0].slice(1),
        stats: json.slice(1)
      };
  cache.set(cacheKey, { ts: Date.now(), data });
      return data;
    } catch (e) {
      throw new Error(e.message || 'Ошибка получения статистики');
    } finally {
      inflight.delete(article);
    }
  })();
  inflight.set(cacheKey, promise);
  return promise;
}
