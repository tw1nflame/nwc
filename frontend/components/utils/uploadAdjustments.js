/**
 * Загружает файл корректировок на сервер
 * @param {File} file - Excel файл с корректировками
 * @param {string} dateColumn - название столбца с датой (по умолчанию 'Дата')
 * @returns {Promise<Object>} Результат загрузки
 */
export async function uploadAdjustments(file, dateColumn = 'Дата') {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const url = backendUrl.replace(/\/$/, '') + '/upload_adjustments/';
  
  const formData = new FormData();
  formData.append('adjustments_file', file);
  formData.append('date_column', dateColumn);
  
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Ошибка загрузки файла корректировок');
  }
  
  return await response.json();
}
