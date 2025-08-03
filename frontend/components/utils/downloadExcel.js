export async function downloadExcel() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const url = backendUrl.replace(/\/$/, '') + '/export_excel/';
  let response;
  try {
    response = await fetch(url);
    if (!response.ok) throw new Error('Ошибка скачивания файла');
    const blob = await response.blob();
    return blob;
  } catch (err) {
    throw err;
  }
}
