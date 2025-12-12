export async function downloadTaxExcel(accessToken) {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  const url = backendUrl.replace(/\/$/, '') + '/taxes/export_excel';
  let response;
  try {
    response = await fetch(url, {
      headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
    });
    if (!response.ok) throw new Error('Ошибка скачивания файла');
    const blob = await response.blob();
    return blob;
  } catch (err) {
    throw err;
  }
}
