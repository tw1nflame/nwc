// utils/trainStatus.js
export async function fetchTrainStatus(accessToken) {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  let url = backendUrl.replace(/\/$/, '') + '/train_status/';
  let response;
  try {
    response = await fetch(url, {
      headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
    });
  } catch (e) {
    return { status: 'error', error: 'Network error', details: e.toString() };
  }
  let data;
  try {
    data = await response.json();
  } catch {
    data = { status: 'error', error: 'Invalid JSON response', statusCode: response.status };
  }
  // Если с бэкенда пришёл пустой объект или нет status, возвращаем idle
  if (!data || !data.status) {
    return { status: 'idle' };
  }
  return data;
}
