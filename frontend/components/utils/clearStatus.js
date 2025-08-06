export async function clearTrainStatus() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  let url = backendUrl.replace(/\/$/, '') + '/clear_status/';
  const resp = await fetch(url, { method: 'POST' });
  return await resp.json();
}
