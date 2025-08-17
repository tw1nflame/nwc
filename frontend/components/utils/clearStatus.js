export async function clearTrainStatus(accessToken) {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
  let url = backendUrl.replace(/\/$/, '') + '/clear_status/';
  const resp = await fetch(url, {
    method: 'POST',
    headers: accessToken ? { 'Authorization': `Bearer ${accessToken}` } : undefined
  });
  return await resp.json();
}
