export async function stopTrainTask() {
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  let url = backendUrl.replace(/\/$/, '') + '/stop_train/';
  const resp = await fetch(url, { method: 'POST' });
  return await resp.json();
}
