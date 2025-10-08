const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export async function checkSecretWord(word) {
  const form = new FormData();
  form.append('word', word);
  const url = backendUrl.replace(/\/$/, '') + '/check-secret-word';
  const res = await fetch(url, { method: 'POST', body: form });
  return await res.json();
}

export async function downloadLogsArchive(word) {
  const form = new FormData();
  form.append('word', word);
  const url = backendUrl.replace(/\/$/, '') + '/download-logs-archive';
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Ошибка скачивания');
  return await res.blob();
}

export async function uploadOldForecast(word, file) {
  const form = new FormData();
  form.append('word', word);
  form.append('file', file);
  const url = backendUrl.replace(/\/$/, '') + '/upload-excel';
  const res = await fetch(url, { method: 'POST', body: form });
  return await res.json();
}

export async function stopAndClearTraining(word) {
  const form = new FormData();
  form.append('word', word);
  const url = backendUrl.replace(/\/$/, '') + '/stop-and-clear';
  const res = await fetch(url, { method: 'POST', body: form });
  return await res.json();
}
