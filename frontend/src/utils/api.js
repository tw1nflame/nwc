export async function sendTrainRequest({ pipeline, selectedItems, date, dataFile, prevResultsFile }) {
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const formData = new FormData();
  formData.append('pipeline', pipeline);
  formData.append('items', JSON.stringify(selectedItems));
  formData.append('date', date);
  formData.append('data_file', dataFile);
  formData.append('prev_results_file', prevResultsFile);

  let url = backendUrl.replace(/\/$/, '') + '/train/';
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      body: formData
    });
  } catch (e) {
    return { error: 'Network error', details: e.toString() };
  }
  let data;
  try {
    data = await response.json();
  } catch {
    data = { error: 'Invalid JSON response', status: response.status };
  }
  if (!response.ok) {
    data.error = data.error || `HTTP error: ${response.status}`;
  }
  return data;
}
