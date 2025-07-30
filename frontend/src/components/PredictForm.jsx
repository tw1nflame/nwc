import React, { useState } from 'react';

export default function PredictForm({ config, onSubmit }) {
  const [pipeline, setPipeline] = useState('BASE+');
  const [selectedItems, setSelectedItems] = useState(Object.keys(config?.Статья || {}));
  const [date, setDate] = useState('2025-01-01');
  const [dataFile, setDataFile] = useState(null);
  const [prevResultsFile, setPrevResultsFile] = useState(null);

  const handleItemsChange = e => {
    const options = Array.from(e.target.options);
    setSelectedItems(options.filter(o => o.selected).map(o => o.value));
  };

  const handleSubmit = e => {
    e.preventDefault();
    if (!dataFile || !prevResultsFile) return;
    onSubmit({ pipeline, selectedItems, date, dataFile, prevResultsFile });
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-xl mx-auto bg-white p-6 rounded shadow">
      <div className="mb-4">
        <label className="block mb-1 font-medium">Выберите алгоритм прогноза:</label>
        <label className="mr-4"><input type="radio" value="BASE" checked={pipeline === 'BASE'} onChange={() => setPipeline('BASE')} className="mr-1" /> BASE</label>
        <label><input type="radio" value="BASE+" checked={pipeline === 'BASE+'} onChange={() => setPipeline('BASE+')} className="mr-1" /> BASE+</label>
      </div>
      <div className="mb-4">
        <label className="block mb-1 font-medium">Выберите статьи для прогноза:</label>
        <select multiple value={selectedItems} onChange={handleItemsChange} className="w-full h-24 border rounded">
          {Object.keys(config?.Статья || {}).map(key => (
            <option key={key} value={key}>{key}</option>
          ))}
        </select>
      </div>
      <div className="mb-4">
        <label className="block mb-1 font-medium">Выберите месяц и год предикта:</label>
        <input type="month" value={date.slice(0,7)} onChange={e => setDate(e.target.value + '-01')} className="border rounded px-2 py-1" />
      </div>
      <div className="mb-4">
        <label className="block mb-1 font-medium">Загрузите файл с данными (ЧОК исторические):</label>
        <input type="file" accept=".xlsm,.xlsx" onChange={e => setDataFile(e.target.files[0])} className="block" />
      </div>
      <div className="mb-4">
        <label className="block mb-1 font-medium">Загрузите файл с предыдущими прогнозами:</label>
        <input type="file" accept=".xlsm,.xlsx" onChange={e => setPrevResultsFile(e.target.files[0])} className="block" />
      </div>
      <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded">Запустить расчёт</button>
    </form>
  );
}
