import React, { useEffect, useState } from 'react';
import { parseYamlConfig } from './utils/parseYaml';
import { sendTrainRequest } from './utils/api';
import { fetchTrainStatus } from './utils/trainStatus';
import { stopTrainTask } from './utils/stopTrain';

function Sidebar({ onConfigLoad, onLogDownload }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [open, setOpen] = useState(false);

  const handleLogin = () => {
    if (username === 'admin' && password === 'admin') {
      setIsAdmin(true);
      onConfigLoad && onConfigLoad();
    } else {
      setIsAdmin(false);
    }
  };

  return (
    <aside className="w-72 p-6 bg-white border-r border-gray-200 min-h-screen flex flex-col shadow">
      <div className="mb-8">
        <button
          className="w-full flex items-center justify-between px-3 py-2 border rounded font-semibold bg-gray-100 hover:bg-gray-200 mb-3"
          onClick={() => setOpen(o => !o)}
        >
          <span>Технический доступ</span>
          <span className="ml-2">{open ? '▲' : '▼'}</span>
        </button>
        {open && (
          <div>
            <input
              type="text"
              placeholder="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="w-full mb-3 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
            <input
              type="password"
              placeholder="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full mb-3 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
            <button
              onClick={handleLogin}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded font-semibold transition"
            >
              Войти
            </button>
            {isAdmin && (
              <div className="space-y-2 mt-4">
                <button
                  onClick={onConfigLoad}
                  className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded font-semibold transition"
                >
                  Загрузить конфиг
                </button>
                <button
                  onClick={onLogDownload}
                  className="w-full bg-gray-600 hover:bg-gray-700 text-white py-2 rounded font-semibold transition"
                >
                  Скачать логи
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

function FileInput({ label, accept, file, onFileChange, onRemove }) {
  const inputRef = React.useRef();
  return (
    <div>
      <label className="block mb-2 font-semibold text-gray-700">{label}</label>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current.click()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold shadow transition border border-blue-700/30"
        >
          {file ? 'Выбрать другой файл' : 'Выберите файл'}
        </button>
        {file && (
          <span className="flex items-center bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
            {file.name}
            <button
              type="button"
              className="ml-2 text-blue-500 hover:text-red-500 text-lg font-bold"
              onClick={onRemove}
              title="Удалить файл"
            >×</button>
          </span>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={e => onFileChange(e.target.files[0])}
      />
    </div>
  );
}

function PredictForm({ config, onSubmit, trainStatus }) {
  const [pipeline, setPipeline] = useState('BASE+');
  const [selectedItems, setSelectedItems] = useState([]);
  const [date, setDate] = useState('2025-01-01');
  const [dataFile, setDataFile] = useState(null);
  const [prevResultsFile, setPrevResultsFile] = useState(null);
  const [selectValue, setSelectValue] = useState('');

  useEffect(() => {
    if (config && config['Статья']) {
      setSelectedItems(Object.keys(config['Статья']));
    }
  }, [config]);

  const allItems = config && config['Статья'] ? Object.keys(config['Статья']) : [];
  const availableItems = allItems.filter(item => !selectedItems.includes(item));

  const handleAddItem = e => {
    const value = e.target.value;
    if (value && !selectedItems.includes(value)) {
      setSelectedItems([...selectedItems, value]);
      setSelectValue('');
    }
  };

  const handleRemove = (item) => {
    setSelectedItems(selectedItems.filter(i => i !== item));
  };

  const handleSubmit = async e => {
    e.preventDefault();
    if (!dataFile || !prevResultsFile) return;
    // Передаём данные наверх, не отправляем запрос здесь
    onSubmit && onSubmit({ pipeline, selectedItems, date, dataFile, prevResultsFile });
  };

  const handleStop = async () => {
    const resp = await stopTrainTask(); // без taskId
    alert(resp.status === 'revoked' ? 'Обучение остановлено' : (resp.error || JSON.stringify(resp)));
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-5xl mx-auto space-y-6">
      <div>
        <label className="block mb-2 font-semibold text-gray-700">Алгоритм прогноза .venv\scripts\activate & cd backend/app & uv run uvicorn main:app --reload</label>
        <div className="flex gap-6">
          <label className="flex items-center cursor-pointer">
            <input type="radio" value="BASE" checked={pipeline === 'BASE'} onChange={() => setPipeline('BASE')} className="accent-blue-600" />
            <span className="ml-2">BASE</span>
          </label>
          <label className="flex items-center cursor-pointer">
            <input type="radio" value="BASE+" checked={pipeline === 'BASE+'} onChange={() => setPipeline('BASE+')} className="accent-blue-600" />
            <span className="ml-2">BASE+</span>
          </label>
        </div>
      </div>
      <div>
        <label className="block mb-2 font-semibold text-gray-700">Статьи для прогноза .venv\scripts\activate & cd backend/app & celery -A tasks worker --loglevel=info  --pool=solo</label>
        <div className="flex flex-wrap gap-2 mb-2">
          {selectedItems.map(item => (
            <span key={item} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full flex items-center text-sm">
              {item}
              <button type="button" className="ml-2 text-blue-500 hover:text-red-500" onClick={() => handleRemove(item)} title="Убрать">×</button>
            </span>
          ))}
        </div>
        <select value={selectValue} onChange={handleAddItem} className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200">
          <option value="">Добавить статью...</option>
          {availableItems.map(key => (
            <option key={key} value={key}>{key}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="block mb-2 font-semibold text-gray-700">Месяц и год предикта .venv\scripts\activate & cd backend/app & set FLOWER_UNAUTHENTICATED_API=1 & celery -A tasks flower --port=5555 --broker=redis://localhost:6379/0</label>
        <input type="month" value={date.slice(0, 7)} onChange={e => setDate(e.target.value + '-01')} className="border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200" />
      </div>
      <div>
        <FileInput
          label="Файл с данными (ЧОК исторические)"
          accept=".xlsm,.xlsx"
          file={dataFile}
          onFileChange={setDataFile}
          onRemove={() => setDataFile(null)}
        />
      </div>
      <div>
        <FileInput
          label="Файл с предыдущими прогнозами"
          accept=".xlsm,.xlsx"
          file={prevResultsFile}
          onFileChange={setPrevResultsFile}
          onRemove={() => setPrevResultsFile(null)}
        />
      </div>
      <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg text-lg transition mb-2">Запустить расчёт</button>
      <div className="mt-2 flex flex-col items-center">
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-base shadow-sm transition-colors
          ${trainStatus?.status === 'running' || trainStatus?.status === 'pending' ? 'bg-blue-100 text-blue-800 border border-blue-300' : ''}
          ${trainStatus?.status === 'done' ? 'bg-green-100 text-green-800 border border-green-300' : ''}
          ${trainStatus?.status === 'error' ? 'bg-red-100 text-red-800 border border-red-300' : ''}
          ${trainStatus?.status === 'revoked' ? 'bg-yellow-100 text-yellow-800 border border-yellow-300' : ''}
          ${(trainStatus?.status === 'idle' || trainStatus?.status === 'not_found' || !trainStatus) ? 'bg-gray-100 text-gray-700 border border-gray-300' : ''}
        `}>
          <span className="inline-block w-2 h-2 rounded-full mr-2"
            style={{background:
              trainStatus?.status === 'running' || trainStatus?.status === 'pending' ? '#2563eb' :
              trainStatus?.status === 'done' ? '#22c55e' :
              trainStatus?.status === 'error' ? '#ef4444' :
              trainStatus?.status === 'revoked' ? '#eab308' : '#64748b'}}></span>
          {(trainStatus?.status === 'running' || trainStatus?.status === 'pending') && 'Обучение запущено...'}
          {trainStatus?.status === 'done' && 'Обучение завершено!'}
          {trainStatus?.status === 'error' && 'Ошибка при обучении'}
          {trainStatus?.status === 'revoked' && 'Обучение было остановлено'}
          {(trainStatus?.status === 'idle' || trainStatus?.status === 'not_found' || !trainStatus) && 'Ожидание запуска'}
          {trainStatus?.result_file && trainStatus?.status === 'done' && (
            <span className="ml-4 text-blue-700">Файл результата: {trainStatus.result_file}</span>
          )}
        </div>
        {trainStatus?.status === 'running' && (
          <button type="button" onClick={handleStop} className="mt-3 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition">
            Остановить обучение
          </button>
        )}
      </div>
    </form>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [result, setResult] = useState(null);
  const [trainStatus, setTrainStatus] = useState({ status: 'idle' });
  const [polling, setPolling] = useState(false);
  const [taskId, setTaskId] = useState(null);

  useEffect(() => {
    fetch('/config_refined.yaml')
      .then(res => res.text())
      .then(text => setConfig(parseYamlConfig(text)));
  }, []);

  // Проверяем статус при загрузке страницы
  useEffect(() => {
    let ignore = false;
    async function checkStatus() {
      const status = await fetchTrainStatus();
      console.log('trainStatus (init):', status);
      setTrainStatus(status && status.status ? status : { status: 'idle' });
      if (status.status === 'running') setPolling(true);
    }
    checkStatus();
    return () => { ignore = true; };
  }, []);

  // Poll train status только если polling=true
  useEffect(() => {
    if (!polling) return;
    const interval = setInterval(async () => {
      const status = await fetchTrainStatus();
      console.log('trainStatus (poll):', status);
      setTrainStatus(status && status.status ? status : { status: 'idle' });
      if (status.status === 'done' || status.status === 'error' || status.status === 'not_found' || !status.status) {
        setPolling(false);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [polling]);

  const handleConfigLoad = async () => {
    const resp = await fetch('/config_refined.yaml');
    const text = await resp.text();
    setConfig(parseYamlConfig(text));
  };

  const handleLogDownload = () => {
    window.open('/logs', '_blank');
  };

  const handleTrainSubmit = async ({ pipeline, selectedItems, date, dataFile, prevResultsFile }) => {
    setTrainStatus({ status: 'running' });
    setPolling(true);
    // Отправляем объект, а не FormData!
    const resp = await sendTrainRequest({ pipeline, selectedItems, date, dataFile, prevResultsFile });
    setTaskId(resp.task_id);
  };

  return (
    <div className="flex">
      <Sidebar onConfigLoad={handleConfigLoad} onLogDownload={handleLogDownload} />
      <main className="flex-1 p-8">
        <h1 className="text-3xl font-bold mb-6">Прогнозирование</h1>
        {config && (
          <PredictForm config={config} onSubmit={handleTrainSubmit} trainStatus={trainStatus} />
        )}
      </main>
    </div>
  );
}
