import React, { useState } from 'react';

export default function Sidebar({ onConfigLoad, onLogDownload }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);

  const handleLogin = () => {
    if (username === 'admin' && password === 'admin') {
      setIsAdmin(true);
      onConfigLoad && onConfigLoad();
    } else {
      setIsAdmin(false);
    }
  };

  return (
    <aside className="w-64 p-4 bg-white border-r border-gray-200 min-h-screen flex flex-col">
      <div className="mb-6">
        <h4 className="font-semibold mb-2">tech user:</h4>
        <input
          type="text"
          placeholder="username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          className="w-full mb-2 px-2 py-1 border rounded"
        />
        <input
          type="password"
          placeholder="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          className="w-full mb-2 px-2 py-1 border rounded"
        />
        <button onClick={handleLogin} className="w-full bg-blue-600 text-white py-1 rounded">Войти</button>
      </div>
      {isAdmin && (
        <>
          <button onClick={onConfigLoad} className="w-full bg-green-600 text-white py-1 rounded mb-2">Загрузить конфиг</button>
          <button onClick={onLogDownload} className="w-full bg-gray-600 text-white py-1 rounded">Скачать логи</button>
        </>
      )}
    </aside>
  );
}
