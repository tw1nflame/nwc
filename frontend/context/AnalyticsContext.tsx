import React, { createContext, useContext, useState } from "react";


export interface AnalyticsContextType {
  bulletChartData: any[];
  setBulletChartData: (data: any[]) => void;
  exchangeRates: any[];
  setExchangeRates: (rates: any[]) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  startMonth: string | null;
  setStartMonth: (month: string | null) => void;
  endMonth: string | null;
  setEndMonth: (month: string | null) => void;
  currency: 'RUB' | 'USD';
  setCurrency: (currency: 'RUB' | 'USD') => void;
}

const AnalyticsContext = createContext<AnalyticsContextType | undefined>(undefined);

export function useAnalyticsContext() {
  const ctx = useContext(AnalyticsContext);
  if (!ctx) throw new Error("useAnalyticsContext must be used within AnalyticsProvider");
  return ctx;
}

export function AnalyticsProvider({ children }: { children: React.ReactNode }) {
  const [bulletChartData, setBulletChartData] = useState<any[]>([]);
  const [exchangeRates, setExchangeRates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [startMonth, setStartMonth] = useState<string | null>(null);
  const [endMonth, setEndMonth] = useState<string | null>(null);
  const [currency, setCurrency] = useState<'RUB' | 'USD'>('RUB');
  return (
    <AnalyticsContext.Provider value={{ bulletChartData, setBulletChartData, exchangeRates, setExchangeRates, loading, setLoading, startMonth, setStartMonth, endMonth, setEndMonth, currency, setCurrency }}>
      {children}
    </AnalyticsContext.Provider>
  );
}
