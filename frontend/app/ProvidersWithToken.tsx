"use client"

import React from "react";
import { useAuth } from "../context/AuthContext";
import { ConfigProvider } from "../context/ConfigContext";
import { ExcelProvider } from "../context/ExcelContext";

export default function ProvidersWithToken({ children }: { children: React.ReactNode }) {
  const { session } = useAuth();
  const accessToken = session?.access_token;
  return (
    <ConfigProvider accessToken={accessToken}>
      <ExcelProvider>
        {children}
      </ExcelProvider>
    </ConfigProvider>
  );
}
