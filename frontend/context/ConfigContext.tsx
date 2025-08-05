"use client"

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react"
import { parseYamlConfig } from "../components/utils/parseYaml"

interface ConfigContextType {
  config: any
  loading: boolean
}

const ConfigContext = createContext<ConfigContextType>({ config: null, loading: true })

export function ConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch((process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000") + "/config")
      .then((res) => res.text())
      .then((text) => {
        setConfig(parseYamlConfig(text))
        setLoading(false)
      })
  }, [])

  return (
    <ConfigContext.Provider value={{ config, loading }}>
      {children}
    </ConfigContext.Provider>
  )
}

export function useConfig() {
  return useContext(ConfigContext)
}
