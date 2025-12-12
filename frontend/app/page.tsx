"use client"

import { AppSidebar } from "@/components/app-sidebar"
import TrainingPage from "@/components/training-page"
import { AnalysisPage } from "@/components/analysis-page"
import { ExportPage } from "@/components/export-page"
import { AnalyticsPage } from "@/components/analytics-page"
import { TaxForecastPage } from "@/components/tax-forecast-page"
import { TaxExportPage } from "@/components/tax-export-page"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { useState } from "react"
import RequireAuth from "@/components/RequireAuth"
import { AnalyticsProvider } from "../context/AnalyticsContext"

export default function App() {
  const [currentPage, setCurrentPage] = useState<"training" | "analysis" | "export" | "analytics" | "tax-forecast" | "tax-export">("training")

  return (
    <RequireAuth>
      <AnalyticsProvider>
        <SidebarProvider>
          <AppSidebar currentPage={currentPage} onPageChange={setCurrentPage} />
          <SidebarInset className="bg-gradient-to-br from-gray-50 via-white to-gray-100" style={{ width: '100%', maxWidth: 'calc(100vw - 240px)', overflow: 'hidden' }}>
            {currentPage === "training" && <TrainingPage />}
            {currentPage === "analytics" && <AnalyticsPage />}
            {currentPage === "analysis" && <AnalysisPage />}
            {currentPage === "export" && <ExportPage />}
            {currentPage === "tax-forecast" && <TaxForecastPage />}
            {currentPage === "tax-export" && <TaxExportPage />}
          </SidebarInset>
        </SidebarProvider>
      </AnalyticsProvider>
    </RequireAuth>
  )
}
