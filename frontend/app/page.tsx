"use client"

import { AppSidebar } from "@/components/app-sidebar"
import TrainingPage from "@/components/training-page"
import { AnalysisPage } from "@/components/analysis-page"
import { ExportPage } from "@/components/export-page"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { useState } from "react"
import RequireAuth from "@/components/RequireAuth"

export default function App() {
  const [currentPage, setCurrentPage] = useState<"training" | "analysis" | "export">("training")

  const renderCurrentPage = () => {
    switch (currentPage) {
      case "training":
        return <TrainingPage />
      case "analysis":
        return <AnalysisPage />
      case "export":
        return <ExportPage />
      default:
        return <TrainingPage />
    }
  }

  return (
    <RequireAuth>
      <SidebarProvider>
        <AppSidebar currentPage={currentPage} onPageChange={setCurrentPage} />
        <SidebarInset className="bg-gradient-to-br from-gray-50 via-white to-gray-100" style={{ width: '100%', maxWidth: 'calc(100vw - 240px)', overflow: 'hidden' }}>
          {renderCurrentPage()}
        </SidebarInset>
      </SidebarProvider>
    </RequireAuth>
  )
}
