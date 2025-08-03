"use client"

import { AppSidebar } from "@/components/app-sidebar"
import { TrainingPage } from "@/components/training-page"
import { AnalysisPage } from "@/components/analysis-page"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { useState } from "react"

export default function App() {
  const [currentPage, setCurrentPage] = useState<"training" | "analysis">("training")

  return (
    <SidebarProvider>
      <AppSidebar currentPage={currentPage} onPageChange={setCurrentPage} />
      <SidebarInset className="bg-gradient-to-br from-gray-50 via-white to-gray-100">
        {currentPage === "training" ? <TrainingPage /> : <AnalysisPage />}
      </SidebarInset>
    </SidebarProvider>
  )
}
