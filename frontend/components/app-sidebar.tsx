"use client"

import { useState } from "react"
import { useAuth } from "@/context/AuthContext"
import { useRouter } from "next/navigation"
import { TechAccessPanel } from "./TechAccessPanel"
import { motion, AnimatePresence } from "framer-motion"
import { Activity, Settings, ChevronDown, ChevronUp, Database, Download, GraduationCap, BarChart3, FileDown, Calculator } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"

interface AppSidebarProps {
  currentPage: "training" | "analysis" | "export" | "analytics" | "tax-forecast" | "tax-export"
  onPageChange: (page: "training" | "analysis" | "export" | "analytics" | "tax-forecast" | "tax-export") => void
}

export function AppSidebar({ currentPage, onPageChange }: AppSidebarProps) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [isAdmin, setIsAdmin] = useState(false)
  const [open, setOpen] = useState(false)

  const handleLogin = () => {
    if (username === "admin" && password === "admin") {
      setIsAdmin(true)
    } else {
      setIsAdmin(false)
    }
  }

  const handleConfigLoad = () => {
  }

  const handleLogDownload = () => {
  }

  const { logout } = useAuth()
  const router = useRouter()

  const handleLogout = async () => {
    await logout()
    router.replace("/login")
  }

  return (
    <Sidebar className="border-r border-gray-200 bg-white shadow-lg">
      <SidebarHeader className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Базовое моделирование</h2>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-gray-700 font-semibold">Прогноз ЧОК</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("training")}
                  isActive={currentPage === "training"}
                  className="w-full justify-start"
                >
                  <GraduationCap className="w-4 h-4" />
                  <span>Запуск прогноза</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("analytics")}
                  isActive={currentPage === "analytics"}
                  className="w-full justify-start"
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Общая аналитика</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("analysis")}
                  isActive={currentPage === "analysis"}
                  className="w-full justify-start"
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Аналитика по статьям</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("export")}
                  isActive={currentPage === "export"}
                  className="w-full justify-start"
                >
                  <FileDown className="w-4 h-4" />
                  <span>Выгрузка и корректировки</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-gray-700 font-semibold">Налоги</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("tax-forecast")}
                  isActive={currentPage === "tax-forecast"}
                  className="w-full justify-start"
                >
                  <Calculator className="w-4 h-4" />
                  <span>Прогноз налогов</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("tax-export")}
                  isActive={currentPage === "tax-export"}
                  className="w-full justify-start"
                >
                  <FileDown className="w-4 h-4" />
                  <span>Выгрузка</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupContent>
            <Collapsible open={open} onOpenChange={setOpen}>
              <CollapsibleTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700 font-semibold"
                >
                  <div className="flex items-center gap-2">
                    <Settings className="w-4 h-4" />
                    Технический доступ
                  </div>
                  {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </Button>
              </CollapsibleTrigger>

              <CollapsibleContent>
                <TechAccessPanel />
              </CollapsibleContent>
            </Collapsible>
          </SidebarGroupContent>
        </SidebarGroup>
        {/* Кнопка выйти снизу */}
        <div className="mt-auto p-4">
          <Button
            variant="outline"
            className="w-full border-gray-300 text-gray-700 hover:bg-gray-50 bg-transparent"
            onClick={handleLogout}
          >
            Выйти
          </Button>
        </div>
      </SidebarContent>
    </Sidebar>
  )
}