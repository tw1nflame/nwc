"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Activity, Settings, ChevronDown, ChevronUp, Database, Download, GraduationCap, BarChart3 } from "lucide-react"
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
  currentPage: "training" | "analysis"
  onPageChange: (page: "training" | "analysis") => void
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
    console.log("Загрузка конфига...")
  }

  const handleLogDownload = () => {
    console.log("Скачивание логов...")
  }

  return (
    <Sidebar className="border-r border-gray-200 bg-white shadow-lg">
      <SidebarHeader className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Прогнозирование</h2>
            <p className="text-sm text-gray-500">Система анализа</p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-gray-700 font-semibold">Основные разделы</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("training")}
                  isActive={currentPage === "training"}
                  className="w-full justify-start"
                >
                  <GraduationCap className="w-4 h-4" />
                  <span>Обучение</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onPageChange("analysis")}
                  isActive={currentPage === "analysis"}
                  className="w-full justify-start"
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Анализ</span>
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

              <CollapsibleContent className="space-y-4 mt-4">
                <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                  <div>
                    <Label htmlFor="username" className="text-gray-700 font-medium">
                      Имя пользователя
                    </Label>
                    <Input
                      id="username"
                      type="text"
                      placeholder="username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      className="bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <Label htmlFor="password" className="text-gray-700 font-medium">
                      Пароль
                    </Label>
                    <Input
                      id="password"
                      type="password"
                      placeholder="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>

                  <Button
                    onClick={handleLogin}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold"
                  >
                    Войти
                  </Button>

                  <AnimatePresence>
                    {isAdmin && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-2 pt-4 border-t border-gray-200"
                      >
                        <Button
                          onClick={handleConfigLoad}
                          variant="outline"
                          className="w-full bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100"
                        >
                          <Database className="w-4 h-4 mr-2" />
                          Загрузить конфиг
                        </Button>
                        <Button
                          onClick={handleLogDownload}
                          variant="outline"
                          className="w-full bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Скачать логи
                        </Button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </CollapsibleContent>
            </Collapsible>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}
