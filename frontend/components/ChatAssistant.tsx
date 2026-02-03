"use client"

import * as React from "react"
import { MessageCircle, X, Send, Loader2, FileText, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { useAuth } from "@/context/AuthContext"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import * as XLSX from "xlsx"

// --- Helper: VegaChart ---
const VegaChart = ({ spec, title, data }: { spec: any, title?: string, data?: any }) => {
  const containerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (containerRef.current && spec) {
      const embedOptions = {
        actions: { export: true, source: false, compiled: false, editor: false },
        renderer: "svg" as const,
        mode: "vega-lite" as const
      }
      
      const finalSpec = JSON.parse(JSON.stringify(spec))
      
      // Inject data if provided and spec expects "table_data"
      let chartData = []
      if (data && data.headers && data.rows) {
          chartData = data.rows.map((row: any[]) => {
              const obj: any = {}
              data.headers.forEach((h: string, i: number) => {
                  obj[h] = row[i]
              })
              return obj
          })
      }
      
       if (finalSpec.data && finalSpec.data.name === "table_data") {
           delete finalSpec.data.name
           finalSpec.data.values = chartData
       }
      
      // Dynamic import to avoid SSR issues
      import("vega-embed").then((module) => {
          const embed = module.default
          // @ts-ignore
          embed(containerRef.current, finalSpec, embedOptions).catch(console.error)
      }).catch(console.error)
    }
  }, [spec, data])

  return (
    <div className="mb-4 bg-white p-2 rounded border border-gray-300">
      {title && <div className="font-semibold text-sm text-gray-700 mb-2 border-b pb-1">{title}</div>}
      <div ref={containerRef} className="w-full overflow-x-auto" />
    </div>
  )
}

// --- Helper: ChatMessage ---
interface MessageProps {
    message: {
        role: string;
        content: string;
        files?: any[];
        tables?: any[];
        charts?: any[];
        timestamp?: Date;
    };
    token?: string | null;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

const ChatMessage = ({ message, token }: MessageProps) => {
    const isUser = message.role === "user"
    const [exportingTable, setExportingTable] = React.useState<number | null>(null)

    const exportTableClient = async (table: any, idx: number) => {
        if (!table || !table.rows) return
        const maxRows = 20000
        if ((table.rows?.length || 0) > maxRows) {
            const ok = confirm(`Таблица содержит ${table.rows.length} строк. Экспорт может занять много времени. Продолжить?`)
            if (!ok) return
        }
        setExportingTable(idx)
        try {
            const aoa = [table.headers || [], ...(table.rows || [])]
            const ws = XLSX.utils.aoa_to_sheet(aoa)
            const wb = XLSX.utils.book_new()
            XLSX.utils.book_append_sheet(wb, ws, 'Sheet1')
            const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
            const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'export.xlsx'
            document.body.appendChild(a)
            a.click()
            a.remove()
            window.URL.revokeObjectURL(url)
        } catch (err) {
            console.error('Export failed', err)
            alert('Ошибка при экспорте таблицы')
        } finally {
            setExportingTable(null)
        }
    }

    return (
        <div className={`mb-4 flex min-w-0 ${isUser ? "justify-end" : "justify-start"}`}>
            <div
                className={cn(
                    "flex flex-col gap-2 rounded-lg px-3 py-2 text-sm max-w-[80%] min-w-0",
                    isUser ? "bg-blue-600 text-white mr-6" : "bg-gray-100 text-gray-900"
                )}
                style={{ overflowWrap: 'anywhere', minWidth: 0 }} // Ensure long text breaks and allow shrinking
            >
                {/* Content */}
                {isUser ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                ) : (
                    <div className="prose prose-sm max-w-none dark:prose-invert overflow-x-auto">
                        <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
                    </div>
                )}

                {/* Files */}
                {message.files && message.files.length > 0 && (
                    <div className={cn("mt-2 pt-2 border-t", isUser ? "border-blue-500" : "border-gray-200")}>
                        <div className="text-xs mb-1 opacity-70">
                            {isUser ? "Прикрепленные файлы:" : "Файлы:"}
                        </div>
                        {message.files.map((file, idx) => (
                             <div key={idx} className="flex items-center gap-1 text-xs mb-1">
                                <FileText className="h-3 w-3" />
                                <span className="truncate max-w-[150px]">{file.name}</span>
                                {file.download_url && (
                                     <button
                                     onClick={async (e) => {
                                         e.preventDefault()
                                         const url = `${BACKEND_URL}${file.download_url}`
                                         try {
                                             const headers: any = {}
                                             if (token) headers['Authorization'] = `Bearer ${token}`
                                             
                                             const resp = await fetch(url, { headers })
                                             if (!resp.ok) throw new Error("Download failed")
                                             
                                             const blob = await resp.blob()
                                             const blobUrl = window.URL.createObjectURL(blob)
                                             const a = document.createElement('a')
                                             a.href = blobUrl
                                             a.download = file.name
                                             document.body.appendChild(a)
                                             a.click()
                                             a.remove()
                                             window.URL.revokeObjectURL(blobUrl)
                                         } catch (err) {
                                             console.error(err)
                                             window.open(url, '_blank')
                                         }
                                     }}
                                     className="ml-1 hover:underline flex items-center"
                                 >
                                     <Download className="h-3 w-3 mr-1" />
                                 </button>
                                )}
                             </div>
                        ))}
                    </div>
                )}

                {/* Tables */}
                {message.tables && message.tables.length > 0 && (
                    <div className={cn("mt-2 pt-2 border-t", isUser ? "border-blue-500" : "border-gray-200")}>
                        <div className="text-xs mb-1 opacity-70">Таблицы:</div>
                        {message.tables.map((table, idx) => (
                            <div key={idx} className="mb-2 bg-white rounded border border-gray-200 overflow-hidden text-gray-800">
                                <div className="bg-gray-50 px-2 py-1 flex justify-between items-center border-b border-gray-200">
                                    <span className="font-medium text-xs truncate max-w-[150px]">{table.title || `Таблица ${idx+1}`}</span>
                                    {table.download_url ? (
                                        <button
                                            onClick={async () => {
                                                const url = `${BACKEND_URL}${table.download_url}`
                                                try {
                                                    const headers: any = {}
                                                    if (token) headers['Authorization'] = `Bearer ${token}`
                                                    const resp = await fetch(url, { headers })
                                                    if(!resp.ok) throw new Error("Download failed")
                                                    const blob = await resp.blob()
                                                    const blobUrl = window.URL.createObjectURL(blob)
                                                    const a = document.createElement('a')
                                                    a.href = blobUrl
                                                    a.download = `${table.title || 'export'}.xlsx`
                                                    document.body.appendChild(a)
                                                    a.click()
                                                    a.remove()
                                                } catch(e) {
                                                    window.open(url, '_blank')
                                                }
                                            }}
                                            className="text-[10px] bg-green-500 hover:bg-green-600 text-white px-1.5 py-0.5 rounded"
                                        >
                                            Excel
                                        </button>
                                    ) : (
                                        <button
                                            onClick={() => exportTableClient(table, idx)}
                                            disabled={exportingTable === idx}
                                            className="text-[10px] bg-green-500 hover:bg-green-600 text-white px-1.5 py-0.5 rounded"
                                        >
                                            {exportingTable === idx ? 'Подготовка...' : 'Excel'}
                                        </button>
                                    )}
                                </div>
                                <div className="overflow-x-auto max-h-40">
                                    <table className="w-full text-xs text-left">
                                        <thead className="text-xs text-gray-700 bg-gray-50 uppercase sticky top-0">
                                            <tr>
                                                {table.headers?.map((h: string, i: number) => (
                                                    <th key={i} className="px-2 py-1 border-r last:border-r-0 border-gray-200">{h}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {table.rows?.slice(0, 10).map((row: any[], i: number) => ( // limit preview rows
                                                <tr key={i} className="border-b last:border-b-0 hover:bg-gray-50">
                                                    {row.map((cell, j) => (
                                                        <td key={j} className="px-2 py-1 border-r last:border-r-0 border-gray-200 whitespace-nowrap">
                                                            {typeof cell === 'object' ? JSON.stringify(cell) : String(cell)}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                            {table.rows?.length > 10 && (
                                                <tr><td colSpan={table.headers?.length} className="px-2 py-1 text-center italic text-gray-500">... {table.rows.length - 10} more rows ...</td></tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Charts */}
                {message.charts && message.charts.length > 0 && (
                    <div className={cn("mt-2 pt-2 border-t", isUser ? "border-blue-500" : "border-gray-200")}>
                        <div className="text-xs mb-1 opacity-70">Графики:</div>
                        {message.charts.map((chart, idx) => (
                            <VegaChart 
                                key={idx} 
                                spec={chart.spec} 
                                title={chart.title} 
                                data={message.tables?.[0]} 
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}


export function ChatAssistant() {
  const [isOpen, setIsOpen] = React.useState(false)
  const [messages, setMessages] = React.useState<any[]>([
    { role: "assistant", content: "Привет! Я ваш ИИ-ассистент. Чем могу помочь?" }
  ])
  const [input, setInput] = React.useState("")
  const [isLoading, setIsLoading] = React.useState(false)
  const { session } = useAuth()
  const scrollAreaRef = React.useRef<HTMLDivElement>(null)

  // Load messages from localStorage on mount
  React.useEffect(() => {
    const saved = localStorage.getItem("chat_messages")
    if (saved) {
        try {
            setMessages(JSON.parse(saved))
        } catch (e) {
            console.error("Failed to load chat history", e)
        }
    }
  }, [])

  // Save messages to localStorage whenever they update
  React.useEffect(() => {
    if (messages.length > 0) {
        localStorage.setItem("chat_messages", JSON.stringify(messages))
    }
  }, [messages])

  // Scroll to bottom when messages change
  React.useEffect(() => {
    if (isOpen && scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
        if (scrollContainer) {
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }
    }
  }, [messages, isOpen, isLoading])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return
    
    const userMessage = { role: "user", content: input, timestamp: new Date() }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput("")
    setIsLoading(true)

    try {
        const formData = new FormData()
        formData.append('role', 'user')
        formData.append('content', userMessage.content)
        
        // Prepare history
        // Mapping simple content for validation context, backend might parse better if structure matches
        const history = messages.map(m => ({
            role: m.role,
            content: m.content
        }))
        formData.append('previous_messages', JSON.stringify(history))

        const token = session?.access_token || ""
        
        // Build API URL: NEXT_PUBLIC_CHAT_API_URL should be like "http://.../api/v1"
        const apiBase = process.env.NEXT_PUBLIC_CHAT_API_URL || 'http://localhost:8000/api/v1'
        const apiUrl = `${apiBase}/temporary/chat`

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        })

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`)
        }

        const data = await response.json()
        
        if (data.assistant_message) {
            setMessages(prev => [...prev, { 
                role: "assistant", 
                content: data.assistant_message.content,
                files: data.assistant_message.files,
                tables: data.assistant_message.tables,
                charts: data.assistant_message.charts,
                timestamp: new Date()
            }])
        }

    } catch (error) {
        console.error("Chat error:", error)
        setMessages(prev => [...prev, { 
            role: "assistant", 
            content: "Извините, произошла ошибка при соединении с сервером." 
        }])
    } finally {
        setIsLoading(false)
    }
  }

  return (
    <>
      {!isOpen && (
        <div 
            onClick={() => setIsOpen(true)}
            style={{
                position: 'fixed',
                bottom: '30px',
                right: '30px',
                zIndex: 9999,
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                backgroundColor: '#2563eb',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
                transition: 'all 0.2s ease-in-out'
            }}
            onMouseEnter={(e: any) => e.currentTarget.style.transform = 'scale(1.1)'}
            onMouseLeave={(e: any) => e.currentTarget.style.transform = 'scale(1)'}
        >
            <MessageCircle size={32} strokeWidth={2} />
        </div>
      )}

      {isOpen && (
        <div style={{
            position: 'fixed',
            bottom: '24px',
            right: '24px',
            zIndex: 9999,
            width: '800px',
            maxWidth: '90vw',
            height: '600px',
            maxHeight: '80vh',
            boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
            borderRadius: '12px',
            backgroundColor: 'white',
            display: 'flex',
            flexDirection: 'column',
            border: '1px solid #e2e8f0',
            overflow: 'visible' // allow tooltips to overflow the chat container
        }}>
            <div className="flex flex-row items-center justify-between p-3 border-b bg-muted/20">
                <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center">
                        <MessageCircle className="h-5 w-5 text-white" />
                    </div>
                    <span className="text-base font-medium">Ассистент</span>
                </div>
                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full" onClick={() => setIsOpen(false)}>
                    <X className="h-4 w-4" />
                    <span className="sr-only">Close</span>
                </Button>
            </div>
          
            <div className="flex-1 overflow-hidden p-0 bg-background relative">
                <ScrollArea ref={scrollAreaRef} className="h-full p-4" style={{ paddingRight: 24 }}>
                    <div className="flex flex-col">
                        {messages.map((msg, i) => (
                            <ChatMessage key={i} message={msg} token={session?.access_token} />
                        ))}
                        {isLoading && (
                            <div className="flex w-fit max-w-[90%] flex-col gap-2 rounded-lg px-3 py-2 text-sm bg-gray-100 text-gray-900 border border-gray-200">
                                <div className="flex items-center gap-2 text-gray-500">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    <span>Печатает...</span>
                                </div>
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </div>

            <div className="p-3 pt-2 bg-muted/20 border-t">
                <form
                className="flex w-full items-center space-x-2"
                onSubmit={(e) => {
                    e.preventDefault()
                    handleSend()
                }}
                >
                <Input
                    placeholder="Введите сообщение..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="bg-background"
                    disabled={isLoading}
                />
                <Button type="submit" size="icon" className="bg-blue-600 hover:bg-blue-700" disabled={isLoading || !input.trim()}>
                    <Send className="h-4 w-4 text-white" />
                    <span className="sr-only">Send</span>
                </Button>
                </form>
            </div>

            {/* Ensure Vega tooltips render above the chat modal */}
            <style dangerouslySetInnerHTML={{__html: `
              /* Vega tooltip element id used by vega-embed */
              #vg-tooltip-element, .vega-tooltip {
                position: fixed !important;
                z-index: 10000 !important; /* higher than chat modal 9999 */
                pointer-events: auto !important;
                max-width: 90vw !important;
                white-space: normal !important;
                font-size: 12px !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
              }
            `}} />

        </div>
      )}
    </>
  )
}
