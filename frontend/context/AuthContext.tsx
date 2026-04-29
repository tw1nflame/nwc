"use client"

import React, { createContext, useContext, useEffect, useMemo, useRef, useState, ReactNode } from "react"
import Keycloak from "keycloak-js"

type AppUser = {
  email?: string
}

type AppSession = {
  access_token: string
  user?: AppUser
}

interface AuthContextType {
  user: AppUser | null
  session: AppSession | null
  login: (_email?: string, _password?: string) => Promise<{ error: string | null }>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AppUser | null>(null)
  const [session, setSession] = useState<AppSession | null>(null)
  const [loading, setLoading] = useState(true)

  const keycloak = useMemo(() => {
    const url = process.env.NEXT_PUBLIC_KEYCLOAK_URL
    const realm = process.env.NEXT_PUBLIC_KEYCLOAK_REALM
    const clientId = process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID

    if (!url || !realm || !clientId) {
      // Fail fast: without these envs auth will never work.
      throw new Error(
        "Missing Keycloak env vars: NEXT_PUBLIC_KEYCLOAK_URL, NEXT_PUBLIC_KEYCLOAK_REALM, NEXT_PUBLIC_KEYCLOAK_CLIENT_ID",
      )
    }

    return new Keycloak({
      url,
      realm,
      clientId,
    })
  }, [])

  const initializedRef = useRef(false)
  const refreshIntervalRef = useRef<number | null>(null)

  const syncFromKeycloak = () => {
    const token = keycloak.token
    if (!token) {
      setSession(null)
      setUser(null)
      return
    }
    const parsed: any = keycloak.tokenParsed || {}
    const email: string | undefined = parsed.email
    const nextUser: AppUser = { email }
    const nextSession: AppSession = { access_token: token, user: nextUser }
    setSession(nextSession)
    setUser(nextUser)
  }

  useEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true

    let cancelled = false

    ;(async () => {
      try {
        // We use check-sso to avoid auto-redirect loops.
        // The app itself can redirect unauthenticated users to /login.
        const authenticated = await keycloak.init({
          onLoad: "check-sso",
          pkceMethod: "S256",
          // iframe checks can be flaky in some environments; keep it off for simplicity.
          checkLoginIframe: false,
        })

        if (cancelled) return

        if (authenticated) {
          syncFromKeycloak()
        } else {
          setSession(null)
          setUser(null)
        }
      } catch (e) {
        if (cancelled) return
        setSession(null)
        setUser(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    keycloak.onAuthSuccess = () => syncFromKeycloak()
    keycloak.onAuthRefreshSuccess = () => syncFromKeycloak()
    keycloak.onAuthLogout = () => {
      setSession(null)
      setUser(null)
    }
    keycloak.onTokenExpired = async () => {
      try {
        await keycloak.updateToken(60)
        syncFromKeycloak()
      } catch {
        setSession(null)
        setUser(null)
      }
    }

    // Periodic refresh to keep long-running screens (polling/training) authenticated.
    refreshIntervalRef.current = window.setInterval(async () => {
      try {
        if (!keycloak.authenticated) return
        await keycloak.updateToken(60)
        syncFromKeycloak()
      } catch {
        // ignore; onTokenExpired will handle hard failures
      }
    }, 30_000)

    return () => {
      cancelled = true
      if (refreshIntervalRef.current) {
        window.clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [])

  const login = async () => {
    try {
      await keycloak.login({
        redirectUri: window.location.origin + "/",
      })
      return { error: null }
    } catch (e) {
      return { error: e instanceof Error ? e.message : "Keycloak login failed" }
    }
  }

  const logout = async () => {
    try {
      await keycloak.logout({
        redirectUri: window.location.origin + "/login",
      })
    } finally {
      setSession(null)
      setUser(null)
    }
  }

  if (loading) return null

  return (
    <AuthContext.Provider value={{ user, session, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
