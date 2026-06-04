import { useCallback, useEffect, useRef, useState } from 'react'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS = 30_000
const BACKOFF_MULT = 2

export function usePriceStream() {
  const [prices, setPrices] = useState({})
  const [status, setStatus] = useState('disconnected')

  const wsRef = useRef(null)
  const retryRef = useRef(null)
  const attemptRef = useRef(0)
  const unmountedRef = useRef(false)

  const getDelay = useCallback((attempt) => {
    const base = Math.min(BASE_DELAY_MS * Math.pow(BACKOFF_MULT, attempt), MAX_DELAY_MS)
    return base + base * 0.2 * (Math.random() * 2 - 1)
  }, [])

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    setStatus('connecting')

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${window.location.host}/ws/prices`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return }
      setStatus('connected')
      attemptRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'price_update' && msg.prices) {
          setPrices((prev) => ({ ...prev, ...msg.prices }))
        }
      } catch {
        // malformed message; ignore
      }
    }

    ws.onclose = () => {
      if (unmountedRef.current) return
      setStatus('disconnected')
      const delay = getDelay(attemptRef.current)
      attemptRef.current += 1
      retryRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [getDelay])

  useEffect(() => {
    unmountedRef.current = false
    connect()
    return () => {
      unmountedRef.current = true
      clearTimeout(retryRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])

  return { prices, status }
}
