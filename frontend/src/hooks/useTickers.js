import { useQuery } from '@tanstack/react-query'
import { fetchTickers } from '../api/endpoints'

export function useTickers() {
  return useQuery({
    queryKey: ['tickers'],
    queryFn: fetchTickers,
    staleTime: 60_000,
  })
}
