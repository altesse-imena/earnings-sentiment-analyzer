import { useQuery } from '@tanstack/react-query'
import { fetchNews } from '../api/endpoints'

export function useNews(ticker) {
  return useQuery({
    queryKey: ['news', ticker],
    queryFn: () => fetchNews(ticker),
    enabled: !!ticker,
    staleTime: 5 * 60_000,
    retry: false,
  })
}
