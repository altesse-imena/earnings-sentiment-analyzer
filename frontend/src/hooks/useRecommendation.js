import { useQuery } from '@tanstack/react-query'
import { fetchRecommendation } from '../api/endpoints'

export function useRecommendation(ticker) {
  return useQuery({
    queryKey: ['recommendation', ticker],
    queryFn: () => fetchRecommendation(ticker),
    enabled: !!ticker,
    staleTime: 5 * 60_000,
    retry: false,
  })
}
