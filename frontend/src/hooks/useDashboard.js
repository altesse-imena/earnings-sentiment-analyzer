import { useQuery } from '@tanstack/react-query'
import {
  fetchDates,
  fetchModelReport,
  fetchPrediction,
  fetchPrices,
  fetchSentiment,
  fetchShap,
} from '../api/endpoints'

export function useDates(ticker) {
  return useQuery({
    queryKey: ['dates', ticker],
    queryFn: () => fetchDates(ticker),
    enabled: !!ticker,
    staleTime: 60_000,
  })
}

export function useSentiment(ticker, date) {
  return useQuery({
    queryKey: ['sentiment', ticker, date],
    queryFn: () => fetchSentiment(ticker, date),
    enabled: !!(ticker && date),
  })
}

export function usePrediction(ticker, date) {
  return useQuery({
    queryKey: ['prediction', ticker, date],
    queryFn: () => fetchPrediction(ticker, date),
    enabled: !!(ticker && date),
    retry: false,
  })
}

export function usePrices(ticker, date) {
  return useQuery({
    queryKey: ['prices', ticker, date],
    queryFn: () => fetchPrices(ticker, date),
    enabled: !!(ticker && date),
  })
}

export function useShap() {
  return useQuery({
    queryKey: ['shap'],
    queryFn: fetchShap,
    staleTime: 120_000,
  })
}

export function useModelReport() {
  return useQuery({
    queryKey: ['modelReport'],
    queryFn: fetchModelReport,
    staleTime: 120_000,
  })
}
