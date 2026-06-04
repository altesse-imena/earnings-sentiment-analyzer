import { usePrediction, usePrices, useSentiment, useShap } from '../../hooks/useDashboard'
import EmptyState from '../shared/EmptyState'
import Loader from '../shared/Loader'
import PredictionPanel from './PredictionPanel'
import PriceChart from './PriceChart'
import SectionChart from './SectionChart'
import SentimentChart from './SentimentChart'
import ShapChart from './ShapChart'
import StatCards from './StatCards'

export default function Dashboard({ ticker, date }) {
  const { data: sentiment, isLoading, isError } = useSentiment(ticker, date)
  const { data: prediction } = usePrediction(ticker, date)
  const { data: prices = [] } = usePrices(ticker, date)
  const { data: shapFeatures = [] } = useShap()

  if (isLoading) return <Loader />
  if (isError || !sentiment) {
    return <EmptyState message={`No sentiment data for ${ticker} on ${date}.`} />
  }

  return (
    <div>
      <StatCards sentiment={sentiment} prediction={prediction} />
      <div className="chart-row">
        <SentimentChart sentiment={sentiment} />
        <SectionChart sentiment={sentiment} />
      </div>
      <PriceChart prices={prices} eventDate={date} overallSentiment={sentiment.overall_sentiment} />
      <div className="chart-row-60-40">
        <ShapChart features={shapFeatures} />
        <PredictionPanel prediction={prediction} />
      </div>
    </div>
  )
}
