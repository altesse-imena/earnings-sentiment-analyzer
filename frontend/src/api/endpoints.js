import client from './client'

export const fetchTickers = () => client.get('/tickers').then(r => r.data.tickers)
export const fetchDates = (ticker) => client.get(`/tickers/${ticker}/dates`).then(r => r.data.dates)
export const fetchSentiment = (ticker, date) => client.get(`/sentiment/${ticker}/${date}`).then(r => r.data)
export const fetchPrediction = (ticker, date) => client.get(`/prediction/${ticker}/${date}`).then(r => r.data)
export const fetchPrices = (ticker, date) => client.get(`/prices/${ticker}/${date}`).then(r => r.data.prices)
export const fetchShap = () => client.get('/shap').then(r => r.data.features)
export const fetchModelReport = () => client.get('/model/report').then(r => r.data)
export const fetchNews = (ticker) => client.get(`/news/${ticker}`).then(r => r.data)
export const fetchRecommendation = (ticker) => client.get(`/recommendation/${ticker}`).then(r => r.data)
export const triggerIngest = (tickers, years) => client.post('/pipeline/ingest', { tickers, years })
export const triggerProcess = () => client.post('/pipeline/process')
export const triggerTrain = () => client.post('/pipeline/train')
