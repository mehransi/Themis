import sys
from aiohttp import web
from prometheus_client import start_http_server, Histogram

DEFAULT_BUCKETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.5, 3, 4, 5, 7, 10, float("inf")]
DEFAULT_BUCKETS.append(float(sys.argv[1]))  # Adding SLO to buckets
DEFAULT_BUCKETS.sort()

dispatcher_stage0_histogram = Histogram(
    'dispatcher_stage0_latency',
    'Dispatcher stage0 latency',
    buckets=DEFAULT_BUCKETS
)
audio_histogram = Histogram(
    'audio_latency',
    'Audio to text latency',
    buckets=DEFAULT_BUCKETS
)
dispatcher_stage1_histogram = Histogram(
    'dispatcher_stage1_latency',
    'Dispatcher stage1 latency',
    buckets=DEFAULT_BUCKETS
)
sentiment_histogram = Histogram(
    'sentiment_latency', 
    'Sentiment analysis latency',
    buckets=DEFAULT_BUCKETS
)
e2e_histogram = Histogram(
    'pelastic_requests_latency', 
    'End-to-end pipeline latency',
    buckets=DEFAULT_BUCKETS
)

class Exporter:
    def __init__(self) -> None:
        self.c = 0
        
    async def receive(self, data: dict):
        self.c += 1
        data["dispatcher-stage0"] = round(data["leaving-stage-0"] - data["arrival-stage-0"], 2)
        data["audio-e2e"] = round(data["leaving-AudioToText"] - data["arrival-AudioToText"], 2)
        data["dispatcher-stage1"] = round(data["leaving-stage-1"] - data["arrival-stage-1"], 2)
        data["sentiment-e2e"] = round(data["leaving-SentimentAnalysis"] - data["arrival-SentimentAnalysis"], 2)
        data["e2e"] = round(data["leaving-SentimentAnalysis"] - data["arrival-stage-0"], 2)
        dispatcher_stage0_histogram.observe(data["dispatcher-stage0"])
        audio_histogram.observe(data["audio-e2e"])
        dispatcher_stage1_histogram.observe(data["dispatcher-stage1"])
        sentiment_histogram.observe(data["sentiment-e2e"])
        e2e_histogram.observe(data["e2e"])
        
        print(
            f'c={self.c}, dispatcher-stage0={data["dispatcher-stage0"]}, audio={data["audio-e2e"]}, dispatcher-stage1={data["dispatcher-stage1"]}, sentiment={data["sentiment-e2e"]}, e2e={data["e2e"]}'
        )
        print()
 
        return {"saved": True}


exporter = Exporter()


async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))


app = web.Application()
app.add_routes(
    [
        web.post("/receive", receive),
    ]
)

if __name__ == '__main__':
    start_http_server(8009)
    web.run_app(app, host="0.0.0.0", port=8008)
