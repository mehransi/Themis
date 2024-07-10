
from aiohttp import web
from prometheus_client import start_http_server, Histogram

DEFAULT_BUCKETS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6, 2, 2.5, 3, 4, 5, 7, 10, float("inf")]
dispatcher_stage0_histogram = Histogram(
    'dispatcher_stage0_latency',
    'Dispatcher stage0 latency',
    buckets=DEFAULT_BUCKETS
)
detector_histogram = Histogram(
    'detector_latency',
    'Detector latency',
    buckets=DEFAULT_BUCKETS
)
dispatcher_stage1_histogram = Histogram(
    'dispatcher_stage1_latency',
    'Dispatcher stage1 latency',
    buckets=DEFAULT_BUCKETS
)
classifier_histogram = Histogram(
    'classifier_latency', 
    'Classifier latency',
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
        data["detector-e2e"] = round(data["leaving-Detector"] - data["arrival-Detector"], 2)
        data["dispatcher-stage1"] = round(data["leaving-stage-1"] - data["arrival-stage-1"], 2)
        data["classifier-e2e"] = round(data["leaving-Classifier"] - data["arrival-Classifier"], 2)
        data["e2e"] = round(data["leaving-Classifier"] - data["arrival-stage-0"], 2)
        dispatcher_stage0_histogram.observe(data["dispatcher-stage0"])
        detector_histogram.observe(data["detector-e2e"])
        dispatcher_stage1_histogram.observe(data["dispatcher-stage1"])
        classifier_histogram.observe(data["classifier-e2e"])
        e2e_histogram.observe(data["e2e"])
        
        print(
            f'c={self.c}, dispatcher-stage0={data["dispatcher-stage0"]}, detector={data["detector-e2e"]}, dispatcher-stage1={data["dispatcher-stage1"]}, classifier={data["classifier-e2e"]}, e2e={data["e2e"]}'
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
