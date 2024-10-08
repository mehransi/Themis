import sys
from aiohttp import web
import json
import os
from datetime import datetime
from prometheus_client import start_http_server, Histogram

DEFAULT_BUCKETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.5, 3, 4, 5, 7, 10, float("inf")]
DEFAULT_BUCKETS.append(float(sys.argv[1]))  # Adding SLO to buckets
DEFAULT_BUCKETS.sort()

dispatcher_stage0_histogram = Histogram(
    'dispatcher_stage0_latency',
    'Dispatcher stage0 latency',
    buckets=DEFAULT_BUCKETS
)
identification_histogram = Histogram(
    'identification_latency',
    'Identification latency',
    buckets=DEFAULT_BUCKETS
)
dispatcher_stage1_histogram = Histogram(
    'dispatcher_stage1_latency',
    'Dispatcher stage1 latency',
    buckets=DEFAULT_BUCKETS
)
translation_histogram = Histogram(
    'translation_latency', 
    'Translation latency',
    buckets=DEFAULT_BUCKETS
)
dispatcher_stage2_histogram = Histogram(
    'dispatcher_stage2_latency',
    'Dispatcher stage2 latency',
    buckets=DEFAULT_BUCKETS
)
summarizer_histogram = Histogram(
    'summarizer_latency', 
    'Summarizer latency',
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
        self.per_request_list = []
        
    async def receive(self, data: dict):
        self.c += 1
        data["dispatcher-stage0"] = round(data["leaving-stage-0"] - data["arrival-stage-0"], 2)
        data["identification-e2e"] = round(data["leaving-LanguageIdentification"] - data["arrival-LanguageIdentification"], 2)
        data["dispatcher-stage1"] = round(data["leaving-stage-1"] - data["arrival-stage-1"], 2)
        data["translation-e2e"] = round(data["leaving-Translation"] - data["arrival-Translation"], 2)
        data["dispatcher-stage2"] = round(data["leaving-stage-2"] - data["arrival-stage-2"], 2)
        data["summarizer-e2e"] = round(data["leaving-Summarizer"] - data["arrival-Summarizer"], 2)
        data["e2e"] = round(data["leaving-Summarizer"] - data["arrival-stage-0"], 2)

        dispatcher_stage0_histogram.observe(data["dispatcher-stage0"])
        identification_histogram.observe(data["identification-e2e"])
        dispatcher_stage1_histogram.observe(data["dispatcher-stage1"])
        translation_histogram.observe(data["translation-e2e"])
        dispatcher_stage2_histogram.observe(data["dispatcher-stage2"])
        summarizer_histogram.observe(data["summarizer-e2e"])
        e2e_histogram.observe(data["e2e"])
        
        per_request = {}
        for k in ["dispatcher-stage0", "identification-e2e", "dispatcher-stage1", "translation-e2e", "dispatcher-stage2", "summarizer-e2e", "e2d"]:
            per_request[k] = data[k]
        per_request["timestamp"] = str(datetime.now())
        self.per_request_list.append(per_request)
        
        if data["dispatcher-stage0"] > 2.5 or data["dispatcher-stage1"] > 2.5 or data["dispatcher-stage2"] > 2.5:
            print("************************************", data["dispatcher-stage0"], data["dispatcher-stage1"]), data["dispatcher-stage2"]
        
        
        print(
            f'c={self.c}, {per_request}'
        )
        print()
 
        return {"saved": True}


exporter = Exporter()


async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))


filepath = os.path.dirname(os.path.dirname(__file__))
async def save_to_file(request):
    data = await request.json()
    with open(f"{filepath}/nlp-{data['adapter']}.json", "w") as f:
        json.dump(exporter.per_request_list, f, indent=2)
    return web.json_response({"saved": True})
        

app = web.Application()
app.add_routes(
    [
        web.post("/receive", receive),
        web.post("/save", save_to_file),
    ]
)

if __name__ == '__main__':
    start_http_server(8009)
    web.run_app(app, host="0.0.0.0", port=8008)
