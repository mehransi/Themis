
from aiohttp import web
from prometheus_client import start_http_server, Histogram

e2e_histogram = Histogram('pelastic_requests_latency', 'End-to-end pipeline latency')

class Exporter:
    def __init__(self) -> None:
        pass
        
    async def receive(self, data: dict):
        # data["dispatcher-stage0"] = f'{data["leaving-stage-0"] - data["arrival-stage-0"]:.3f}'
        # data["detector-e2e"] = f'{data["leaving-Detector"] - data["arrival-Detector"]:.3f}'
        # data["dispatcher-stage1"] = f'{data["leaving-stage-1"] - data["arrival-stage-1"]:.3f}'
        # data["classifier-e2e"] = f'{data["leaving-Classifier"] - data["arrival-Classifier"]:.3f}'
        data["e2e"] = data["leaving-Classifier"] - data["arrival-stage-0"]
        e2e_histogram.observe(data["e2e"])
 
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
