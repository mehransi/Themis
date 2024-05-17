
from aiohttp import web


class Exporter:
    def __init__(self) -> None:
        pass

    
    async def receive(self, data: dict):
        # TODO: save latency metrics for exporter
        print("Exporter received")
        print(data)
        return {"saved": True}


exporter = Exporter()


async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))


async def export_metrics(request):
    # TODO: export for Prometheus
    content = ""
    return web.Response(body=content)


app = web.Application()
app.add_routes(
    [
        web.post("/receive", receive),
        web.get("/metrics", export_metrics),
    ]
)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8080)
