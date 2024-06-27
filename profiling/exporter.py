import sys
import json
from aiohttp import web

class Exporter:
    def __init__(self) -> None:
        self.source_name = sys.argv[2]
        self.logs = []
    
    async def receive(self, data: dict):
        latency = data[f"leaving-{self.source_name}"] - data[f"arrival-{self.source_name}"]
        del data[f"leaving-{self.source_name}"]
        del data[f"arrival-{self.source_name}"]
        del data["data"]
        data["e2e"] = latency
        self.logs.append(data)
        return {"saved": True}

    async def write_to_file(self, filename):
        with open(filename, "a") as f:
            json.dump(self.logs, f, indent=2)
        self.logs = []
        return {"saved_to_file": True}
        

exporter = Exporter()

async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))

async def write_to_file(request):
    data = await request.json()
    return web.json_response(await exporter.write_to_file(data["filename"]))


app = web.Application()
app.add_routes(
    [
        web.post("/receive", receive),
        web.post("/write", write_to_file)
    ]
)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(sys.argv[1]))
