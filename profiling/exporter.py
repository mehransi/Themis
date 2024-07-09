import sys
import json
import os
from aiohttp import web


current_dir = os.path.dirname(__file__)

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

    async def write_to_file(self, data: dict):
        p = f"{current_dir}/models/{self.source_name.lower()}"
        os.system(f"mkdir -p {p}/data")
        s = ""
        for k, v in data.items():
            if k != "batch":
                s = s + f"{k}{v}_"
        with open(f"{p}/data/{self.source_name}_latencies_{s}batch{data['batch']}.json", "w") as f:
            json.dump(self.logs[2*data["batch"]:], f, indent=2)
        self.logs = []
        return {"saved_to_file": True}
        

exporter = Exporter()

async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))

async def write_to_file(request):
    data = await request.json()
    return web.json_response(await exporter.write_to_file(data))


app = web.Application()
app.add_routes(
    [
        web.post("/receive", receive),
        web.post("/write", write_to_file)
    ]
)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(sys.argv[1]))
