import asyncio
import json
import logging
import os
import time
from aiohttp import ClientSession, web, ClientTimeout, TCPConnector


class Dispatcher:
    def __init__(self) -> None:
        self.is_free = {}
        self.backend_port = None
        self.url_path = os.getenv("URL_PATH", "/infer")
        self.idx = 0
        self.total_requests = 0
        self.dispatcher_name = None
        self.sessions = {}
        self.backend_names = []
        self.batch_size = None
        self.queue = asyncio.Queue()  # might consider PriorityQueue for EDF
        self.event = asyncio.Event()
        self.logger = logging.getLogger()
        
    
    async def initialize(self, data: dict):
        asyncio.create_task(self.dispatch())
        self.dispatcher_name = data["dispatcher_name"]
        self.backend_port = int(data["backends_port"])
        self.batch_size = data["batch_size"]
        for backend in data["backends"]:
            self.sessions[backend["name"]] = ClientSession(
                base_url=f"http://{backend['ip']}:{self.backend_port}",
                timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
                connector=TCPConnector(limit=0)
            )
            self.is_free[backend["name"]] = True
            self.backend_names.append(backend["name"])

    def reset_batch_size(self, data: dict):
        if data.get("batch_size"):
            self.batch_size = int(data["batch_size"])
            self.logger.info("New batch size:", self.batch_size)
    
    async def reset_backends(self, data: dict):
        for backend in data["backends"]:
            if backend["name"] not in self.backend_names:
                self.sessions[backend["name"]] = ClientSession(
                    base_url=f"http://{backend['ip']}:{self.backend_port}",
                    timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
                    connector=TCPConnector(limit=0)
                )
                self.is_free[backend["name"]] = True
                self.backend_names.append(backend["name"])
        
        deleted_backends = []
        for backend_name in self.backend_names:
            if backend_name not in list(map(lambda x: x["name"], data["backends"])):
                deleted_backends.append(backend_name)
        
        self.backend_names = list(set(self.backend_names) - set(deleted_backends))
        self.is_free = {k: v for k, v in self.is_free.items() if k not in deleted_backends}
        self.sessions = {k: v for k, v in self.sessions.items() if k not in deleted_backends}
        self.idx = 0

    
    async def receive(self, data: dict):
        self.total_requests += 1
        await self.queue.put({f"arrival-{self.dispatcher_name}": time.time(), **data})
        if self.queue.qsize() < self.batch_size:
            return {"received": True}
        self.event.set()
        return {"scheduled": True}


    def select_backend_to_dispatch(self):
        backend_name = None
        while True:
            if self.is_free[self.backend_names[self.idx]]:
                backend_name = self.backend_names[self.idx]
            self.idx += 1
            if self.idx == len(self.backend_names):
                self.idx = 0
            if backend_name:
                return backend_name
    

    async def dispatch(self):
        while True:
            await self.event.wait()
            self.event.clear()
            backend_name = self.select_backend_to_dispatch()
            batch = []
            if self.queue.qsize() >= self.batch_size:
                for _ in range(self.batch_size):
                    qd = await self.queue.get()
                    qd[f"leaving-{self.dispatcher_name}"] = time.time()
                    batch.append(qd)
            else:
                pass # Fixme

            session: ClientSession = self.sessions[backend_name]
            self.is_free[backend_name] = False
            async with session.post(f"{self.url_path}", data=json.dumps(batch)) as response:
                response = await response.text()
                self.is_free[backend_name] = True
                


dispatcher = Dispatcher()


async def initialize(request):
    data = await request.json()
    if dispatcher.batch_size is None:
        await dispatcher.initialize(data)
        return web.json_response({"success": True})
    return web.json_response({"success": False, "message": "Already initialized."})


async def predict(request):
    data = await request.json()
    return web.json_response(await dispatcher.receive(data))


async def reset_backends(request):
    data = await request.json()
    await dispatcher.reset_backends(data)
    return web.json_response({"success": True})


async def reset_batch_size(request):
    data = await request.json()
    dispatcher.reset_batch_size(data)
    return web.json_response({"success": True})


async def export_request_count(request):
    content = "# HELP dispatcher_requests_total Total number of requests\n"
    content += "# TYPE dispatcher_requests_total counter\n"
    if dispatcher.dispatcher_name:
        content += f'dispatcher_requests_total {dispatcher.total_requests}\n'
    return web.Response(body=content)


app = web.Application()
routes = [
    web.post("/initialize", initialize),
    web.post("/reset-backends", reset_backends),
    web.post("/reset-batch", reset_batch_size),
    web.post("/predict", predict),
]
if os.getenv("EXPORT_REQUESTS_TOTAL"):
    routes.append( web.get("/metrics", export_request_count))

app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("DISPATCHER_PORT", 8002)))
