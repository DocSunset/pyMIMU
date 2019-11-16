import asyncio
import typing

from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from .flag import Flag

class AsyncOSC:
  def __init__(self, port: int, handler, *handler_args, outip='192.168.0.255', outport = 6000):
    self.running = Flag()
    self.transport = None
    self.port = port
    self.osc_dispatcher = Dispatcher()
    self.osc_dispatcher.map("/raw", self.handler_wrapper(handler), *handler_args)
    self.output = SimpleUDPClient(outip, outport, allow_broadcast=True)

  def handler_wrapper(self, handler):
    def _handler(address, refs, *args):
      if not self.running: return
      else: handler(address, refs, *args)
    return _handler

  def stop(self) -> None:
    print("Stopping AsyncOSC")
    self.running.set(False)

  def resume(self) -> None:
    print("Restarting AsyncOSC")
    self.running.set(True)

  def send(self, address, *args):
    print(address, args)
    self.output.send_message(address, args)
  
  async def setup(self, ):
    loop = asyncio.get_event_loop()
    osc_server = AsyncIOOSCUDPServer(
        ("0.0.0.0", self.port), 
        self.osc_dispatcher, 
        loop)
    self.transport, _ = await osc_server.create_serve_endpoint()
    self.running.set(True)
