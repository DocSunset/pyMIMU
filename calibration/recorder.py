import asyncio
import sys
import typing

from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from calibration.data import Flag

class AsyncOSC:
  def __init__(self, port: int, handler, *handler_args):
    self.recording = Flag()
    self.transport = None
    self.port = port
    self.osc_dispatcher = Dispatcher()
    self.osc_dispatcher.map("/raw", self.handler_wrapper(handler), *handler_args)
    # TODO make out ip an arg
    self.output = SimpleUDPClient("192.168.0.255", 6006, allow_broadcast=True)

  def handler_wrapper(self, handler):
    def _handler(address, refs, *args):
      if not self.recording: return
      else: handler(address, refs, *args)
    return _handler

  def stop_recording(self) -> None:
    print("Stopping recorder")
    self.recording.set(False)

  def resume_recording(self) -> None:
    print("Restarting recorder")
    self.recording.set(True)

  def send(self, address, *args):
    print(address, args)
    self.output.send_message(address, args)
  
  async def start_recording(self, ):
    self.output.send_message("/state/calibrate", [])
    loop = asyncio.get_event_loop()
    osc_server = AsyncIOOSCUDPServer(
        ("0.0.0.0", self.port), 
        self.osc_dispatcher, 
        loop)
    self.transport, _ = await osc_server.create_serve_endpoint()
    self.recording.set(True)
