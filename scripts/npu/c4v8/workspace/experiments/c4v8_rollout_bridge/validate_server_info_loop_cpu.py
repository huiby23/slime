"""Read-only pinned-method reproduction of server_info receive-loop dependency.

Uses the actual AST of _Communicator and get_internal_state with fake sockets.
No SGLang import, torch, NPU, HTTP server, model, or running-source mutation.
"""
import ast,asyncio,copy,hashlib,json,types,typing
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'pinned-source/tokenizer_communicator_mixin.py'
module=ast.parse(SOURCE.read_text())
comm=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='_Communicator')
method=next(n for cls in module.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='get_internal_state')
ns={'asyncio':asyncio,'copy':copy,'deque':deque,'T':typing.TypeVar('T'),'TokenizerManager':object,
    'GetInternalStateReq':type('GetInternalStateReq',(),{}),'GetInternalStateReqOutput':types.SimpleNamespace,
    'zmq':types.SimpleNamespace(Socket=object)}
ns.update({name:getattr(typing,name) for name in ('Generic','Optional','List','Deque','Dict','Any')})
exec(compile(ast.fix_missing_locations(ast.Module(body=[comm,method],type_ignores=[])),str(SOURCE),'exec'),ns)

async def test():
    queue=asyncio.Queue()
    class Sender:
        def send_pyobj(self,obj):queue.put_nowait(obj)
    communicator=ns['_Communicator'](Sender(),1)
    manager=types.SimpleNamespace(get_internal_state_communicator=communicator,auto_create_calls=0)
    def auto_create():manager.auto_create_calls+=1
    manager.auto_create_handle_loop=auto_create
    get_state=types.MethodType(ns['get_internal_state'],manager)
    pending=asyncio.create_task(get_state())
    await asyncio.sleep(.03)
    assert not pending.done() and queue.qsize()==1 and manager.auto_create_calls==0
    async def start_receive_loop():
        await queue.get()
        communicator.handle_recv(types.SimpleNamespace(internal_state={'ready':True}))
    await start_receive_loop()
    result=await asyncio.wait_for(pending,1)
    assert result==[{'ready':True}]
    return {'request_sent_without_receive_loop':True,'remains_pending_before_receive_loop':True,
            'get_internal_state_auto_create_calls':manager.auto_create_calls,
            'completes_after_receive_loop_dispatch':True,'fan_out':1}

if __name__=='__main__':
    r=asyncio.run(test());r.update(passed=True,validation_kind='CPU actual pinned AST with fake sockets',
        production_network_or_NPU_test=False,source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    out=ROOT/'evidence/server-info-loop-cpu.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
