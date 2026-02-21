from echolib.di import container
from echolib.services import ConnectorManager
 
_mgr = ConnectorManager()
container.register('connector.manager', lambda:_mgr)
 
 