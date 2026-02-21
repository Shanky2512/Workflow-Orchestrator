from echolib.di import container
from echolib.adapters import MemcachedSessionStore, AzureADAuth, OTelLogger

container.register('session.store', lambda: MemcachedSessionStore())
container.register('auth.provider', lambda: AzureADAuth())
container.register('logger', lambda: OTelLogger())
