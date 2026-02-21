"""
Convobi package container.

You create this file in the new app folder (e.g. apps/convobi/). In it you
register any services from echolib that the convobi app uses. The gateway
imports this module so those registrations are available app-wide.

Usage in routes.py or main.py:
  from echolib.di import container
  logger = container.resolve("convobi.logger")
"""
from echolib.di import container
from echolib.adapters import OTelLogger

# Register echolib services used by convobi. Add more as needed when you use
# other echolib services (e.g. ConnectorManager, DocumentStore, event.bus).
container.register("convobi.logger", lambda: OTelLogger())
