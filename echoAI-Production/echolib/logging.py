
import logging
from .config import settings
_level = getattr(logging, settings.log_level.upper(), logging.INFO)
logging.basicConfig(level=_level, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(settings.app_name)
