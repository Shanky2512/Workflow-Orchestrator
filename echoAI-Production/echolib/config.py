"""
EchoAI Configuration Settings

LLM Provider Configuration:
---------------------------
This module loads LLM provider settings from .env file.
Configure ONE provider at a time by setting the appropriate USE_* flag:
- USE_OLLAMA=true    -> On-Premise Ollama
- USE_OPENROUTER=true -> OpenRouter (current default for development)
- USE_AZURE=true     -> Azure OpenAI (for Azure deployment)
- USE_OPENAI=true    -> Direct OpenAI API

Database Configuration:
-----------------------
- DATABASE_URL: PostgreSQL connection string (async format)
- DATABASE_POOL_SIZE: Connection pool size
- DATABASE_MAX_OVERFLOW: Max connections above pool size

Memcached Configuration:
------------------------
- MEMCACHED_ENABLED: Enable/disable session caching
- MEMCACHED_HOSTS: Comma-separated host:port list
- MEMCACHED_TTL: Default TTL in seconds
- MEMCACHED_FALLBACK: Fall back to DB on cache failure

See .env file for detailed configuration options.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List, Tuple

# Load environment variables from .env file
load_dotenv()


class LLMSettings(BaseModel):
    """LLM Provider Settings"""

    # Provider flags (only one should be true at a time)
    # TO USE OLLAMA: Set USE_OLLAMA default to 'true' and USE_OPENROUTER default to 'false'
    # use_ollama: bool = os.getenv('USE_OLLAMA', 'true').lower() == 'true'
    # use_openrouter: bool = os.getenv('USE_OPENROUTER', 'false').lower() == 'true'
    use_ollama: bool = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
    use_openrouter: bool = os.getenv('USE_OPENROUTER', 'true').lower() == 'true'
    use_azure: bool = os.getenv('USE_AZURE', 'false').lower() == 'true'
    use_openai: bool = os.getenv('USE_OPENAI', 'false').lower() == 'true'

    # Ollama settings
    ollama_base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://10.188.100.130:8002/v1')
    ollama_model: str = os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')

    # OpenRouter settings (current default)
    openrouter_api_key: str = os.getenv(
        'OPENROUTER_API_KEY',
        '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'
    )
    openrouter_base_url: str = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    openrouter_model: str = os.getenv('OPENROUTER_MODEL', 'nvidia/nemotron-3-nano-30b-a3b:free')

    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = os.getenv('AZURE_OPENAI_API_KEY')
    azure_openai_endpoint: Optional[str] = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_deployment: Optional[str] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    azure_openai_api_version: str = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

    # Direct OpenAI settings
    openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
    openai_model: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    # Default LLM parameters
    default_temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    default_max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '4000'))

    def get_active_provider(self) -> str:
        """Get the currently active LLM provider name."""
        if self.use_azure:
            return 'azure'
        elif self.use_openrouter:
            return 'openrouter'
        elif self.use_ollama:
            return 'ollama'
        elif self.use_openai:
            return 'openai'
        else:
            # TO USE OLLAMA AS FALLBACK: return 'ollama'
            return 'openrouter'  # Default fallback


class Settings(BaseModel):
    """Application Settings"""

    # App settings
    app_name: str = os.getenv('APP_NAME', 'echo-mermaid-platform')
    service_mode: str = os.getenv('SERVICE_MODE', 'mono')  # mono | micro
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

    # JWT settings
    jwt_secret: str = os.getenv('JWT_SECRET', 'dev-secret-change-me')
    jwt_issuer: str = os.getenv('JWT_ISSUER', 'echo')
    jwt_audience: str = os.getenv('JWT_AUDIENCE', 'echo-clients')

    # LLM settings
    llm: LLMSettings = LLMSettings()

    # Database settings (PostgreSQL with asyncpg)
    database_url: str = os.getenv(
        'DATABASE_URL',
        'postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai'
    )
    database_pool_size: int = int(os.getenv('DATABASE_POOL_SIZE', '5'))
    database_max_overflow: int = int(os.getenv('DATABASE_MAX_OVERFLOW', '10'))

    # Memcached settings (session caching)
    memcached_enabled: bool = os.getenv('MEMCACHED_ENABLED', 'false').lower() == 'true'
    memcached_hosts: str = os.getenv('MEMCACHED_HOSTS', 'localhost:11211')
    memcached_ttl: int = int(os.getenv('MEMCACHED_TTL', '1800'))  # 30 minutes
    memcached_fallback: bool = os.getenv('MEMCACHED_FALLBACK', 'true').lower() == 'true'
    memcached_pool_size: int = int(os.getenv('MEMCACHED_POOL_SIZE', '10'))
    memcached_timeout: int = int(os.getenv('MEMCACHED_TIMEOUT', '5'))

    # Auth enforcement
    auth_enforcement: str = os.getenv('AUTH_ENFORCEMENT', 'optional')  # optional | required

    # Execution Transparency (WebSocket step events)
    transparency_enabled: bool = os.getenv('ECHO_TRANSPARENCY_ENABLED', 'true').lower() == 'true'

    @property
    def memcached_host_list(self) -> List[Tuple[str, int]]:
        """
        Parse memcached_hosts string into list of (host, port) tuples.

        Supports formats:
        - "localhost:11211"
        - "host1:11211,host2:11211"
        - "host1,host2" (uses default port 11211)

        Returns:
            List of (host, port) tuples for Memcached connections.
        """
        hosts: List[Tuple[str, int]] = []
        for host_str in self.memcached_hosts.split(','):
            host_str = host_str.strip()
            if not host_str:
                continue
            parts = host_str.split(':')
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 11211
            hosts.append((host, port))
        return hosts


# Global settings instance
settings = Settings()

# Convenience access to LLM settings
llm_settings = settings.llm
