"""vLLM imports with test-time fallbacks.

The connector must inherit vLLM's real KVConnectorBase_V1 in production. Unit
tests in this repo should still run without a vLLM wheel, so this module keeps
fallback shims narrow and explicit.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

try:  # pragma: no cover - exercised in an environment with vLLM installed.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - the fallback is covered by local tests.
    VLLM_AVAILABLE = False

    class KVConnectorRole(Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(
            self,
            vllm_config: Any,
            role: KVConnectorRole,
            kv_cache_config: Any | None = None,
        ) -> None:
            self._vllm_config = vllm_config
            self._role = role
            self._kv_cache_config = kv_cache_config
            self._kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            self._connector_metadata = None

        @property
        def role(self) -> KVConnectorRole:
            return self._role

        def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self) -> KVConnectorMetadata:
            assert self._connector_metadata is not None
            return self._connector_metadata

