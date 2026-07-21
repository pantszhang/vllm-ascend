# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC MemCache Connector package."""

from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.connector import (
    ECMemCacheConnector,
)

__all__ = ["ECMemCacheConnector"]
