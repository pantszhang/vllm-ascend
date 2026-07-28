#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import zmq
from vllm.utils.network_utils import make_zmq_socket

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_zmq_rpc_path_ec_lookup(vllm_config: "VllmConfig") -> str:
    """Return the IPC socket path for EC lookup ZMQ channel.

    Follows the same pattern as ``get_zmq_rpc_path_lookup`` in
    ``pool_scheduler.py``.
    """
    import vllm.envs as vllm_envs

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    base_url = vllm_envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_url}/ec_lookup_dp_rank{dp_rank}"


class EncoderCacheStoreClient:
    """ZMQ REQ client for querying encoder cache existence in memcache.

    Used by the scheduler-side ``EncoderCacheManagerWithStore`` to check
    whether a given mm_hash already has a cached encoder output in
    the remote memcache store.
    """

    def __init__(self, socket_path: str):
        self._ctx = zmq.Context()  # type: ignore[attr-defined]
        self._socket = make_zmq_socket(
            self._ctx, socket_path, zmq.REQ, bind=False  # type: ignore[attr-defined]
        )

    def exists(self, mm_hash: str) -> bool:
        """Check whether *mm_hash* exists in the memcache store."""
        self._socket.send_string(mm_hash)
        return self._socket.recv() == b"1"

    def close(self) -> None:
        """Close the ZMQ socket cleanly."""
        self._socket.close(linger=0)
