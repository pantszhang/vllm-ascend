#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""EC backend registry and factory.

Follows the same pattern as the KV backend registry
(``vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend``)
but is completely independent — EC backends and KV backends can
be configured to use different stores simultaneously.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.distributed.ec_transfer.ec_backend.backend import ECBackend

ec_backend_map: dict[str, dict[str, str]] = {
    "memcache": {
        "name": "ECMemcacheBackend",
        "path": "vllm_ascend.distributed.ec_transfer.ec_backend.memcache_backend",
    },
}


def create_ec_backend(name: str, **kwargs) -> "ECBackend":
    """Dynamically import and instantiate an EC backend by name.

    Args:
        name: backend name key in ``ec_backend_map``.
        **kwargs: forwarded to the backend constructor.

    Returns:
        ECBackend: the instantiated backend.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    entry = ec_backend_map.get(name.lower())
    if entry is None:
        raise ValueError(
            f"Unsupported EC backend: {name!r}. "
            f"Registered backends: {list(ec_backend_map)}"
        )
    module_path = entry["path"]
    class_name = entry["name"]

    module = importlib.import_module(module_path)
    backend_cls = getattr(module, class_name)
    return backend_cls(**kwargs)
