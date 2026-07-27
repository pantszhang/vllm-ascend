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
"""ECBackend — abstract storage backend for EC connector encoder cache offloading.

This module is independent of the KV transfer backend. It defines a
minimal interface sufficient for encoder-cache save / load / lookup.
"""

from abc import ABC, abstractmethod


class ECBackend(ABC):
    """Abstract backend for encoder cache storage.

    The scheduler side uses a lightweight client that only needs
    ``batch_is_exist`` (created via :meth:`create_scheduler_client`).
    The worker side uses the full backend for alloc, copy, and lease.
    """

    # ------------------------------------------------------------------
    # Core operations (worker side)
    # ------------------------------------------------------------------

    @abstractmethod
    def batch_is_exist(self, keys: list[str]) -> list[int]:
        """Check whether each key exists in the store.

        Returns:
            list[int]: 1 for existing, 0 for missing (one per key).
        """

    @abstractmethod
    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        """Allocate blobs and return a GVA (global virtual address) per key.

        Args:
            keys: string identifiers.
            sizes: byte sizes per blob.

        Returns:
            list[int]: GVA per key (int handle).
        """

    @abstractmethod
    def batch_get_key_info(self, keys: list[str]):
        """Return key-info objects for the given keys.

        Each object exposes ``.size()`` (byte size) and ``.gva_list()``
        (list of int GVAs for the blob).
        """

    @abstractmethod
    def batch_copy(
        self,
        gvas: list[int],
        addrs: list[int],
        sizes: list[int],
        direction: int,
    ) -> int:
        """DMA copy between local NPU memory and the global store.

        Args:
            gvas: global virtual addresses (from ``batch_alloc``).
            addrs: local NPU data pointers.
            sizes: byte lengths.
            direction: 0 = L2G (save), 1 = G2L (load).

        Returns:
            int: 0 on success, non-zero on failure.
        """

    @abstractmethod
    def batch_add_lease(self, keys: list[str], ttl_ms: int) -> list[int]:
        """Acquire a read lease on each key (prevents eviction during load).

        Returns:
            list[int]: 0 per key on success, non-zero on failure.
        """

    @abstractmethod
    def batch_remove_lease(self, keys: list[str]) -> int:
        """Release read leases acquired via ``batch_add_lease``.

        Returns:
            int: 0 on success.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_scheduler_client(cls, **kwargs) -> "ECBackend":
        """Create a lightweight scheduler-side client.

        The returned client only needs to support ``batch_is_exist``.
        Default implementation delegates to the constructor with
        ``init_bm=False``.
        """
        return cls(**kwargs)
