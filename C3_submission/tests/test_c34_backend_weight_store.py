from __future__ import annotations

import unittest

import numpy as np

from scheduler.memory import (
    DeviceMemoryBackend,
    DeviceMemoryPool,
    WeightStore,
)


class DeviceBackendAndWeightStoreTests(unittest.TestCase):
    def test_backend_view_matches_host_values(self) -> None:
        pool = DeviceMemoryPool(
            4096,
            alignment_bytes=256,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            4096,
            prefer_gpu=False,
        )
        backend.attach_to_pool(pool)

        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        allocation = pool.allocate("w", array.nbytes)
        backend.copy_from_host(array, allocation)

        view = backend.get_view(
            allocation,
            dtype=array.dtype,
            shape=array.shape,
        )
        np.testing.assert_array_equal(
            backend.to_host(view),
            array,
        )

    def test_weight_preload_uploads_once(self) -> None:
        initializers = {
            "w": np.arange(16, dtype=np.float32).reshape(4, 4),
            "b": np.arange(4, dtype=np.float32),
        }
        capacity = WeightStore.required_capacity(
            initializers,
            alignment_bytes=256,
        )

        pool = DeviceMemoryPool(
            capacity,
            alignment_bytes=256,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            capacity,
            prefer_gpu=False,
        )
        store = WeightStore(pool, backend)

        store.preload(initializers)
        self.assertEqual(store.copy_count, 2)

        # Repeated preload must not copy again.
        store.preload(initializers)
        self.assertEqual(store.copy_count, 2)

        np.testing.assert_array_equal(
            backend.to_host(store.get("w")),
            initializers["w"],
        )
        np.testing.assert_array_equal(
            backend.to_host(store.get("b")),
            initializers["b"],
        )

    def test_weight_records_have_distinct_offsets(self) -> None:
        initializers = {
            "a": np.ones((32,), dtype=np.float32),
            "b": np.ones((64,), dtype=np.float32),
        }
        capacity = WeightStore.required_capacity(initializers)

        pool = DeviceMemoryPool(
            capacity,
            alignment_bytes=256,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            capacity,
            prefer_gpu=False,
        )
        store = WeightStore(pool, backend)
        store.preload(initializers)

        a = store.get_record("a").allocation
        b = store.get_record("b").allocation

        self.assertLessEqual(
            a.end_offset_bytes,
            b.offset_bytes,
        )
        self.assertTrue(pool.validate())

    def test_insufficient_capacity_raises(self) -> None:
        pool = DeviceMemoryPool(
            256,
            alignment_bytes=256,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            256,
            prefer_gpu=False,
        )
        store = WeightStore(pool, backend)

        with self.assertRaises(MemoryError):
            store.upload(
                "too_large",
                np.ones((128,), dtype=np.float32),
            )

    def test_stats(self) -> None:
        initializers = {
            "w": np.ones((8,), dtype=np.float32),
        }
        capacity = WeightStore.required_capacity(initializers)

        pool = DeviceMemoryPool(
            capacity,
            alignment_bytes=256,
            pool_name="weights",
        )
        backend = DeviceMemoryBackend(
            capacity,
            prefer_gpu=False,
        )
        store = WeightStore(pool, backend)
        store.preload(initializers)

        stats = store.stats()
        self.assertEqual(stats["preloaded_weight_count"], 1)
        self.assertEqual(stats["copy_count"], 1)
        self.assertEqual(stats["uploaded_bytes"], 32)
        self.assertEqual(stats["backend"], "numpy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
