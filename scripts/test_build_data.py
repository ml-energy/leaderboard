import unittest
from build_data import smooth_chunked_itl


class TestSmoothChunkedItl(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(smooth_chunked_itl([]), [])

    def test_all_zeros(self):
        self.assertEqual(smooth_chunked_itl([0, 0, 0]), [])

    def test_leading_zeros_filtered(self):
        # Leading zeros should be removed, then chunk smoothed
        result = smooth_chunked_itl([0, 0, 100, 0, 0, 0])
        self.assertEqual(result, [25, 25, 25, 25])

    def test_single_chunk(self):
        # [100, 0, 0, 0] -> 4 tokens, 100 total -> [25, 25, 25, 25]
        result = smooth_chunked_itl([100, 0, 0, 0])
        self.assertEqual(result, [25, 25, 25, 25])

    def test_multiple_chunks(self):
        # [100, 0, 50, 0] -> chunk1: [100,0]->2 tokens, chunk2: [50,0]->2 tokens
        result = smooth_chunked_itl([100, 0, 50, 0])
        self.assertEqual(result, [50, 50, 25, 25])

    def test_proper_streaming_unchanged(self):
        # All non-zero values should remain unchanged
        result = smooth_chunked_itl([50, 60, 70, 80])
        self.assertEqual(result, [50, 60, 70, 80])

    def test_mixed_streaming_and_chunks(self):
        # [100, 50, 0, 0] -> 100 is single, [50, 0, 0] is chunk of 3
        result = smooth_chunked_itl([100, 50, 0, 0])
        expected = [100, 50 / 3, 50 / 3, 50 / 3]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_chunk_at_end(self):
        # [30, 0] at end
        result = smooth_chunked_itl([25, 30, 0])
        self.assertEqual(result, [25, 15, 15])

    def test_single_nonzero_value(self):
        result = smooth_chunked_itl([100])
        self.assertEqual(result, [100])


if __name__ == "__main__":
    unittest.main()
