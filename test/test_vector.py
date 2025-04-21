import unittest
import numpy as np
from src.core import Vector  # Adjust the import path as needed

class TestVector(unittest.TestCase):

    def setUp(self):
        self.data1 = np.array([1, 2, 3, 4, 5])
        self.data2 = np.array([5, 4, 3, 2, 1])
        self.vec1 = Vector(label=1, data_points=self.data1)
        self.vec2 = Vector(label=2, data_points=self.data2)

    def test_count(self):
        count = self.vec1.count(self.data1, 3)
        self.assertEqual(count, 1)

    def test_get_mean(self):
        mean = self.vec1.get_mean()
        self.assertAlmostEqual(mean, 3.0)

    def test_get_median(self):
        median = self.vec1.get_median()
        self.assertEqual(median, 3.0)

    def test_get_std(self):
        std = self.vec1.get_std()
        self.assertAlmostEqual(std, np.std(self.data1))

    def test_normalize(self):
        normalized = self.vec1.normalize()
        self.assertAlmostEqual(np.mean(normalized), 0, places=6)
        self.assertAlmostEqual(np.std(normalized), 1, places=6)

    def test_resample(self):
        resampled = self.vec1.resample(size=10)
        self.assertEqual(len(resampled), 10)
        for value in resampled:
            self.assertIn(value, self.data1)

    def test_add(self):
        result = self.vec1.add(self.vec2)
        np.testing.assert_array_equal(result.x, np.add(self.data1, self.data2))

    def test_subtract(self):
        result = self.vec1.subtract(self.vec2)
        np.testing.assert_array_equal(result.x, np.subtract(self.data1, self.data2))

    def test_dot(self):
        dot_result = self.vec1.dot(self.vec2)
        self.assertEqual(dot_result, np.dot(self.data1, self.data2))

    def test_set_operations(self):
        union, intersection, jaccard = Vector.set_operations(self.vec1, self.vec2)
        expected_union = set(self.data1).union(set(self.data2))
        expected_intersection = set(self.data1).intersection(set(self.data2))
        expected_jaccard = len(expected_intersection) / len(expected_union)

        self.assertEqual(union, expected_union)
        self.assertEqual(intersection, expected_intersection)
        self.assertAlmostEqual(jaccard, expected_jaccard)

    def test_calculate_distance(self):
        dist = self.vec1.calculate_distance(self.vec2)
        expected = np.linalg.norm(self.data1 - self.data2)
        self.assertAlmostEqual(dist, expected)

    def test_get_prob_vector(self):
        prob_vector = self.vec1.get_prob_vector()
        expected_keys = set(self.data1)
        self.assertEqual(set(prob_vector.keys()), expected_keys)
        self.assertAlmostEqual(sum(prob_vector.values()), 1.0)

    def test_cross(self):
        v1 = Vector(data_points=np.array([1, 0, 0]))
        v2 = Vector(data_points=np.array([0, 1, 0]))
        cross = v1.cross(v2)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(cross, expected)

if __name__ == '__main__':
    unittest.main()
