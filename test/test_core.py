# import numpy as np
# from vectorstats.core import Vector

# def test_init_and_basic():
#     data = [1, 2, 3, 4, 5]
#     v = Vector(label=1, data_points=data)
#     assert v.n == 5
#     assert np.allclose(v.v, data)

# def test_normalize():
#     v = Vector(data_points=[0, 5, 10])
#     result = v.normalize()
#     assert np.allclose(result, [0.0, 0.5, 1.0])

# def test_standardize():
#     v = Vector(data_points=[1, 2, 3])
#     result = v.standardize()
#     assert np.isclose(np.mean(result), 0)
#     assert np.isclose(np.std(result), 1)

# def test_pearson():
#     v = Vector(data_points=[1, 2, 3, 4, 5])
#     assert np.isclose(v.pearson_correlation(), 1.0)

# def test_rolling_average():
#     v = Vector(data_points=[1, 2, 3, 4, 5])
#     result = v.rolling_average(window=3)
#     assert np.allclose(result, [2.0, 3.0, 4.0])

# def test_split():
#     v = Vector(data_points=[10, 20, 30, 40, 50])
#     x_train, x_test, y_train, y_test = v.split(test_size=0.4)
#     assert len(x_test) == 2
#     assert len(y_train) == 3

# def test_knn_classify():
#     v1 = Vector(label=0, data_points=[1, 1, 1])
#     v2 = Vector(label=1, data_points=[10, 10, 10])
#     test = np.array([9, 9, 9])
#     result = Vector.knn_classify(test, [v1, v2], k=1)
#     assert result == 1
