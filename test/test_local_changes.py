import unittest
import numpy as np
from unittest.mock import patch
from core import Vector

class TestVectorPlots(unittest.TestCase):
    def setUp(self):
        self.data = np.random.randint(0, 10, 100)
        self.vec = Vector(data_points=self.data)

    @patch("matplotlib.pyplot.show")
    def test_linear_scale_runs(self, mock_show):
        """Ensure linear_scale runs and shows a plot."""
        self.vec.linear_scale()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_pdf_runs(self, mock_show):
        """Ensure plot_pdf runs and shows a plot."""
        self.vec.plot_pdf()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_basic_stats_runs(self, mock_show):
        """Ensure plot_basic_stats runs and shows a plot."""
        self.vec.plot_basic_stats()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_log_binning_returns_values_and_plots(self, mock_show):
        """Ensure log_binning runs, plots, and returns valid bounds."""
        in_min, in_max = self.vec.log_binning()
        self.assertTrue(in_min < in_max)
        mock_show.assert_called_once()

if __name__ == "__main__":
    unittest.main()
