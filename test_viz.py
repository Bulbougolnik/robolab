"""Дымовой тест визуализации: рисунки строятся без исключений."""
import matplotlib
matplotlib.use("Agg")
import pytest

pytest.importorskip("matplotlib")
from robolab import viz


def test_viz_smoke():
    for f in (viz.demo_point_vs_vector, viz.demo_composition_order, viz.show_leg,
              viz.plot_force_projection, viz.plot_mass_matrix):
        fig = f()
        assert fig.axes
