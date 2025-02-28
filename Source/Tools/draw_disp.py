from PIL import Image
from . import visualization


viz = visualization.Visualizer(rgb)
visualized_output = viz.draw_disparity(disp_pred, colormap="kitti")
