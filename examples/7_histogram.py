from bokeh.plotting import figure, output_file, save
import numpy as np
measured = np.random.normal(0, 0.5, 1000)
hist, edges = np.histogram(measured, density=True, bins=50)
p = figure(title="Histogram", background_fill_color="#fafafa")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
output_file('/tmp/bokeh-repo/examples/7_histogram.html')
save(p)