from bokeh.plotting import figure, output_file, save
import numpy as np
x = np.random.standard_normal(500)
y = np.random.standard_normal(500)
p = figure(title="Hexbin Plot", match_aspect=True, background_fill_color='#440154')
p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)
output_file('/tmp/bokeh-repo/examples/10_hexbin_plot.html')
save(p)