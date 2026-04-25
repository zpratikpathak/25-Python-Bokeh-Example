from bokeh.plotting import figure, output_file, save
from bokeh.models import Range1d
p = figure(title="Custom Range Plot", x_range=Range1d(0, 10), y_range=Range1d(0, 20))
p.scatter([1, 5, 9], [2, 10, 18], size=10, color="navy")
output_file('/tmp/bokeh-repo/examples/24_range_tool.html')
save(p)