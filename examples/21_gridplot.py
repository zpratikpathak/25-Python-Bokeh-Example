from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
p1 = figure(width=250, height=250, title="Plot 1")
p1.scatter(x, y0, size=10, color="firebrick")
p2 = figure(width=250, height=250, title="Plot 2")
p2.line(x, y1, line_width=3, color="navy")
p = gridplot([[p1, p2]])
output_file('/tmp/bokeh-repo/examples/21_gridplot.html')
save(p)