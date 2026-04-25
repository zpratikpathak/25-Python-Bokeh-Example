from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Scatter Plot Example", x_axis_label='X', y_axis_label='Y')
p.scatter(x, y, size=15, color="navy", alpha=0.5)
output_file('/tmp/bokeh-repo/examples/2_scatter_plot.html')
save(p)