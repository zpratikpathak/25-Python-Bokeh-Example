from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Line Plot Python Bokeh Example", x_axis_label='X', y_axis_label='Y')
p.line(x, y, line_width=2)
output_file('/tmp/bokeh-repo/examples/1_line_plot.html')
save(p)