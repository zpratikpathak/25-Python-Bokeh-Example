from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 5, 6]
y = [1, 4, 2, 5, 2, 6]
p = figure(title="Step Chart", x_axis_label='X', y_axis_label='Y')
p.step(x, y, line_width=2, mode="center")
output_file('/tmp/bokeh-repo/examples/5_step_chart.html')
save(p)