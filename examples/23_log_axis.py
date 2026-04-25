from bokeh.plotting import figure, output_file, save
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y = [10**i for i in x]
p = figure(title="Logarithmic Axis Example", y_axis_type="log")
p.line(x, y, line_width=2)
p.scatter(x, y, fill_color="white", size=8)
output_file('/tmp/bokeh-repo/examples/23_log_axis.html')
save(p)