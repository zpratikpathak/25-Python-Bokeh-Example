from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 2, 4, 5]
y2 = [5, 2, 8, 6, 2]
p = figure(title="Multiple Lines")
p.line(x, y1, legend_label="Temp", color="blue", line_width=2)
p.line(x, y2, legend_label="Rate", color="red", line_width=2)
output_file('/tmp/bokeh-repo/examples/8_multiple_lines.html')
save(p)