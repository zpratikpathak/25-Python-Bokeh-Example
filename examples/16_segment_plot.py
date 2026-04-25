from bokeh.plotting import figure, output_file, save
p = figure(title="Segment Plot Example")
p.segment(x0=[1, 2, 3], y0=[1, 2, 3], x1=[1.2, 2.5, 3.7], y1=[1.5, 2.1, 3.9], color="#F4A582", line_width=3)
output_file('/tmp/bokeh-repo/examples/16_segment_plot.html')
save(p)