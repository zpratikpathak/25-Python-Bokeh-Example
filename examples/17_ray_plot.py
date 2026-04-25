from bokeh.plotting import figure, output_file, save
p = figure(title="Ray Plot Example")
p.ray(x=[1, 2, 3], y=[1, 2, 3], length=45, angle=0.6, color="#1B9E77", line_width=2)
output_file('/tmp/bokeh-repo/examples/17_ray_plot.html')
save(p)