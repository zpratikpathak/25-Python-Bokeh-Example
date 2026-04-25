from bokeh.plotting import figure, output_file, save
p = figure(title="Wedge Plot Example")
p.wedge(x=[1, 2, 3], y=[1, 2, 3], radius=0.2, start_angle=0.4, end_angle=4.8, color="firebrick", alpha=0.6, direction="clock")
output_file('/tmp/bokeh-repo/examples/14_wedge_plot.html')
save(p)