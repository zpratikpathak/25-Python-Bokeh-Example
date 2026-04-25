from bokeh.plotting import figure, output_file, save
p = figure(title="Oval Plot Example")
p.oval(x=[1, 2, 3], y=[1, 2, 3], width=0.2, height=0.4, color="#CAB2D6", angle=0.5)
output_file('/tmp/bokeh-repo/examples/15_oval_plot.html')
save(p)