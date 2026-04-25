from bokeh.plotting import figure, output_file, save
p = figure(title="HBar Example")
p.hbar(y=[1, 2, 3], height=0.5, left=0, right=[1.2, 2.5, 3.7], color="navy")
output_file('/tmp/bokeh-repo/examples/26_hbar_plot.html')
save(p)