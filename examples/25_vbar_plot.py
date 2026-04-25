from bokeh.plotting import figure, output_file, save
p = figure(title="VBar Example")
p.vbar(x=[1, 2, 3], width=0.5, bottom=0, top=[1.2, 2.5, 3.7], color="firebrick")
output_file('/tmp/bokeh-repo/examples/25_vbar_plot.html')
save(p)