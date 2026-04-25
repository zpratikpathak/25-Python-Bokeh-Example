from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 3, 2]
y = [2, 1, 1, 2, 3, 3]
p = figure(title="Patch Plot Example")
p.patch(x, y, alpha=0.5, line_width=2, fill_color="firebrick")
output_file('/tmp/bokeh-repo/examples/11_patch_plot.html')
save(p)