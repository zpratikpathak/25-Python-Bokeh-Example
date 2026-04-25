from bokeh.plotting import figure, output_file, save
p = figure(title="Multi-Line Example")
p.multi_line(xs=[[1, 2, 3], [2, 3, 4]], ys=[[2, 1, 4], [4, 3, 5]], color=["firebrick", "navy"], line_width=2)
output_file('/tmp/bokeh-repo/examples/18_multi_line.html')
save(p)