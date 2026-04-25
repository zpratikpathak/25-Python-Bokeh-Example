from bokeh.plotting import figure, output_file, save
p = figure(title="Multi-Polygon Example")
p.multi_polygons(xs=[[[[1, 2, 2, 1]]]], ys=[[[[1, 1, 2, 2]]]], color="olive", alpha=0.5)
output_file('/tmp/bokeh-repo/examples/19_multi_polygon.html')
save(p)