from bokeh.plotting import figure, output_file, save
p = figure(title="Annulus Plot")
p.annulus(x=[1, 2, 3], y=[1, 2, 3], color="#7FC97F", inner_radius=0.1, outer_radius=0.25)
output_file('/tmp/bokeh-repo/examples/13_annulus.html')
save(p)