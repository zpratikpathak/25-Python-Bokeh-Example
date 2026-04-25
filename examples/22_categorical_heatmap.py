from bokeh.plotting import figure, output_file, save
factors = ["a", "b", "c", "d"]
x = ["a", "b", "c", "d"]
y = ["a", "b", "c", "d"]
p = figure(title="Categorical Heatmap", x_range=factors, y_range=factors)
p.rect(x=x, y=y, width=1, height=1, color="#718dbf", alpha=0.5)
output_file('/tmp/bokeh-repo/examples/22_categorical_heatmap.html')
save(p)