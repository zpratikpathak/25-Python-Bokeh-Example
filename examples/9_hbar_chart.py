from bokeh.plotting import figure, output_file, save
factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x = [50, 40, 65, 10, 25, 37, 80, 60]
p = figure(y_range=factors, title="Horizontal Bar Chart")
p.hbar(y=factors, right=x, height=0.5, color="green")
output_file('/tmp/bokeh-repo/examples/9_hbar_chart.html')
save(p)