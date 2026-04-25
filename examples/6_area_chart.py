from bokeh.plotting import figure, output_file, save
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 5, 8, 4]
y2 = [1, 2, 4, 6, 3]
p = figure(title="Area Chart")
p.varea(x=x, y1=y1, y2=y2, fill_color="blue", alpha=0.3)
output_file('/tmp/bokeh-repo/examples/6_area_chart.html')
save(p)