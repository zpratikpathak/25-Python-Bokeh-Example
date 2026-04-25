from bokeh.plotting import figure, output_file, save
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]
p = figure(x_range=fruits, height=350, title="Fruit Counts", toolbar_location=None, tools="")
p.vbar(x=fruits, top=counts, width=0.9)
p.xgrid.grid_line_color = None
p.y_range.start = 0
output_file('/tmp/bokeh-repo/examples/3_bar_chart.html')
save(p)