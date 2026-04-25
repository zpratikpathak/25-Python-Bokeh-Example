from bokeh.plotting import figure, output_file, save
fruits = ['Apples', 'Pears', 'Nectarines']
years = ["2015", "2016", "2017"]
colors = ["#c9d9d3", "#718dbf", "#e84d60"]
data = {'fruits': fruits, '2015': [2, 1, 4], '2016': [5, 3, 4], '2017': [3, 2, 4]}
p = figure(x_range=fruits, height=350, title="Fruit Counts by Year", toolbar_location=None, tools="")
p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data, legend_label=years)
output_file('/tmp/bokeh-repo/examples/12_stacked_bar.html')
save(p)