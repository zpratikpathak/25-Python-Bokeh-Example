from bokeh.plotting import figure, output_file, save
from math import pi
import pandas as pd
from bokeh.transform import cumsum
x = { 'United States': 157, 'United Kingdom': 93, 'Japan': 89, 'China': 63, 'Germany': 44, 'India': 114 }
data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c']
p = figure(height=350, title="Pie Chart", toolbar_location=None, tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))
p.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), line_color="white", fill_color='color', legend_field='country', source=data)
output_file('/tmp/bokeh-repo/examples/4_pie_chart.html')
save(p)