<p>
  <img width="200" align='Right' src="./Python Bokeh examples.jpg">
</p>

Python Bokeh is one of the best Python packages for data visualization. Today we are going to see some Python Bokeh Examples. I have also provided the Python Bokeh project source code GitHub. Learn this easy visualization tool and add it to your Python stack.

## What is Python Bokeh?

Python [Bokeh](https://bokeh.org/) is a data visualization tool or we can also say Python Bokeh is used to plot various types of graphs. There are various other graph plotting libraries like matplotlib but Python Bokeh graphs are dynamic in nature means you can interact with the generated graph. See the below examples…

## Installation 💻:

Python Bokeh can be easily installed using PIP. You can install the Python Bokeh easily by running the command:

```
pip install bokeh
```

Now everything is ready let’s go through the examples 🏃‍♂️…

## 1. LinePlot

```
from bokeh.plotting import figure, show, output_notebook
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="{LinePlot Python Bokeh Example")
p.line(x, y, line_width=2)
output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 2. **Scatter Plot**

```
from bokeh.plotting import figure, show, output_notebook
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Scatter Plot Python Bokeh Example by PratikPathak.com")
p.circle(x, y, size=10, color="navy", alpha=0.5)
output_notebook()
show(p)
```

## 3. **Bar Chart**

```
from bokeh.plotting import figure, show, output_notebook
categories = ["A", "B", "C", "D", "E"]
values = [10, 15, 8, 12, 6]
p = figure(x_range=categories, title="Bar Chart Python Bokeh Example by PratikPathak.com")
p.vbar(x=categories, top=values, width=0.9)
output_notebook()
show(p)
```

## 4. **Histogram**

```
from bokeh.plotting import figure, show, output_notebook
import numpy as np
data = np.random.normal(0, 1, 1000)
p = figure(title="Histogram Python Bokeh Example by PratikPathak.com")
p.hist(data, bins=30, color="navy", alpha=0.5)
output_notebook()
show(p)
```

## 5. **Pie Chart**

```
from bokeh.plotting import figure, show, output_notebook
labels = ["A", "B", "C", "D"]
values = [10, 15, 8, 12]
p = figure(title="Pie Chart Python Bokeh Example by PratikPathak.com")
p.wedge(x=0, y=0, radius=0.4, start_angle=0.6, end_angle=2.6, color=["red", "green", "blue", "yellow"], legend_label=labels)
output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 6. **Time Series Plot**

```
from bokeh.plotting import figure, show, output_notebook
from datetime import datetime, timedelta
start = datetime(2023, 1, 1)
end = start + timedelta(days=30)
x = [start + timedelta(days=i) for i in range((end-start).days)]
y = [10, 15, 8, 12, 6, 18, 9, 14, 7, 11, 5, 16, 8, 13, 6]
p = figure(x_axis_type="datetime", title="Time Series Plot Python Bokeh Example by PratikPathak.com")
p.line(x, y, line_width=2)
output_notebook()
show(p)
```

## 7. **Linked Brushing**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
x1 = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 3]
x2 = [2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 4]
source = ColumnDataSource(data=dict(x1=x1, y1=y1, x2=x2, y2=y2))
p1 = figure(title="Scatter Plot 1")
p1.circle('x1', 'y1', source=source)
p2 = figure(title="Scatter Plot 2 Python Bokeh Example by PratikPathak.com")
p2.circle('x2', 'y2', source=source)
output_notebook()
show(p1, p2)
```

## 8. **Hover Tooltips**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Hover Tooltips Python Bokeh Example by PratikPathak.com")
p.circle(x, y, size=15, fill_color="navy", line_color="white", alpha=0.5)
hover = HoverTool(tooltips=[("(x,y)", "(@x, @y)")])
p.add_tools(hover)
output_notebook()
show(p)
```

## 9. **Annotations**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Arrow, VectorRenderer, Label
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Annotations Python Bokeh Example by PratikPathak.com")
p.circle(x, y, size=15, fill_color="navy", line_color="white", alpha=0.5)
arrow = Arrow(x_start=2, y_start=4, x_end=3, y_end=5, line_width=2, line_color="red")
label = Label(x=3, y=6, text="This is a label", render_mode='css', border_line_color='black', border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=0.5)
p.add_layout(arrow)
p.add_layout(label)
output_notebook()
show(p)
```

## 10. **Custom Glyphs**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models.glyphs import Asterisk
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Custom Glyphs Python Bokeh Example by PratikPathak.com")
p.add_glyph(x, y, Asterisk(size=20, line_color="red", fill_color="yellow"))
output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 11. **Gridlines and Axes**

```
from bokeh.plotting import figure, show, output_notebook
x = [1, 2, 3, 4, 5]
y = [4, 5, 5, 7, 3]
p = figure(title="Gridlines and Axes Python Bokeh Example by PratikPathak.com", x_range=(0, 6), y_range=(0, 8))
p.grid.grid_line_color = "grey"
p.grid.grid_line_dash = [6, 4]
p.xaxis.axis_label = "X-axis"
p.yaxis.axis_label = "Y-axis"
p.line(x, y, line_width=2)
output_notebook()
show(p)
```

## 12. **Legend**

```
from bokeh.plotting import figure, show, output_notebook
x1 = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 3]
x2 = [2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 4]
p = figure(title="Legend Python Bokeh Example by PratikPathak.com")
p.line(x1, y1, line_width=2, color="red", legend_label="Line 1")
p.line(x2, y2, line_width=2, color="blue", legend_label="Line 2")
p.legend.location = "top_left"
output_notebook()
show(p)
```

## 13. **Categorical Plots**

```
from bokeh.plotting import figure, show, output_notebook
fruits = ["Apples", "Pears", "Nectarines", "Plums", "Grapes", "Strawberries"]
counts = [5, 3, 4, 2, 4, 6]
p = figure(x_range=fruits, title="Categorical Plots Python Bokeh Example by PratikPathak.com")
p.vbar(x=fruits, top=counts, width=0.9)
p.xaxis.axis_label = "Fruit"
p.yaxis.axis_label = "Count"
p.xaxis.axis_label_text_font_size = "12pt"
p.yaxis.axis_label_text_font_size = "12pt"
output_notebook()
show(p)
```

## 14. **Subplots**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
x1 = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 3]
x2 = [2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 4]
p1 = figure(title="Scatter Plot 1 Python Bokeh Example by PratikPathak.com")
p1.circle(x1, y1, size=10, color="navy", alpha=0.5)
p2 = figure(title="Scatter Plot 2 Python Bokeh Example by PratikPathak.com")
p2.circle(x2, y2, size=10, color="red", alpha=0.5)
grid = gridplot([[p1, p2]], plot_width=400, plot_height=400)
output_notebook()
show(grid)
```

## 15. **Interactive Plots**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool
import numpy as np
x = np.random.random(1000)
y = np.random.random(1000)
source = ColumnDataSource(data=dict(x=x, y=y))
p = figure(title="Interactive Plots Python Bokeh Example by PratikPathak.com", tools="hover,box_select")
p.circle('x', 'y', source=source, size=3, color="navy", alpha=0.5)
hover = HoverTool(tooltips=[("(x,y)", "(@x, @y)")])
p.add_tools(hover)
output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 16. **Linked Panning and Zooming**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d
x1 = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 3]
x2 = [2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 4]
p1 = figure(title="Scatter Plot 1 Python Bokeh Example by PratikPathak.com", x_range=Range1d(0, 6), y_range=Range1d(0, 8))
p1.circle(x1, y1, size=10, color="navy", alpha=0.5)
p2 = figure(title="Scatter Plot 2 Python Bokeh Example by PratikPathak.com", x_range=p1.x_range, y_range=p1.y_range)
p2.circle(x2, y2, size=10, color="red", alpha=0.5)
output_notebook()
show(p1, p2)
```

## 17. **Linked Axis Ranges**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d
x1 = [1, 2, 3, 4, 5]
y1 = [4, 5, 5, 7, 3]
x2 = [2, 3, 4, 5, 6]
y2 = [2, 4, 6, 8, 4]
p1 = figure(title="Scatter Plot 1", x_range=(0, 6), y_range=(0, 8))
p1.circle(x1, y1, size=10, color="navy", alpha=0.5)
p2 = figure(title="Scatter Plot 2", x_range=p1.x_range, y_range=p1.y_range)
p2.circle(x2, y2, size=10, color="red", alpha=0.5)
output_notebook()
show(p1, p2)
```

## 18. **Datetime Axis**

```
from bokeh.plotting import figure, show, output_notebook
from datetime import datetime, timedelta
start = datetime(2023, 1, 1)
end = start + timedelta(days=30)
x = [start + timedelta(days=i) for i in range((end-start).days)]
y = [10, 15, 8, 12, 6, 18, 9, 14, 7, 11, 5, 16, 8, 13, 6]
p = figure(x_axis_type="datetime", title="Datetime Axis Python Bokeh Example by PratikPathak.com")
p.line(x, y, line_width=2)
output_notebook()
show(p)
```

## 19. **Heatmap**

```
from bokeh.plotting import figure, show, output_notebook
import numpy as np
data = np.random.rand(10, 10)
p = figure(title="Heatmap Python Bokeh Example by PratikPathak.com")
p.image(image=[data], x=0, y=0, dw=10, dh=10, palette="Viridis256")
output_notebook()
show(p)
```

## 20. **Streamgraph**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
import numpy as np
# Generate some sample data
x = np.linspace(0, 10, 100)
y1 = np.cumsum(np.random.randn(100))
y2 = np.cumsum(np.random.randn(100))
y3 = np.cumsum(np.random.randn(100))
y4 = np.cumsum(np.random.randn(100))
y5 = np.cumsum(np.random.randn(100))
source = ColumnDataSource(data=dict(x=x, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5))
p = figure(title="Streamgraph Python Bokeh Example by PratikPathak.com")
p.area(x='x', y1='y1', y2='y2', y3='y3', y4='y4', y5='y5', source=source, line_color=None)
output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 21. **Polar Plot**

```
from bokeh.plotting import figure, show, output_notebook
import numpy as np
r = np.linspace(0.1, 1, 10)
theta = np.linspace(0, 2*np.pi, 10)
p = figure(title="Polar Plot Python Bokeh Example by PratikPathak.com", plot_width=600, plot_height=600, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), toolbar_location=None)
p.polar(r, theta, color="navy", alpha=0.5, line_width=2)
output_notebook()
show(p)
```

## 22. **Donut Chart**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
labels = ["A", "B", "C", "D"]
values = [10, 15, 8, 12]
source = ColumnDataSource(data=dict(labels=labels, values=values))
p = figure(title="Donut Chart Python Bokeh Example by PratikPathak.com", x_range=labels)
p.wedge(x=0, y=1, radius=0.4, start_angle=0.6, end_angle=2.6, color=["red", "green", "blue", "yellow"], legend_field="labels")
p.wedge(x=0, y=1, inner_radius=0.3, outer_radius=0.4, start_angle=0.6, end_angle=2.6, color="white")
p.axis.visible = False
p.grid.grid_line_color = None
output_notebook()
show(p)
```

## 23. **Treemap**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import TreemapRenderer, ColumnDataSource
import pandas as pd

# Sample data
data = pd.DataFrame({
    'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'parent': ['', 'A', 'A', 'B', 'B', 'C', 'C', 'D'],
    'value': [10, 5, 4, 3, 2, 1, 1, 1]
})

source = ColumnDataSource(data)
p = figure(title="Treemap Python Bokeh Example by PratikPathak.com")
treemap = TreemapRenderer(
    source=source,
    label='label',
    parent='parent',
    value='value',
    color_discrete_sequence=["#e6550d", "#3182bd", "#31a354", "#756bb1", "#636363", "#9e9ac8", "#de2d26", "#fd8d3c"],
)
p.add_layout(treemap)
output_notebook()
show(p)
```

 If you get any error make sure you have installed python pandas by using “pip install pandas” command 

## 24. **Network Plot**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, NodesAndLinkedEdges
import networkx as nx

# Sample data
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F')]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

node_positions = nx.spring_layout(G)
node_x = [node_positions[node][0] for node in G.nodes()]
node_y = [node_positions[node][1] for node in G.nodes()]
node_color = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

source = ColumnDataSource(data=dict(
    x=node_x,
    y=node_y,
    color=node_color,
    node=nodes
))

p = figure(title="Network Plot Python Bokeh Example by PratikPathak.com", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1))
p.add_tools(HoverTool(tooltips=[("Node", "@node")]))
p.circle('x', 'y', size=15, color='color', source=source)

edge_x = []
edge_y = []
for start, end in edges:
    x0, y0 = node_positions[start]
    x1, y1 = node_positions[end]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

p.multi_line(edge_x, edge_y, color="black", alpha=0.5, line_width=1)

output_notebook()
show(p)
```

## 25. **Sankey Diagram**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Sankey, ColumnDataSource

# Sample data
flows = [
    ('A', 'X', 5),
    ('B', 'X', 3),
    ('A', 'Y', 2),
    ('B', 'Y', 4),
    ('X', 'C', 4),
    ('X', 'D', 3),
    ('Y', 'C', 2),
    ('Y', 'D', 4)
]

source = ColumnDataSource(data=dict(
    source=[flow[0] for flow in flows],
    target=[flow[1] for flow in flows],
    value=[flow[2] for flow in flows]
))

p = figure(plot_width=800, plot_height=400, title="Sankey Diagram Python Bokeh Example by PratikPathak.com")
sankey = Sankey(source=source, label_text_color="white", label_text_font_size="8pt")
p.renderers.append(sankey)

output_notebook()
show(p)
```

[Live Preview](https://colab.research.google.com/drive/1hQD7qMu5rbhV8JQ6gZ61drGiTQfUOyTE?usp=sharing) | [Source Code](https://github.com/zpratikpathak/25-Python-Bokeh-Example) | [🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## 26. **Chord Diagram**

```
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, CategoricalColorMapper
import numpy as np

# Sample data
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
connections = [
    ('A', 'B', 0.5), ('A', 'C', 0.3), ('A', 'D', 0.2),
    ('B', 'C', 0.4), ('B', 'D', 0.3), ('B', 'E', 0.3),
    ('C', 'D', 0.2), ('C', 'E', 0.3), ('C', 'F', 0.5),
    ('D', 'E', 0.2), ('D', 'F', 0.3), ('E', 'F', 0.4)
]

source = ColumnDataSource(data=dict(
    start=np.array([con[0] for con in connections]),
    end=np.array([con[1] for con in connections]),
    value=np.array([con[2] for con in connections])
))

p = figure(title="Chord Diagram Python Bokeh Example by PratikPathak.com", plot_width=800, plot_height=800)
chord = p.chord(
    source=source, 
    start_angle=np.pi / 2,
    direction="clockwise",
    colors=CategoricalColorMapper(factors=nodes, palette="Spectral11")
)

output_notebook()
show(p)
```

## How to Contribute?

Feel free to open a PR request on our GitHub repo.

Steps to contribute:

1. Fork the repo
2. Make changes in the Forked repo and save
3. Open a Pull Request
4. That’s it 😄!

[🌿 Contribute](https://github.com/zpratikpathak/25-Python-Bokeh-Example)

## Conclusion 

In this Repo I have shared you 25+ Python Bokeh examples which can help you to learn python bokeh. Feel free to contribute to our github repo and keep it updated.

Originally Published at [25+ Python Bokeh Example](https://pratikpathak.com/25-python-bokeh-example-learn-bokeh-from-examples/)