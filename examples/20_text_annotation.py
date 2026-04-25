from bokeh.plotting import figure, output_file, save
p = figure(title="Text Annotation Example")
p.text(x=[1, 2, 3], y=[1, 2, 3], text=["A", "B", "C"], text_color="firebrick", text_font_size="20pt")
output_file('/tmp/bokeh-repo/examples/20_text_annotation.html')
save(p)