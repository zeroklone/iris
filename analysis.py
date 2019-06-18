#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sklearn
from bokeh.plotting import figure, show, save
from bokeh.io import output_file
from bokeh.models import HoverTool

iris_data = pd.read_csv('iris.data', sep=',', header=None)
iris_data.head()
iris_data.columns = ["sepal_length","sepal_width","petal_length","petal_width","class"]
iris_data.loc[lambda x: x['class'] == 'Iris-setosa',['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

setosa = iris_data.loc[lambda x: x['class'] == 'Iris-setosa',['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
versicolour = iris_data.loc[lambda x: x['class'] == 'Iris-versicolor',['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
virginica = iris_data.loc[lambda x: x['class'] == 'Iris-virginica',['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

tools = 'pan,wheel_zoom,box_zoom,reset,hover,save'
plot_title = "Sepal Length & Width"

p = figure(title=plot_title, tools=tools, 
           x_axis_label="Sepal Length",
          y_axis_label="Sepal Width")

hover = p.select(dict(type=HoverTool))
hover.point_policy = 'follow_mouse'

# Sepal

setosa_sepal_plot = p.scatter(setosa.loc[:,'sepal_length'].get_values(),
                        setosa.loc[:,'sepal_width'].get_values(),
                        marker="square", fill_color="red", legend="Iris-setosa"
                       )

versicolour_sepal_plot = p.scatter(versicolour.loc[:,'sepal_length'].get_values(),
                        versicolour.loc[:,'sepal_width'].get_values(),
                        marker="circle", fill_color="blue", legend="Iris-versicolour"
                       )

virginica_sepal_plot = p.scatter(virginica.loc[:,'sepal_length'].get_values(),
                        virginica.loc[:,'sepal_width'].get_values(),
                        marker="triangle", fill_color="yellow", legend="Iris-virginica"
                       )

output_file(filename="sepal_scatter.html", title="Sepal Length & Width")
save(p)

# Plotting Sepal length and width reveals that Iris-setosa can be identified with these two attributes
# Consider usnig these to form a bew feature

plot_title = "Petal Length & Width"

petals = figure(title=plot_title, tools=tools, 
           x_axis_label="Petal Length",
          y_axis_label="Petal Width")

hover = petals.select(dict(type=HoverTool))
hover.point_policy = 'follow_mouse'

setosa_petal_plot = petals.scatter(setosa.loc[:,'petal_length'].get_values(),
                        setosa.loc[:,'petal_width'].get_values(),
                        marker="square", fill_color="red", legend="Iris-setosa"
                       )

versicolour_petal_plot = petals.scatter(versicolour.loc[:,'petal_length'].get_values(),
                        versicolour.loc[:,'petal_width'].get_values(),
                        marker="circle", fill_color="blue", legend="Iris-versicolour"
                       )

virginica_petal_plot = petals.scatter(virginica.loc[:,'petal_length'].get_values(),
                        virginica.loc[:,'petal_width'].get_values(),
                        marker="triangle", fill_color="yellow", legend="Iris-virginica"
                       )


output_file(filename="petal_scatter.html", title="Petal Length & Width")
save(petals)

# Plotting Petal length and width reveals that Iris-setosa can be still be identified with these two attributes
# Consider usnig these to form a new feature. This new attribute can also strongly split versicolour and virginica