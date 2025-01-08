import plotly.express as px
import numpy as np

X = np.random.randint(0, 10, (10, 3))
fig = px.scatter_3d(x=X[:,0], y=X[:, 1], z=X[:, 2])
fig.show() 
