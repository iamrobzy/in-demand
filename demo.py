import gradio as gr
import plotly.graph_objects as go
import numpy as np

# Function to create a 3D plot
def create_3d_plot(x_range, y_range):
    # Generate 3D data
    x = np.linspace(-x_range, x_range, 100)
    y = np.linspace(-y_range, y_range, 100)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))
    
    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=1000    
    )
    return fig

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Interactive 3D Plot with Gradio and Plotly")
    with gr.Row():
        x_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="X Range")
        y_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Y Range")
    plot_output = gr.Plot(label="3D Surface Plot")
    
    # Update the plot on slider change
    x_slider.change(create_3d_plot, inputs=[x_slider, y_slider], outputs=plot_output)
    y_slider.change(create_3d_plot, inputs=[x_slider, y_slider], outputs=plot_output)

# Launch the app
demo.launch()
