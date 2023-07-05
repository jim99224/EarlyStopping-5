# import plotly.plotly as py
from chart_studio import plotly as py
import plotly.graph_objs as go

import numpy as np

import matplotlib

import copy

rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = matplotlib.colors.to_rgb(hex)

show = ['class_average_loss', 'person', 'toaster', 'teddy bear']

def plot_2D(table, x_axis, color):

    data = []
    for idx, (key, val) in enumerate(table.items()):

        if '_not_converged' in key: continue
        if key not in show: continue

        y = copy.deepcopy(val[0])
        z = copy.deepcopy(val[1])
        colors = [color[key] for i in range(len(z))]

        if key+'_not_converged' in table.keys():
            # print(key)
            # print(len(y), len(table[key+'_not_converged'][0]), len(set(y+table[key+'_not_converged'][0])))
            y.extend(table[key+'_not_converged'][0])
            z.extend(table[key+'_not_converged'][1])
            colors.extend([color[key+'_not_converged'] for i in range(len(z))])
            y, z, colors = zip(*sorted(zip(y, z, colors)))

        trace = go.Scatter(
            x=list(y[::10])+[y[-1]],
            y=list(z[::10])+[z[-1]],
            mode='lines+markers',
            name = key,
            marker=dict(
                size=5,
                color=list(colors[::10])+[colors[-1]],                # set color to an array/list of desired values
                # colorscale='Viridis',   # choose a colorscale
                # opacity=0.8
            ),
            line=dict(
                width=2,
                color=color[key],
                # colorscale='Viridis',   # choose a colorscale
                # opacity=0.8
            )
        )
        data.append(trace)

    trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name="not converged",
        marker=dict(
            size=5,
            color='rgb(0,0,0)',                # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            # opacity=0.8
        ),
    )
    data.append(trace)


    # data = [trace1]
    layout = go.Layout(
        title='ES_5(subclass analysis)',
        showlegend=True,
        # xaxis_visible=False,
        # legend=dict(sss=''),
        legend=dict(title_font_family="Times New Roman",
                          font=dict(size= 20)), 
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100
        ),
        xaxis=dict(
            title="iterations"
        ),
        yaxis=dict(
            title="loss"
        )
    )
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
                    title=dict(text="ES_5(subclass analysis)", font=dict(size=50)),
                    title_font_family="Times New Roman",
                    legend=dict(title_font_family="Times New Roman",
                          font=dict(size= 20)), 
                    font=dict(
                        family="Times New Roman",
                        size=20
                    )
                    )

    # fig.update_yaxes(title='y', visible=False, showticklabels=False)
    # fig.update_layout(legend = dict(font = dict(family = "Courier", size = 50)),
    #               legend_title = dict(font = dict(family = "Courier", size = 30)))
    fig.write_html('ES_5(MNIST)_2D.html', auto_open=False)
# py.iplot(fig, filename='3d-scatter-colorscale')