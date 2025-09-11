from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.colors as pc

import pandas as pd
import numpy as np
import re 
import os 
import sys
import base64
import dash_daq as daq







MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if MAIN_DIRECTORY not in sys.path:
    sys.path.insert(0, str(MAIN_DIRECTORY))

from utils.helpers import retrieve_results

img_path = os.path.join(MAIN_DIRECTORY,"dashboard_app", "train_test_legend.png")
with open(img_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()
img_source = "data:image/png;base64," + encoded

results = retrieve_results()  # type: ignore
expn_uniq = results["Experiment_name"].unique().tolist() # type:ignore
modn_uniq = results["Model_name"].unique().tolist() # type:ignore





app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    # Title
    html.H1(" Experiment Tracking in Dash"),
    

    # Tabs for the exp and model filters and HP filters
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Select experiments and models', value='tab-1'),
        dcc.Tab(label='Filter hyperparameters', value='tab-2'),]),
    html.Div(id='tabs-content',children=html.Div(dbc.Row([
                    dbc.Col(dcc.Dropdown(id="exp_dropdown",options=expn_uniq, 
                                         value=expn_uniq[0],multi=True)),
                    dbc.Col(dcc.Dropdown(id="mod_dropdown",options=modn_uniq, 
                                         value=modn_uniq[0],multi=True)),
                    ]))),


    # Row containing the train test legend, plot title and legend toggle
    dbc.Row([
        # train test legend image
        dbc.Col(html.Div(html.Img(
            src=img_source,   # place image in assets/ folder
            style={
                "width": "5.5rem",      # control size
                "height": "auto",     # keep aspect ratio
                "marginLeft": "4rem",
                "marginTop": "1rem",
            }),
            style={"display": "flex",
                   #"justifyContent": "center",
                   "alignItems": "center",
                   #"backgroundColor": "black"
            }),width=4),

        # Title
        dbc.Col(html.Div(" Training and testing accuracy",
                         style={'textAlign': 'center','width': '100%',
                                "fontWeight": "bold", "fontSize": "18px"}
                         ),
                        width=4,
                        style={'display': 'flex',
                               'justifyContent': 'center',
                               'alignItems': 'end',
                               #"backgroundColor":"red"
                               }
                ),

        # Legend toggle
        dbc.Col(html.Div([daq.BooleanSwitch(id='acc_legend_toggle', on=False,),
                          html.Div("Legend",style={"marginLeft": "0.25rem",
                                                   "marginBottom": "0.2rem"})],
                         style={'display': 'flex','justifyContent': 'flex-end',
                                'alignItems': 'end','height': '100%',
                                'margin-right':'2.5vw',
                                #"backgroundColor": "blue"
                                }),
                width=4)
        ]),
    dcc.Graph(id="acc_plot"), #will be replaced by the plot acc callback


    # Row containing the train test legend, plot title and legend toggle
    dbc.Row([
        # train test legend image
        dbc.Col(html.Div(html.Img(
            src=img_source,   # place image in assets/ folder
            style={
                "width": "5.5rem",      # control size
                "height": "auto",     # keep aspect ratio
                "marginLeft": "4rem",
                "marginTop": "1rem",
            }),
            style={"display": "flex",
                   #"justifyContent": "center",
                   "alignItems": "center",
                   #"backgroundColor": "black"
            }),width=4),

        # Title
        dbc.Col(html.Div(" Training and testing loss",
                         style={'textAlign': 'center','width': '100%',
                                "fontWeight": "bold", "fontSize": "18px"}
                         ),
                        width=4,
                        style={'display': 'flex',
                               'justifyContent': 'center',
                               'alignItems': 'end',
                               #"backgroundColor":"red"
                               }
                ),

        # Legend toggle
        dbc.Col(html.Div([daq.BooleanSwitch(id='loss_legend_toggle', on=False,),
                          html.Div("Legend",style={"marginLeft": "0.25rem",
                                                   "marginBottom": "0.2rem"})],
                         style={'display': 'flex','justifyContent': 'flex-end',
                                'alignItems': 'end','height': '100%',
                                'margin-right':'2.5vw',
                                #"backgroundColor": "blue"
                                }),
                width=4)
        ]),
    dcc.Graph(id="loss_plot"),
    dcc.Store(id="selected_exps"),
    dcc.Store(id="selected_mods"),
])


#Stores chosen experiments
@callback(Output("selected_exps","data"),
          Input("exp_dropdown","value"))
def store_exps(exp_chosen):
    if isinstance(exp_chosen, str):
        exp_chosen = [exp_chosen]
    df = pd.DataFrame(exp_chosen)
    return df.to_dict()

#Stores chosen models
@callback(Output("selected_mods","data"),
          Input("mod_dropdown","value"))
def store_mods(mod_chosen):
    if isinstance(mod_chosen, str):
        mod_chosen = [mod_chosen]
    df = pd.DataFrame(mod_chosen)
    return df.to_dict()




# Switches between the different tabs
@callback(Output('tabs-content', 'children'),
            Input('tabs', 'value'),
            Input("selected_exps","data"),
            Input("selected_mods","data"),)
def render_content(tab,exps_dict,mods_dict):
    if tab == 'tab-1':
        try: 
            exps = [exp for exp in exps_dict["0"].values()]
        except KeyError:
            exps = []
        try:
            mods = [exp for exp in mods_dict["0"].values()]
        except KeyError:
            mods = []
        return [
            html.Br(),
            dbc.Row([dbc.Col("Experiments"),dbc.Col("Models")]),
            html.Div(dbc.Row([
                    dbc.Col(dcc.Dropdown(id="exp_dropdown",options=expn_uniq, value=exps,multi=True)),
                    dbc.Col(dcc.Dropdown(id="mod_dropdown",options=modn_uniq, value=mods,multi=True)),
                    ]))]
    elif tab == 'tab-2':
        return [html.Div("Filter Hyperparameters"),html.Br(),
                html.Div("Add multiple range sliders, one for every "
                "hyperparameter. Need to think how I select what is a filterable"
                "HP and what isn't. A hyperparameter is any column before "
                "train_loss. Work in progress.")]


# Updates available models based on selected experiments
@callback(Output("mod_dropdown","options"),
            Input("exp_dropdown", "value"))
def update_models(exp_chosen):
    # If only one experiment is selected
    if isinstance(exp_chosen, str):
        exp_chosen = [exp_chosen]
    filtered_exps = results.query(f"Experiment_name == {list(exp_chosen)}")  # type:ignore
    model_choices = [mod for mod in filtered_exps["Model_name"].unique().tolist()]
    return model_choices


@callback(Output(component_id="acc_plot",component_property="figure"),
          Input(component_id="acc_legend_toggle",component_property="on"),
          Input("exp_dropdown", "value"),
          Input("mod_dropdown", "value"),)
def plot_acc(acc_legend,exp_chosen,mod_chosen):
    color_palette = pc.qualitative.Dark24
    colors = iter(color_palette)
    # Filter the results based on the selected experiments and models
    if isinstance(exp_chosen, str):
        exp_chosen = [exp_chosen]    
    if isinstance(mod_chosen, str):
        mod_chosen = [mod_chosen]
    query = f"Experiment_name == {exp_chosen} & "
    query += f"Model_name == {mod_chosen}"
    filtered_res = results.query(query)  # type: ignore
    fig = go.Figure()

    for id in filtered_res["ID"].unique():
        df_id = filtered_res[filtered_res["ID"] == id]
        expn = df_id["Experiment_name"].unique()[0]
        modn = df_id["Model_name"].unique()[0]
        extra = df_id["Extra"].unique()[0]
        #remove iter_n from extra to remove from the legend name
        x = re.sub("Iter_[0-9]+", "", extra)
        x = re.sub("_$", "", x)
        x = re.sub("^_", "", x)

        ht_template=f"<b>Experiment: </b>{expn}<br><b>Model: </b>{modn}<br>"
        legtitle = "Experiment <b> Model </b> Hyperparameters Iteration<br>"
        color = next(colors)

        # Check if an experiment has been repeated for multiple 
        # iterations. All iterations are plotted the same color

        try: # Check if a column named Iter # exists
            num_iter = df_id["Iter #"].max()
            iter_flag = True
            # The column may exist in the results dataframe but the value 
            # can be NaN. 
            if np.isnan(num_iter):
                num_iter = 1
                iter_flag = False
            else:
                num_iter = int(num_iter)
        except KeyError:
            num_iter = 1
            iter_flag = False

        for it in range(num_iter):
            if iter_flag:
                name = f"{expn} <b>{modn}</b> {x} Iter {it+1}"
                hovertemp = ht_template + f"<b>HP: </b> {extra} <extra> Iter {it+1} </extra>"
                df = df_id[df_id["Iter #"] == it+1]
            else:
                name = f"{expn} <b>{modn}</b> {extra}"
                hovertemp = ht_template + f"<b>HP: </b> {extra} <br><extra></extra>"
                df = df_id

            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["train_acc"],
                                    name=name, 
                                    line={"color": color}, 
                                    hovertemplate=hovertemp))
            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["test_acc"],
                                        name="", 
                                        line={"color": color, "dash": "dash"}, 
                                        hovertemplate=hovertemp))
        #title = {
        # 'text': "Training and testing accuracy",
        # 'y':0.9, # new
        # 'x':0.5,
        # 'xanchor': 'center',
        # 'yanchor': 'top' # new
        #}
        fig.update_layout(xaxis_title="Epochs", yaxis_title="Accuracy",
                            showlegend=acc_legend,
                            legend_title={"text": legtitle},
                            #title = title,
                            margin=dict(l=40, r=40, t=5, b=40),
                            )

    #fig.add_layout_image(dict(
    #    source=img_source,
    #    xref="paper", yref="paper",
    #    x=0.01, y=1.0,
    #    sizex=1, sizey=0.1,
    #    xanchor="left", yanchor="bottom",
    #    sizing="contain", opacity=1.0,
    #    layer="above"
    #))
    return fig

@callback(Output(component_id="loss_plot",component_property="figure"),
          Input(component_id="loss_legend_toggle",component_property="on"),
          Input("exp_dropdown", "value"),
          Input("mod_dropdown", "value"),)
def plot_loss(loss_legend,exp_chosen,mod_chosen):
    color_palette = pc.qualitative.Dark24
    colors = iter(color_palette)

    if isinstance(exp_chosen, str):
        exp_chosen = [exp_chosen]    
    if isinstance(mod_chosen, str):
        mod_chosen = [mod_chosen]
    query = f"Experiment_name == {exp_chosen} & "
    query += f"Model_name == {mod_chosen}"
    filtered_res = results.query(query)  # type: ignore
    fig = go.Figure()

    for id in filtered_res["ID"].unique():
        df_id = filtered_res[filtered_res["ID"] == id]
        expn = df_id["Experiment_name"].unique()[0]
        modn = df_id["Model_name"].unique()[0]
        extra = df_id["Extra"].unique()[0]

        x = re.sub("Iter_[0-9]+", "", extra)
        x = re.sub("_$", "", x)
        x = re.sub("^_", "", x)

        ht_template=f"<b>Experiment: </b>{expn}<br><b>Model: </b>{modn}<br>"
        legtitle ="Experiment <b> Model </b> Hyperparameters Iteration <br>"
        color = next(colors)

        try:
            num_iter = df_id["Iter #"].max()
            iter_flag = True
            if np.isnan(num_iter):
                num_iter = 1
                iter_flag = False
            else:
                num_iter = int(num_iter)
        except KeyError:
            num_iter = 1
            iter_flag = False

        for it in range(num_iter):
            if iter_flag:
                name = f"{expn} <b>{modn}</b> {x} Iter {it+1}"
                hovertemp = ht_template + f"<b>HP: </b> {extra} <extra> Iter {it+1} </extra>"
                df = df_id[df_id["Iter #"] == it+1]
            else:
                name = f"{expn} <b>{modn}</b> {extra}"
                hovertemp = ht_template + f"<b>HP: </b> {extra} <br><extra></extra>"
                df = df_id

            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["train_loss"],
                                        name=name, 
                                        line={"color": color}, 
                                        hovertemplate=hovertemp))
            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["test_loss"],
                                        name="", 
                                        line={"color": color, "dash": "dash"}, 
                                        hovertemplate=hovertemp))

        fig.update_layout(xaxis_title="Epochs", 
                            yaxis_title="Average loss per batch",
                            showlegend=loss_legend, 
                            legend_title={"text": legtitle},
                            margin=dict(l=40, r=40, t=5, b=40))
                            #title = "Training and testing loss",
                            #title_x=0.5)

    #fig.add_layout_image(dict(
    #    source=img_source,
    #    xref="paper", yref="paper",
    #    x=0.01, y=1.0,
    #    sizex=1, sizey=0.1,
    #    xanchor="left", yanchor="bottom",
    #    sizing="contain", opacity=1.0,
    #    layer="above"
    #))    
    return fig


if __name__ == '__main__':
    app.run(debug=True)
'''
"#F7F7F7" #white
"#EAEAEA" #light gray
"#2A3F54" #dark blue
'''
