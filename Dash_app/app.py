from dash import Dash, dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_daq as daq

import plotly.graph_objects as go
import plotly.colors as pc

import pandas as pd
import numpy as np
import base64
import sys
import re 
import os



############## Setup ##############

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if MAIN_DIRECTORY not in sys.path:
    sys.path.insert(0, str(MAIN_DIRECTORY))

from utils.helpers import retrieve_results

img_path = os.path.join(MAIN_DIRECTORY,"Dash_app",
                        "assets","train_test_legend.png")
with open(img_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()
img_source = "data:image/png;base64," + encoded

results = retrieve_results()  # type: ignore
expn_uniq = results["Experiment_name"].unique().tolist() # type:ignore
modn_uniq = results["Model_name"].unique().tolist() # type:ignore

###################################


app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)


############## Layout ##############

app.layout = html.Div(children=[
    # Title
    html.H1(" Experiment Tracking in Dash"),
    

    # Tabs for the exp and model selector and HP filter
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Select experiments and models', value='tab-1'),
        dcc.Tab(label='Filter results', value='tab-2'),
        ]),
    html.Div(id='tabs-content',children=[html.Br(),
            dbc.Row([dbc.Col("Experiments"),dbc.Col("Models")],style={"marginLeft": "1.5rem"}),
            html.Div(dbc.Row([
                    dbc.Col(dcc.Dropdown(id="exp_dropdown",options=expn_uniq, 
                                         value=expn_uniq[0],multi=True)),
                    dbc.Col(dcc.Dropdown(id="mod_dropdown",options=modn_uniq, 
                                         value=modn_uniq[0],multi=True)),
                    ]),style={"marginLeft": "1.5rem",
                              "marginRight": "1.5rem"})]),

    # Row with the train test legend, plot title and legend toggle for accuracy
    dbc.Row([

        # train test legend image
        dbc.Col(html.Div(html.Img(
            src=img_source,
            style={
                "width": "5.5rem",
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
    
    # Accuracy plot
    dcc.Graph(id="acc_plot"), 


    #Row with the train test legend, plot title and legend toggle for loss
    dbc.Row([

        # train test legend image
        dbc.Col(html.Div(html.Img(
            src=img_source,
            style={
                "width": "5.5rem",
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

    # Loss plot
    dcc.Graph(id="loss_plot"),
    html.Br(),
    # Components to store the selected models and experiments so they are not 
    # lost when switching tabs. 
    dcc.Store(id="selected_exps"),
    dcc.Store(id="selected_mods"),
])



############## Callbacks ##############

# Switches between the different tabs
@callback(Output('tabs-content', 'children'),
            Input('tabs', 'value'),
            State("selected_exps","data"),
            State("selected_mods","data"),
            prevent_initial_call=True
            )
def render_content(tab,exps_dict,mods_dict):
    #print("SWITCH TABS")
    if tab == 'tab-1':
        # the try is checking for empty stored models or experiments
        try: 
            exps = [exp for exp in exps_dict["0"].values()]
        except KeyError:
            exps = []
        try:
            mods = [exp for exp in mods_dict["0"].values()]
        except KeyError:
            mods = []
        return [html.Br(),
            dbc.Row([dbc.Col("Experiments"),dbc.Col("Models")],style={"marginLeft": "1.5rem"}),
            html.Div(dbc.Row([
                    dbc.Col(dcc.Dropdown(id="exp_dropdown",options=expn_uniq, 
                                         value=exps,multi=True)),
                    dbc.Col(dcc.Dropdown(id="mod_dropdown",options=modn_uniq, 
                                         value=mods,multi=True)),
                    ]),style={"marginLeft": "1.5rem",
                              "marginRight": "1.5rem"})]
    
    elif tab == 'tab-2':
        try: 
            exps_chosen = [exp for exp in exps_dict["0"].values()]
        except KeyError:
            exps_chosen = []
        try:
            mods_chosen = [exp for exp in mods_dict["0"].values()]
        except KeyError:
            mods_chosen = []
        query = f"Experiment_name == {exps_chosen} & "
        query += f"Model_name == {mods_chosen}"
        filtered_res = results.query(query)  # type: ignore        

        # Select which HP will be available to be filtered from all the HP
        # select all columns where values are not all NAN 
        all_columns = [col for col in filtered_res.columns if not filtered_res[col].isnull().all()]
        # HP are by definition the columns before train_loss
        if 'train_loss' in all_columns:
            hp_names = all_columns[0:all_columns.index('train_loss')]
        else:
            return html.Div([html.Br(),
                             "No experiments or models have been selected."])
        centered = {"display": "flex","justifyContent": "center"}
        names = [html.Div(name,style=centered) for name in hp_names]

        # Create dropdown menus to filter for different HP values
        hp_drop = []
        for hp in hp_names:
            # Possible values for a HP
            values = filtered_res[hp].unique().tolist()
            # If the HP has a NAN value anywhere
            if filtered_res[hp].isnull().any():
                # replace nan with Not specified 
                values_wo_nan = [val for val in values if not np.isnan(val)]
                values = values_wo_nan[:]+["Not specified"]

            hp_drop.append(dcc.Dropdown(id={"type": "hp-dropdown", "index": hp},
                                        options=values,
                                        value=values,
                                        multi=True))
        output = [val for pair in zip(names,hp_drop) for val in pair]
        output.insert(0,html.Br())
        output.insert(1,html.Div("All filters will be reset "
                                "when switching between tabs."))        
        output.insert(2,html.Br())
        output.insert(3,dbc.InputGroup([
            dbc.InputGroupText("Select maximum train accuracy > "),
            dbc.Input(id={"type": "max_acc", "index": 0},
                      type="number",
                      min=0,max=1)
            ]))

        return html.Div(children=output,
                        style={"marginLeft": "1.5rem","marginRight": "1.5rem"})

#Stores chosen experiments
@callback(Output("selected_exps","data"),
          Input("exp_dropdown","value"))
def store_exps(exps_chosen):
    #print("STORE EXPS")
    if isinstance(exps_chosen, str):
        exps_chosen = [exps_chosen]
    df = pd.DataFrame(exps_chosen)
    return df.to_dict()

#Stores chosen models
@callback(Output("selected_mods","data"),
          Input("mod_dropdown","value"))
def store_mods(mods_chosen):
    #print("STORE MODS")
    if isinstance(mods_chosen, str):
        mods_chosen = [mods_chosen]
    df = pd.DataFrame(mods_chosen)
    return df.to_dict()

# Updates available models based on selected experiments
@callback(Output("mod_dropdown","options"),
          Output("mod_dropdown","value"),
            Input("exp_dropdown", "value"))
def update_models(exps_chosen):
    #print("UPDATE MODELS")
    # If only one experiment is selected
    if isinstance(exps_chosen, str):
        exps_chosen = [exps_chosen]
    filtered_exps = results.query(f"Experiment_name == {exps_chosen}")          # type:ignore
    model_choices=[mod for mod in filtered_exps["Model_name"].unique().tolist()]
    return model_choices,model_choices

# Updates available hyperparameters based on selected max acc
@callback(Output({"type": "hp-dropdown", "index": ALL},"value"),
          Output({"type": "hp-dropdown", "index": ALL},"options"),
          State("selected_exps","data"),
          State("selected_mods","data"),
          Input({"type":"max_acc", "index": ALL},"value"),
          prevent_initial_call=True)
def update_hp(exps_dict,mods_dict,thrs):
    #print("UPDATE HP")
    try: 
        exps_chosen = [exp for exp in exps_dict["0"].values()]
    except KeyError:
        exps_chosen = []
    try:
        mods_chosen = [exp for exp in mods_dict["0"].values()]
    except KeyError:
        mods_chosen = []
    if thrs == []:
        thrs = None
    else:
        thrs = thrs[0]
    query = f"Experiment_name == {exps_chosen} & "
    query += f"Model_name == {mods_chosen}"
    filtered_res = results.query(query)  # type: ignore 

    # Filter by maximum train accuracy higher than an input value
    
    if thrs == None or thrs == 0:
        pass
    else:
        # Select all IDs that pass the acc threshold, that way we include the
        # test accuracy and loss even if they are below the threshold
        id=filtered_res["ID"][filtered_res["train_acc"]>=thrs].unique().tolist()
        filtered_res = filtered_res.query(f"ID == {id}")

    # Select which HP will be available to be filtered from all the HP
    # select all columns where values are not all NAN 
    all_columns = [col for col in filtered_res.columns if not filtered_res[col].isnull().all()]
    # HP are by definition the columns before train_loss
    if 'train_loss' in all_columns:
        hp_names = all_columns[0:all_columns.index('train_loss')]
    else:
        raise PreventUpdate
    
    # Find all the new different HP values
    all_drop_values = []
    for hp in hp_names:
        # Possible values for a HP
        values = filtered_res[hp].unique().tolist()
        # If the HP has a NAN value anywhere
        if filtered_res[hp].isnull().any():
            # replace nan with Not specified 
            values_wo_nan = [val for val in values if not np.isnan(val)]
            values = values_wo_nan[:]+["Not specified"]
        all_drop_values.append(values)

    return all_drop_values,all_drop_values


# Plots accuracy
@callback(Output(component_id="acc_plot",component_property="figure"),
          Input(component_id="acc_legend_toggle",component_property="on"),
          Input("selected_exps", "data"),
          Input("selected_mods", "data"),
          Input({"type": "hp-dropdown", "index": ALL},"value"),
          Input({"type": "hp-dropdown", "index": ALL},"id"),
          Input({"type":"max_acc", "index": ALL},"value"),
            )
def plot_acc(acc_legend,exps_chosen,mods_chosen,hp_values,hp_ids,thrs):
    #print("PLOT ACC")
    color_palette = pc.qualitative.Dark24
    colors = iter(color_palette)

    try: 
        exps = [exp for exp in exps_chosen["0"].values()]
    except KeyError:
        exps = []
    try:
        mods = [mod for mod in mods_chosen["0"].values()]
    except KeyError:
        mods = []
    if thrs == []:
        thrs = None
    else:
        thrs = thrs[0]
    # Filter by selected experiments, models and HP values
    query = f"Experiment_name == {exps} & "
    query += f"Model_name == {mods}"
    filt_res = results.query(query) # type: ignore

    # Filter by accuracy threshold
    if thrs == None or thrs == 0:
        add_thrs_line = False
        filtered_res = filt_res
    else:
        # Select all IDs and Iter # that pass the acc threshold, that way we 
        # include the test accuracy and loss even if they are below the 
        # threshold (If we dont group by Iter # we add all iters instead of 
        # those above the threshold)
        gb = filt_res.groupby(["ID","Iter #"]).max("train_acc") #type:ignore
        idx_list = gb[gb["train_acc"]>=thrs].index
        filtered_res = pd.DataFrame()
        for idx in idx_list:
            id = idx[0]
            it = idx[1]
            df = filt_res[(filt_res["ID"]==id) & (filt_res["Iter #"]==it)]      #type:ignore
            filtered_res = pd.concat([filtered_res,df],ignore_index=True)
        add_thrs_line = True

    # Filter by hyperparameters
    for val_list, id_dict in zip(hp_values,hp_ids):
        # Not specified corresponds to nan in HP value, so check for that 
        # alongside the other specified values
        if "Not specified" in val_list:
            query = f"{id_dict["index"]}.isnull() | {id_dict["index"]} == {val_list} "
            filtered_res = filtered_res.query(query)
            continue
        query = f"{id_dict["index"]} == {val_list} "
        filtered_res = filtered_res.query(query)


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
        x = re.sub("__", "_", x)

        template=f"<b>Experiment: </b>{expn}<br><b>Model: </b>{modn}<br>"
        legtitle = "Experiment <b> Model </b> Hyperparameters Iteration<br>"
        color = next(colors)

        # Check if an experiment has been repeated for multiple 
        # iterations. All iterations are plotted the same color

        try: # Check if a column named Iter # exists
            num_iter = df_id["Iter #"].max()
            iter_flag = True
            # The column may exist in the results dataframe but the value 
            # may be NaN.
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
                hovertemp = template + f"<b>HP: </b>{extra}<extra>Iter {it+1}<br>" 
                df = df_id[df_id["Iter #"] == it+1]
            else:
                name = f"{expn} <b>{modn}</b> {extra}"
                hovertemp = template + f"<b>HP: </b> {extra} <br><extra>"
                df = df_id

            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["train_acc"],
                                    name=name, 
                                    line={"color": color}, 
                                    hovertemplate=hovertemp+"%{y:.3f}</extra>"))
            fig.add_trace(go.Scatter(x=df["Epoch #"], y=df["test_acc"],
                                    name="", 
                                    line={"color": color, "dash": "dash"}, 
                                    hovertemplate=hovertemp+"%{y:.3f}</extra>"))
        if add_thrs_line:
            fig.add_hline(y=thrs)
        fig.update_layout(xaxis_title="Epochs", yaxis_title="Accuracy",
                            showlegend=acc_legend,
                            legend_title={"text": legtitle},
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
          Input("selected_exps", "data"),
          Input("selected_mods", "data"),
          Input({"type": "hp-dropdown", "index": ALL},"value"),
          Input({"type": "hp-dropdown", "index": ALL},"id"),
          Input({"type":"max_acc", "index": ALL},"value"),
          )
def plot_loss(loss_legend,exps_chosen,mods_chosen,hp_values,hp_ids,thrs):
    #print("PLOT LOSS")
    color_palette = pc.qualitative.Dark24
    colors = iter(color_palette)

    try: 
        exps = [exp for exp in exps_chosen["0"].values()]
    except KeyError:
        exps = []
    try:
        mods = [mod for mod in mods_chosen["0"].values()]
    except KeyError:
        mods = []
    if thrs == []:
        thrs = None
    else:
        thrs = thrs[0]
    query = f"Experiment_name == {exps} & "
    query += f"Model_name == {mods}"
    filt_res = results.query(query) # type: ignore

    # Filter by accuracy threshold
    if thrs == None or thrs == 0:
        add_thrs_line = False
        filtered_res = filt_res
    else:
        # Select all IDs and Iter # that pass the acc threshold, that way we 
        # include the test accuracy and loss even if they are below the 
        # threshold (If we dont group by Iter # we add all iters instead of 
        # those above the threshold)
        gb = filt_res.groupby(["ID","Iter #"]).max("train_acc") #type:ignore
        idx_list = gb[gb["train_acc"]>=thrs].index
        filtered_res = pd.DataFrame()
        for idx in idx_list:
            id = idx[0]
            it = idx[1]
            df = filt_res[(filt_res["ID"]==id) & (filt_res["Iter #"]==it)]      #type:ignore
            filtered_res = pd.concat([filtered_res,df],ignore_index=True)
        #id=filtered_res["ID"][filtered_res["train_acc"]>=thrs].unique().tolist()
        #filtered_res = filtered_res.query(f"ID == {id}")
        add_thrs_line = True
    for val_list, id_dict in zip(hp_values,hp_ids):
        if "Not specified" in val_list:
            query = f"{id_dict["index"]}.isnull() | {id_dict["index"]} == {val_list} "
            filtered_res = filtered_res.query(query)
            continue
        query = f"{id_dict["index"]} == {val_list} "
        filtered_res = filtered_res.query(query)
    fig = go.Figure()

    for id in filtered_res["ID"].unique():
        df_id = filtered_res[filtered_res["ID"] == id]
        expn = df_id["Experiment_name"].unique()[0]
        modn = df_id["Model_name"].unique()[0]
        extra = df_id["Extra"].unique()[0]

        x = re.sub("Iter_[0-9]+", "", extra)
        x = re.sub("_$", "", x)
        x = re.sub("^_", "", x)
        x = re.sub("__", "_", x)

        template=f"<b>Experiment: </b>{expn}<br><b>Model: </b>{modn}<br>"
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
                hovertemp = template + f"<b>HP: </b> {extra} <extra> Iter {it+1} </extra>"
                df = df_id[df_id["Iter #"] == it+1]
            else:
                name = f"{expn} <b>{modn}</b> {extra}"
                hovertemp = template+f"<b>HP: </b> {extra} <br><extra></extra>"
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

    return fig


if __name__ == '__main__':
    app.run(debug=True)
