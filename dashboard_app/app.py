from shiny.express import input, render, ui
from shinywidgets import render_plotly
from shiny import reactive

import plotly.graph_objects as go
import plotly.colors as pc

import numpy as np
import itertools
import sys
import re
import os

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if MAIN_DIRECTORY not in sys.path:
    sys.path.insert(0,str(MAIN_DIRECTORY))

from utils.helpers import retrieve_results

import base64

img_path = os.path.join(MAIN_DIRECTORY,"dashboard_app","train_test_legend.png")

with open(img_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()
img_source = "data:image/png;base64," + encoded

results = retrieve_results() #type:ignore

ui.page_opts(title="Experiment Tracking")

# Adding a sidebar
with ui.sidebar():
    #Adding an accordeon to the sidebar
    with ui.accordion(id="acc", open=None):  
        with ui.accordion_panel("Select Experiments and Models"):
            with ui.layout_columns(col_widths=(10,10)):
                with ui.card():
                    ui.card_header("Experiments")
                    choices={}
                    selected=[]
                    for exp in results["Experiment_name"].unique():#type:ignore
                        choices.update({f"{exp}":f"{exp}"}) 
                        # key and value are the same since input.exp_ch returns
                        # the keys that are selected
                        selected.append(f"{exp}")
                    ui.input_checkbox_group(id="exp_ch",  
                        label="",
                        choices=choices,
                        selected=selected[0],
                    )
                with ui.card():  
                    ui.card_header("Models")
                    choices={}
                    selected=[]
                    for mod in results["Model_name"].unique():#type:ignore
                        choices.update({f"{mod}":f"{mod}"})
                        selected.append(f"{mod}")
                    ui.input_checkbox_group(id="mod_ch",  
                        label="",
                        choices=choices,
                        selected=selected[0],
                    )
        with ui.accordion_panel("Filter Hyperparameters"):  
            "Add a table head that lets you filter hyperparameters by values. "
            "Although with the size of the accordeon better do it in a third "
            "card on the main page. "
            "Work in progress."



# Change the available models based on the selected experiments
@reactive.effect
def update_models():
    #print("UPDATING MODEL CHECKBOXES")
    filtered_models = results.query(#type:ignore
        f"Experiment_name == {input.exp_ch()}")

    choices={}
    for exp in filtered_models["Model_name"].unique():
        choices.update({f"{exp}":f"{exp}"})

    ui.update_checkbox_group(id="mod_ch",
                             choices=choices,
                             selected=input.mod_ch())

# Toggles the legend for the accuracy plot
@reactive.effect
def toggle_legend_acc():
    #print("TOGGLING ACCURACY LEGEND")
    input.legend_acc()

# Toggles the legend for the loss
@reactive.effect
def toggle_legend_loss():
    #print("TOGGLING LOSS LEGEND")
    input.legend_loss()



# Adding two cards to display the plots
# The colummns layout is divided in 12 spaces, so 12 spaces per column make 1 
# card per row  
with ui.layout_columns(col_widths=(12,12)):
    # Card for plotting accuracy
    with ui.card():  
        ui.card_header("Training and testing accuracy")
        # Ading a switch to toggle the legend
        with ui.div(style="position: absolute; top: 0.5rem; right: 0.5rem;"):
            ui.input_switch("legend_acc", "Legend", False)
            #ui.HTML("<p align=right> </p>")
        @render_plotly
        def plot_acc():
            #print("RE-RENDER ACC")
            color_palette = pc.qualitative.Dark24
            colors = iter(color_palette)
            query = ""
            query += f"Experiment_name == {input.exp_ch()}"
            query += " & "
            query += f"Model_name == {input.mod_ch()}"
            filtered_res = results.query(query)#type:ignore
            fig = go.Figure()
            for id in filtered_res["ID"].unique():
                df_id = filtered_res[filtered_res["ID"]==id]
                expn = df_id["Experiment_name"].unique()[0]
                modn = df_id["Model_name"].unique()[0]
                extra = df_id["Extra"].unique()[0]
                #remove iter_n from extra
                x = re.sub("Iter_[0-9]+","", extra)
                x = re.sub("_$","", x)
                x = re.sub("^_","", x)
                ht_template = f"<b>Experiment: </b> {expn} <br>"
                ht_template += f"<b>Model: </b> {modn} <br>"
                legtitle = "Experiment <b> Model </b> Hyperparameters Iteration <br>"
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
                        add=f"<b>HP: </b> {extra} <extra> Iter {it+1} </extra>"
                        hovertemp = ht_template + add
                        df = df_id[df_id["Iter #"]==it+1]
                    else:
                        name = f"{expn} <b>{modn}</b> {extra}"
                        add = f"<b>HP: </b> {extra} <br><extra></extra>"
                        hovertemp = ht_template + add
                        df = df_id

                    fig.add_trace(go.Scatter(x=df["Epoch #"],
                                            y=df["train_acc"],
                                            name=name,
                                            line={"color":color},
                                            hovertemplate=hovertemp))
                    fig.add_trace(go.Scatter(x=df["Epoch #"],
                                            y=df["test_acc"],
                                            name="",
                                            line={"color":color,"dash":"dash"},
                                            hovertemplate=hovertemp))
            
                fig.update_layout(xaxis_title="Epochs",
                                yaxis_title="Accuracy",
                                showlegend=input.legend_acc(),
                                legend_title={"text":legtitle})
                

            # Creating a fake legend for train-test
            fig.add_layout_image(
            dict(
            source=img_source, #105x43 pxls
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.0,
            sizex=1,
            sizey=0.1,
            xanchor="left", 
            yanchor="bottom",
            sizing="contain",
            opacity=1.0,
            layer="above")
            )
            return fig

    # Card for plotting loss
    with ui.card():  
        ui.card_header("Training and testing loss")
        # Ading a switch to toggle the legend
        with ui.div(style="position: absolute; top: 0.5rem; right: 0.5rem;"):
            ui.input_switch("legend_loss", "Legend", False)
        @render_plotly
        def plot_loss():
            #print("RE-RENDER LOSS")
            color_palette = pc.qualitative.Dark24
            colors = iter(color_palette)
            query = ""
            query += f"Experiment_name == {input.exp_ch()}"
            query += " & "
            query += f"Model_name == {input.mod_ch()}"
            filtered_res = results.query(query)#type:ignore
            fig = go.Figure()
            for id in filtered_res["ID"].unique():
                df_id = filtered_res[filtered_res["ID"]==id]
                expn = df_id["Experiment_name"].unique()[0]
                modn = df_id["Model_name"].unique()[0]
                extra = df_id["Extra"].unique()[0]
                #remove iter_n from extra
                x = re.sub("Iter_[0-9]+","", extra)
                x = re.sub("_$","", x)
                x = re.sub("^_","", x)
                ht_template = f"<b>Experiment: </b> {expn} <br>"
                ht_template += f"<b>Model: </b> {modn} <br>"
                legtitle = "Experiment <b> Model </b> Hyperparameters Iteration <br>"
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
                        df = df_id[df_id["Iter #"]==it+1]
                    else:
                        name = f"{expn} <b>{modn}</b> {extra}"
                        hovertemp = ht_template + f"<b>HP: </b> {extra} <br><extra></extra>"
                        df = df_id

                    fig.add_trace(go.Scatter(x=df["Epoch #"],
                                            y=df["train_loss"],
                                            name=name,
                                            line={"color":color},
                                            hovertemplate=hovertemp))
                    fig.add_trace(go.Scatter(x=df["Epoch #"],
                                            y=df["test_loss"],
                                            name="",
                                            line={"color":color,"dash":"dash"},
                                            hovertemplate=hovertemp))

                fig.update_layout(xaxis_title="Epochs",
                                yaxis_title="Average loss per batch",
                                showlegend=input.legend_loss(),
                                legend_title={"text":legtitle})            
            # Creating a fake legend for train-test
            fig.add_layout_image(
            dict(
            source=img_source, #105x43 pxls
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.0,
            sizex=1,
            sizey=0.1,
            xanchor="left", 
            yanchor="bottom",
            sizing="contain",
            opacity=1.0,
            layer="above")
            )
            fig.update_layout_images()
            return fig

