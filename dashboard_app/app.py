from shinywidgets import output_widget, render_widget
from shiny import App, ui, reactive

import plotly.graph_objects as go
import plotly.colors as pc

import numpy as np
import base64
import sys
import re
import os

MAIN_DIRECTORY = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if MAIN_DIRECTORY not in sys.path:
    sys.path.insert(0, str(MAIN_DIRECTORY))

from utils.helpers import retrieve_results

img_path = os.path.join(MAIN_DIRECTORY,"dashboard_app", "train_test_legend.png")
with open(img_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()
img_source = "data:image/png;base64," + encoded

results = retrieve_results()  # type: ignore
expn_uniq = results["Experiment_name"].unique() # type:ignore
modn_uniq = results["Model_name"].unique() # type:ignore


# Creating the whole UI of the dashboard, inside a single ui.page_sidebar
app_ui = ui.page_sidebar(
    
    # Adding a sidebar
    ui.sidebar(
        #Adding an accordeon to the sidebar
        ui.accordion(
            ui.accordion_panel(
                "Select Experiments and Models",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Experiments"),
                        ui.input_checkbox_group(
                            id="exp_ch",
                            label="",
                            choices={exp: exp for exp in expn_uniq},  
                            selected=results["Experiment_name"].unique()[0],    # type:ignore
                        )
                    ),
                    ui.card(
                        ui.card_header("Models"),
                        ui.input_checkbox_group(
                            "mod_ch",
                            label="",
                            choices={mod: mod for mod in modn_uniq},  
                            selected=results["Model_name"].unique()[0],         # type:ignore
                        )
                    ),
                    col_widths=(10, 10)
                )
            ),
            ui.accordion_panel(
                "Filter Hyperparameters",
                "Add a table head that lets you filter hyperparameters by "
                "values. Although with the size of the accordion better do it "
                "in a third card on the main page. Work in progress."
            ),
            id="acc", open=None
        )
    ),

    # Adding two cards to display the plots
    ui.layout_columns(
        # Card for plotting accuracy
        ui.card(
            ui.card_header("Training and testing accuracy"),
            # Ading a switch to toggle the legend
            ui.div(
                {"style": "position: absolute; top: 0.5rem; right: 0.5rem;"},
                ui.input_switch("legend_acc", "Legend", False),
            ),
            output_widget(id="plot_acc")
        ),
        # Card for plotting loss
        ui.card(
            ui.card_header("Training and testing loss"),
            # Ading a switch to toggle the legend
            ui.div(
                {"style": "position: absolute; top: 0.5rem; right: 0.5rem;"},
                ui.input_switch("legend_loss", "Legend", False),
            ),
            output_widget(id="plot_loss")
        ),
        col_widths=(12, 12)
    )
,title="Experiment Tracking") # Title of the whole dashboard


# All reactive effects and the renders of the plots go inside server
def server(input, output, session):

    @reactive.effect
    def update_models():
        filtered_models = results.query(f"Experiment_name == {input.exp_ch()}")  # type:ignore
        choices = {exp: exp for exp in filtered_models["Model_name"].unique()}
        ui.update_checkbox_group(
            "mod_ch",choices=choices,selected=input.mod_ch())

    @reactive.effect
    def toggle_legend_acc():
        # putting input.legend_acc() (the switch's state) in here re-renders the 
        # plots since they have the input.legend_acc() as a variable inside
        input.legend_acc() 
        
    @reactive.effect
    def toggle_legend_loss():
        input.legend_loss()

    # Render the plots
    @output
    @render_widget
    def plot_acc():
        color_palette = pc.qualitative.Dark24
        colors = iter(color_palette)
        # Filter the results based on the selected experiments and models
        query = f"Experiment_name == {input.exp_ch()} & "
        query += f"Model_name == {input.mod_ch()}"
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

            fig.update_layout(xaxis_title="Epochs", yaxis_title="Accuracy",
                              showlegend=input.legend_acc(), 
                              legend_title={"text": legtitle})

        fig.add_layout_image(dict(
            source=img_source,
            xref="paper", yref="paper",
            x=0.01, y=1.0,
            sizex=1, sizey=0.1,
            xanchor="left", yanchor="bottom",
            sizing="contain", opacity=1.0,
            layer="above"
        ))

        
        return go.Figure(fig)

    @output
    @render_widget
    def plot_loss():
        color_palette = pc.qualitative.Dark24
        colors = iter(color_palette)
        query = f"Experiment_name == {input.exp_ch()} & "
        query += f"Model_name == {input.mod_ch()}"
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
                              showlegend=input.legend_loss(), 
                              legend_title={"text": legtitle})

        fig.add_layout_image(dict(
            source=img_source,
            xref="paper", yref="paper",
            x=0.01, y=1.0,
            sizex=1, sizey=0.1,
            xanchor="left", yanchor="bottom",
            sizing="contain", opacity=1.0,
            layer="above"
        ))

        
        return go.Figure(fig)


app = App(app_ui, server)
