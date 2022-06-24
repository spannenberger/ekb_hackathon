# import dash
# import dash_core_components as dcc
# import dash_reusable_components as drc
# import dash_html_components as html
# import pandas as pd
# import numpy as np

# from dash.dependencies import Input, Output, State
# from plotly import graph_objs as go
# from plotly.graph_objs import *
# from datetime import datetime as dt
# from utils import STORAGE_PLACEHOLDER, GRAPH_PLACEHOLDER, \
#     IMAGE_STRING_PLACEHOLDER
# from utils import apply_filters, show_histogram, generate_lasso_mask, \
#     apply_enhancements

# import time
# import json
# DEBUG = True

# app = dash.Dash(
#     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
# )
# app.title = "New York Uber Rides"
# server = app.server


# # Plotly mapbox public token
# mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

# # Dictionary of important locations in New York

# list_of_diseases = {
#     "First_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Second_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Third_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Fourth_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Fifth_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Sixth_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Seventh_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Eight_disease": {"lat": 40.7505, "lon": -73.9934},
#     "Nineth_disease": {"lat": 40.7505, "lon": -73.9934},
# }
# # Layout of Dash App
# app.layout = html.Div(
#     children=[
#         html.Div(
#             className="row",
#             children=[
#                 # Column for user controls
#                 html.Div(
#                     className="four columns div-user-controls",
#                     children=[
#                         html.A(
#                             html.Img(
#                                 className="logo",
#                                 src=app.get_asset_url("dash-logo-new.png"),
#                             ),
#                             href="https://plotly.com/dash/",
#                         ),
#                         html.H2("hyper popis [Napoleon IT] - Teeth Analizes"),
#                         html.P(
#                             """Загрузите фотоснимок и получите результат работы алгоритма. А указав параметры ниже вы сможете сохранить результаты в БД"""
#                         ),
#                         drc.Card([
#                             dcc.Upload(
#                                 id='upload-image',
#                                 children=[
#                                     'Drag and Drop or ',
#                                     html.A('Select an Image')
#                                 ],
#                                 style={
#                                     'width': '100%',
#                                     'height': '50px',
#                                     'lineHeight': '50px',
#                                     'borderWidth': '1px',
#                                     'borderStyle': 'dashed',
#                                     'borderRadius': '5px',
#                                     'textAlign': 'center'
#                                 },
#                                 accept='image/*'
#                             ),

#                             drc.NamedInlineRadioItems(
#                                 name='Selection Mode',
#                                 short='selection-mode',
#                                 options=[
#                                     {'label': ' Rectangular', 'value': 'select'},
#                                     {'label': ' Lasso', 'value': 'lasso'}
#                                 ],
#                                 val='select'
#                             ),

#                             drc.NamedInlineRadioItems(
#                                 name='Image Display Format',
#                                 short='encoding-format',
#                                 options=[
#                                     {'label': ' JPEG', 'value': 'jpeg'},
#                                     {'label': ' PNG', 'value': 'png'}
#                                 ],
#                                 val='jpeg'
#                             ),
#                             # dcc.Graph(id='graph-histogram-colors',
#                             #   config={'displayModeBar': False})
#                         ]),
#                         # html.Div(
#                         #     className="div-for-dropdown",
#                         #     children=[
#                         #         dcc.DatePickerSingle(
#                         #             id="date-picker",
#                         #             min_date_allowed=dt(2014, 4, 1),
#                         #             max_date_allowed=dt(2014, 9, 30),
#                         #             initial_visible_month=dt(2014, 4, 1),
#                         #             date=dt(2014, 4, 1).date(),
#                         #             display_format="MMMM D, YYYY",
#                         #             style={"border": "0px solid black"},
#                         #         )
#                         #     ],
#                         # ),
#                         # Change to side-by-side for mobile layout
#                         html.Div(
#                             className="row",
#                             children=[
#                                 html.Div(
#                                     className="div-for-dropdown",
#                                     children=[
#                                         # Dropdown for locations on map
#                                         dcc.Dropdown(
#                                             id="location-dropdown",
#                                             options=[
#                                                 {"label": i, "value": i}
#                                                 for i in list_of_diseases
#                                             ],
#                                             placeholder="Select Disease",
#                                         )
#                                     ],
#                                 ),
#                                 html.Div(
#                                     className="div-for-dropdown",
#                                     children=[
#                                         # Dropdown to select times
#                                         dcc.Dropdown(
#                                             id="bar-selector",
#                                             options=[
#                                                 {
#                                                     "label": str(n) + ":00",
#                                                     "value": str(n),
#                                                 }
#                                                 for n in range(24)
#                                             ],
#                                             multi=True,
#                                             placeholder="Select certain hours",
#                                         )
#                                     ],
#                                 ),
#                             ],
#                         ),
#                         html.P(id="total-rides"),
#                         html.P(id="total-rides-selection"),
#                         html.P(id="date-value"),
#                         dcc.Markdown(
#                             """
#                             Source: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)

#                             Links: [Source Code](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-uber-rides-demo) | [Enterprise Demo](https://plotly.com/get-demo/)
#                             """
#                         ),
#                     ],
#                 ),
#                 # Column for app graphs and plots
#                html.Div(
#                     className='seven columns',
#                     style={'float': 'right'},
#                     children=[
#                         # The Interactive Image Div contains the dcc Graph
#                         # showing the image, as well as the hidden div storing
#                         # the true image
#                         html.Div(id='div-interactive-image', children=[
#                             GRAPH_PLACEHOLDER,
#                             html.Div(
#                                 id='div-storage',
#                                 children=STORAGE_PLACEHOLDER,
#                                 style={'display': 'none'}
#                             )
#                         ])
#                     ]
#                 )
#             ],
#         )
#     ]
# )

# # Gets the amount of days in the specified month
# # Index represents month (0 is April, 1 is May, ... etc.)
# daysInMonth = [30, 31, 30, 31, 31, 30]

# # Get index for the specified month in the dataframe
# monthIndex = pd.Index(["Apr", "May", "June", "July", "Aug", "Sept"])
# @app.callback(Output('div-interactive-image', 'children'),
#               [Input('upload-image', 'contents'),
#                Input('button-undo', 'n_clicks'),
#                Input('button-run-operation', 'n_clicks')],
#               [State('interactive-image', 'selectedData'),
#                State('dropdown-filters', 'value'),
#                State('dropdown-enhance', 'value'),
#                State('slider-enhancement-factor', 'value'),
#                State('upload-image', 'filename'),
#                State('radio-selection-mode', 'value'),
#                State('radio-encoding-format', 'value'),
#                State('div-storage', 'children'),
#                State('session-id', 'children')])
# def update_graph_interactive_image(content,
#                                    undo_clicks,
#                                    n_clicks,
#                                    selectedData,
#                                    filters,
#                                    enhance,
#                                    enhancement_factor,
#                                    new_filename,
#                                    dragmode,
#                                    enc_format,
#                                    storage,
#                                    session_id):

#     return [
#         drc.InteractiveImagePIL(
#             image_id='interactive-image',
#             image=im_pil,
#             enc_format=enc_format,
#             display_mode='fixed',
#             dragmode=dragmode,
#             verbose=DEBUG
#         ),

#         html.Div(
#             id='div-storage',
#             children=json.dumps(storage),
#             style={'display': 'none'}
#         )
#     ]

# # Get the amount of rides per hour based on the time selected
# # This also higlights the color of the histogram bars based on
# # if the hours are selected
# # def get_selection(month, day, selection):
# #     xVal = []
# #     yVal = []
# #     xSelected = []
# #     colorVal = [
# #         "#F4EC15",
# #         "#DAF017",
# #         "#BBEC19",
# #         "#9DE81B",
# #         "#80E41D",
# #         "#66E01F",
# #         "#4CDC20",
# #         "#34D822",
# #         "#24D249",
# #         "#25D042",
# #         "#26CC58",
# #         "#28C86D",
# #         "#29C481",
# #         "#2AC093",
# #         "#2BBCA4",
# #         "#2BB5B8",
# #         "#2C99B4",
# #         "#2D7EB0",
# #         "#2D65AC",
# #         "#2E4EA4",
# #         "#2E38A4",
# #         "#3B2FA0",
# #         "#4E2F9C",
# #         "#603099",
# #     ]

# #     # Put selected times into a list of numbers xSelected
# #     # xSelected.extend([int(x) for x in selection])

# #     # for i in range(24):
# #     #     # If bar is selected then color it white
# #     #     if i in xSelected and len(xSelected) < 24:
# #     #         colorVal[i] = "#FFFFFF"
# #     #     xVal.append(i)
# #     #     # Get the number of rides at a particular time
# #     #     yVal.append(len(totalList[month][day][totalList[month][day].index.hour == i]))
# #     # return [np.array(xVal), np.array(yVal), np.array(colorVal)]


# # Selected Data in the Histogram updates the Values in the Hours selection dropdown menu
# # @app.callback(
# #     Output("bar-selector", "value"),
# #     [Input("histogram", "selectedData"), Input("histogram", "clickData")],
# # )
# # def update_bar_selector(value, clickData):
# #     holder = []
# #     if clickData:
# #         holder.append(str(int(clickData["points"][0]["x"])))
# #     if value:
# #         for x in value["points"]:
# #             holder.append(str(int(x["x"])))
# #     return list(set(holder))


# # Clear Selected Data if Click Data is used
# # @app.callback(Output("histogram", "selectedData"), [Input("histogram", "clickData")])
# # def update_selected_data(clickData):
# #     if clickData:
# #         return {"points": []}


# # Update the total number of rides Tag
# # @app.callback(Output("total-rides", "children"), [Input("date-picker", "date")])
# # def update_total_rides(datePicked):
# #     date_picked = dt.strptime(datePicked, "%Y-%m-%d")
# #     return "Total Number of rides: {:,d}".format(
# #         len(totalList[date_picked.month - 4][date_picked.day - 1])
# #     )


# # Update the total number of rides in selected times
# # @app.callback(
# #     [Output("total-rides-selection", "children"), Output("date-value", "children")],
# #     [Input("date-picker", "date"), Input("bar-selector", "value")],
# # )
# # def update_total_rides_selection(datePicked, selection):
# #     firstOutput = ""

# #     if selection is not None or len(selection) is not 0:
# #         date_picked = dt.strptime(datePicked, "%Y-%m-%d")
# #         totalInSelection = 0
# #         for x in selection:
# #             totalInSelection += len(
# #                 totalList[date_picked.month - 4][date_picked.day - 1][
# #                     totalList[date_picked.month - 4][date_picked.day - 1].index.hour
# #                     == int(x)
# #                 ]
# #             )
# #         firstOutput = "Total rides in selection: {:,d}".format(totalInSelection)

# #     if (
# #         datePicked is None
# #         or selection is None
# #         or len(selection) is 24
# #         or len(selection) is 0
# #     ):
# #         return firstOutput, (datePicked, " - showing hour(s): All")

# #     holder = sorted([int(x) for x in selection])

# #     if holder == list(range(min(holder), max(holder) + 1)):
# #         return (
# #             firstOutput,
# #             (
# #                 datePicked,
# #                 " - showing hour(s): ",
# #                 holder[0],
# #                 "-",
# #                 holder[len(holder) - 1],
# #             ),
# #         )

# #     holder_to_string = ", ".join(str(x) for x in holder)
# #     return firstOutput, (datePicked, " - showing hour(s): ", holder_to_string)


# # Update Histogram Figure based on Month, Day and Times Chosen
# # @app.callback(
# #     Output("histogram", "figure"),
# #     [Input("date-picker", "date"), Input("bar-selector", "value")],
# # )
# # def update_histogram(datePicked, selection):
# #     date_picked = dt.strptime(datePicked, "%Y-%m-%d")
# #     monthPicked = date_picked.month - 4
# #     dayPicked = date_picked.day - 1

# #     [xVal, yVal, colorVal] = get_selection(monthPicked, dayPicked, selection)

# #     layout = go.Layout(
# #         bargap=0.01,
# #         bargroupgap=0,
# #         barmode="group",
# #         margin=go.layout.Margin(l=10, r=0, t=0, b=50),
# #         showlegend=False,
# #         plot_bgcolor="#323130",
# #         paper_bgcolor="#323130",
# #         dragmode="select",
# #         font=dict(color="white"),
# #         xaxis=dict(
# #             range=[-0.5, 23.5],
# #             showgrid=False,
# #             nticks=25,
# #             fixedrange=True,
# #             ticksuffix=":00",
# #         ),
# #         yaxis=dict(
# #             range=[0, max(yVal) + max(yVal) / 4],
# #             showticklabels=False,
# #             showgrid=False,
# #             fixedrange=True,
# #             rangemode="nonnegative",
# #             zeroline=False,
# #         ),
# #         annotations=[
# #             dict(
# #                 x=xi,
# #                 y=yi,
# #                 text=str(yi),
# #                 xanchor="center",
# #                 yanchor="bottom",
# #                 showarrow=False,
# #                 font=dict(color="white"),
# #             )
# #             for xi, yi in zip(xVal, yVal)
# #         ],
# #     )

# #     return go.Figure(
# #         data=[
# #             go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
# #             go.Scatter(
# #                 opacity=0,
# #                 x=xVal,
# #                 y=yVal / 2,
# #                 hoverinfo="none",
# #                 mode="markers",
# #                 marker=dict(color="rgb(66, 134, 244, 0)", symbol="square", size=40),
# #                 visible=True,
# #             ),
# #         ],
# #         layout=layout,
# #     )


# # Get the Coordinates of the chosen months, dates and times
# # def getLatLonColor(selectedData, month, day):
# #     listCoords = totalList[month][day]

# #     # No times selected, output all times for chosen month and date
# #     if selectedData is None or len(selectedData) is 0:
# #         return listCoords
# #     listStr = "listCoords["
# #     for time in selectedData:
# #         if selectedData.index(time) is not len(selectedData) - 1:
# #             listStr += "(totalList[month][day].index.hour==" + str(int(time)) + ") | "
# #         else:
# #             listStr += "(totalList[month][day].index.hour==" + str(int(time)) + ")]"
# #     return eval(listStr)

# # @app.callback(Output('graph-histogram-colors', 'figure'),
# #               [Input('interactive-image', 'figure')])
# # def update_histogram(figure):
# #     # Retrieve the image stored inside the figure
# #     enc_str = figure['layout']['images'][0]['source'].split(';base64,')[-1]
# #     # Creates the PIL Image object from the b64 png encoding
# #     im_pil = drc.b64_to_pil(string=enc_str)

# #     return show_histogram(im_pil)
# # Update Map Graph based on date-picker, selected data on histogram and location dropdown
# # @app.callback(
# #     Output("map-graph", "figure"),
# #     [
# #         Input("date-picker", "date"),
# #         Input("bar-selector", "value"),
# #         Input("location-dropdown", "value"),
# #     ],
# # )
# # def update_graph(datePicked, selectedData, selectedLocation):
# #     zoom = 12.0
# #     latInitial = 40.7272
# #     lonInitial = -73.991251
# #     bearing = 0

# #     if selectedLocation:
# #         zoom = 15.0
# #         latInitial = list_of_locations[selectedLocation]["lat"]
# #         lonInitial = list_of_locations[selectedLocation]["lon"]

# #     date_picked = dt.strptime(datePicked, "%Y-%m-%d")
# #     monthPicked = date_picked.month - 4
# #     dayPicked = date_picked.day - 1
# #     listCoords = getLatLonColor(selectedData, monthPicked, dayPicked)

# #     return go.Figure(
# #         data=[
# #             # Data for all rides based on date and time
# #             Scattermapbox(
# #                 lat=listCoords["Lat"],
# #                 lon=listCoords["Lon"],
# #                 mode="markers",
# #                 hoverinfo="lat+lon+text",
# #                 text=listCoords.index.hour,
# #                 marker=dict(
# #                     showscale=True,
# #                     color=np.append(np.insert(listCoords.index.hour, 0, 0), 23),
# #                     opacity=0.5,
# #                     size=5,
# #                     colorscale=[
# #                         [0, "#F4EC15"],
# #                         [0.04167, "#DAF017"],
# #                         [0.0833, "#BBEC19"],
# #                         [0.125, "#9DE81B"],
# #                         [0.1667, "#80E41D"],
# #                         [0.2083, "#66E01F"],
# #                         [0.25, "#4CDC20"],
# #                         [0.292, "#34D822"],
# #                         [0.333, "#24D249"],
# #                         [0.375, "#25D042"],
# #                         [0.4167, "#26CC58"],
# #                         [0.4583, "#28C86D"],
# #                         [0.50, "#29C481"],
# #                         [0.54167, "#2AC093"],
# #                         [0.5833, "#2BBCA4"],
# #                         [1.0, "#613099"],
# #                     ],
# #                     colorbar=dict(
# #                         title="Time of<br>Day",
# #                         x=0.93,
# #                         xpad=0,
# #                         nticks=24,
# #                         tickfont=dict(color="#d8d8d8"),
# #                         titlefont=dict(color="#d8d8d8"),
# #                         thicknessmode="pixels",
# #                     ),
# #                 ),
# #             ),
# #             # Plot of important locations on the map
# #             Scattermapbox(
# #                 lat=[list_of_locations[i]["lat"] for i in list_of_locations],
# #                 lon=[list_of_locations[i]["lon"] for i in list_of_locations],
# #                 mode="markers",
# #                 hoverinfo="text",
# #                 text=[i for i in list_of_locations],
# #                 marker=dict(size=8, color="#ffa0a0"),
# #             ),
# #         ],
# #         layout=Layout(
# #             autosize=True,
# #             margin=go.layout.Margin(l=0, r=35, t=0, b=0),
# #             showlegend=False,
# #             mapbox=dict(
# #                 accesstoken=mapbox_access_token,
# #                 center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
# #                 style="dark",
# #                 bearing=bearing,
# #                 zoom=zoom,
# #             ),
# #             updatemenus=[
# #                 dict(
# #                     buttons=(
# #                         [
# #                             dict(
# #                                 args=[
# #                                     {
# #                                         "mapbox.zoom": 12,
# #                                         "mapbox.center.lon": "-73.991251",
# #                                         "mapbox.center.lat": "40.7272",
# #                                         "mapbox.bearing": 0,
# #                                         "mapbox.style": "dark",
# #                                     }
# #                                 ],
# #                                 label="Reset Zoom",
# #                                 method="relayout",
# #                             )
# #                         ]
# #                     ),
# #                     direction="left",
# #                     pad={"r": 0, "t": 0, "b": 0, "l": 0},
# #                     showactive=False,
# #                     type="buttons",
# #                     x=0.45,
# #                     y=0.02,
# #                     xanchor="left",
# #                     yanchor="bottom",
# #                     bgcolor="#323130",
# #                     borderwidth=1,
# #                     bordercolor="#6d6d6d",
# #                     font=dict(color="#FFFFFF"),
# #                 )
# #             ],
# #         ),
# #     )


# if __name__ == "__main__":
#     app.run_server(debug=True)