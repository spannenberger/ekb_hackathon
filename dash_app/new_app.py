from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import io
import base64
import datetime
import plotly.express as px
from PIL import Image
from typing import Dict
import requests
import cv2
from tqdm import tqdm


DEBUG = True
FRAMERATE = 24.0
URL = 'http://10.10.66.112:5010/api/ekb_service'

content_type = 'image/jpeg'
headers = {'content-type': content_type}
app = dash.Dash(__name__)
server = app.server

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


def load_data(path):
    """Load data about a specific footage (given by the path). It returns a dictionary of useful variables such as
    the dataframe containing all the detection and bounds localization, the number of classes inside that footage,
    the matrix of all the classes in string, the given class with padding, and the root of the number of classes,
    rounded."""

    # Load the dataframe containing all the processed object detections inside the video
    images_info_df = pd.read_csv(path)

    # The list of classes, and the number of classes
    file_names = images_info_df["file_name"].unique().tolist()
    classes_list = images_info_df["class"].tolist()
    # classes_list = video_info_df["class_str"].value_counts().index.tolist()
    n_classes = len(classes_list)
    bboxes = images_info_df["bbox"].tolist()
    final_bboxes = [eval(bbox) for bbox in bboxes]
    # predicted_classes = images_info_df["class"].tolist()
    # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    root_round = np.ceil(np.sqrt(len(classes_list)))
    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": images_info_df,
        "n_classes": n_classes,
        "classes_matrix": classes_matrix,
        "classes_padded": classes_padded,
        "root_round": root_round,
        # "final_bboxes": final_bboxes
    }

    if DEBUG:
        print(f'{path} loaded.')

    return data_dict

def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей
    Обрабатываем результат работы моделей, извлекая полученный класс животного
    Args:
        image_array: arr - массив-представление изображения
        metadata: json - словарь, содержащий ответ работы моделей
    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """
    # import pdb;pdb.set_trace()
    for bbox in tqdm(metadata["bbox"]):
        class_name = bbox['class_name']

        # confidence = bbox['confidence']

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        # center_coords = int((botRightCorner[0] + topLeftCorner[0]) / 2), int((botRightCorner[1] + topLeftCorner[1]) / 2)
        cv2.rectangle(
            image_array,
            topLeftCorner,
            botRightCorner,
            (255, 0, 0), 
            1
        )
        cv2.putText(image_array, f'{class_name}', 
                            topLeftCorner,
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 0, 0),
                            1,
                            1)

def markdown_popup():
    return html.Div(
        id='markdown',
        className="model",
        style={'display': 'none'},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={'border': 'none', 'height': '100%'}
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=[dcc.Markdown(
                            children=dedent(
                                '''
                                ##### Что делает это приложение? 
                                
                               Данное приложение позволяет проводить анализ заболевания зубов по видео. Также в правой части мы сможем 
                               наблюдать предсказания модели в реальном времени с обозначением классов на текущем кадре и вероятностей

                                ##### Можно добавить еще описания
                                вот тут текст 
                                '''
                            ))
                        ]
                    )
                ]
            )
        )
    )


# Main App

app.layout = html.Div(
    children=[
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        html.Div(
            className='container',
            children=[
        html.Div(
            id='left-side-column',
            className='eight columns',
            style={'display': 'flex',
                   'flexDirection': 'column',
                   'flex': 1,
                   'height': 'calc(100vh - 5px)',
                   'backgroundColor': '#F2F2F2',
                   'overflow-y': 'scroll',
                   'marginLeft': '0px',
                   'justifyContent': 'flex-start',
                   'alignItems': 'center'},
            children=[
                html.Div(
                    id='header-section',
                    children=[
                        html.H4(
                            'Детекция заболеваний полости рта'
                        ),
                        html.P(
                            'Для начала выберите фотоматериал материал, на котором хотите проверить сервис детекции заболеваний. Затем вы можете просмотреть результат работы сервиса и подсчитанную аналитику'
                        ),
                        html.Button("Узнать больше", id="learn-more-button", n_clicks=0)
                    ]
                ),
                html.Div(
                    className='control-section', 
                    children=[
                        html.Div(
                            className='control-element',
                            children=[
                                dcc.Upload(
                                    id='upload-image',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '100%',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '1px',
                                        'textAlign': 'center',
                                        'margin': '1px'
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                )
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div(id='output-image-upload'),
                            ]
                        ),
                        html.Div(
                            className='control-element',
                            children=[
                                html.Div(children=["Graph View Mode:"], style={'width': '40%'}),
                                dcc.Dropdown(
                                    id="dropdown-graph-view-mode",
                                    options=[
                                        {'label': 'Visual Mode', 'value': 'visual'},
                                        {'label': 'Detection Mode', 'value': 'detection'}
                                    ],
                                    value='detection',
                                    searchable=False,
                                    clearable=False,
                                    style={'width': '60%'}
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            id='right-side-column',
            className='four columns',
            style={
                'height': 'calc(100vh - 5px)',
                'overflow-y': 'scroll',
                'marginLeft': '1%',
                'display': 'flex',
                'backgroundColor': '#F9F9F9',
                'flexDirection': 'column'
            },
            children=[
                html.Div(
                    className='img-container',
                    children=html.Img(
                        style={'height': '100%', 'margin': '2px'},
                        src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png")
                ),
                html.Div(id="div-visual-mode"),
                html.Div(id="div-detection-mode")
            ]
        )]),
        markdown_popup()
    ]
)


# Data Loading
@app.server.before_first_request
def load_all_footage():
    global data_dict, url_dict

    # Load the dictionary containing all the variables needed for analysis
    data_dict = {
        'teeth': load_data("data/service_annot.csv"),
    }

    url_dict = {
        'regular': {
            'teeth': 'https://www.youtube.com/watch?v=wrdEE8Br7Zk',
        },

        'bounding_box': {
            'teeth': 'https://www.youtube.com/watch?v=wrdEE8Br7Zk',
        }
    }

def parse_contents(contents, filename, date):

    # Remove 'data:image/png;base64' from the image string,
    # import pdb;pdb.set_trace()
    # see https://stackoverflow.com/a/26079673/11989081
    data = contents.replace('data:image/jpeg;base64,', '')
    # import pdb;pdb.set_trace()
    image = np.asarray(bytearray(base64.b64decode(data)), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image_array = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', img)
    data = img_encoded.tostring()
    # import pdb;pdb.set_trace()
    response = requests.post(URL, data=data, headers=headers)
    metadata = response.json()['image']
    draw_contours(img, metadata)
    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    # import pdb;pdb.set_trace()
    img = Image.open(buffer)

    # Convert the image string to numpy array and create a
    # Plotly figure, see https://plotly.com/python/imshow/
    fig = go.Figure()

    # Constants
    img_width = 1600
    img_height = 1600
    scale_factor = 0.9

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        dcc.Graph(
            figure=fig,
            config={'displayModeBar': True} # Always display the modebar
        )
    ])

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename'),
     State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    # import pdb;pdb.set_trace()
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children

# Learn more popup
@app.callback(Output("markdown", "style"),
              [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")])
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
        
    else:
        return {"display": "none"}


@app.callback(Output("div-visual-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_output(dropdown_value):
    if dropdown_value == "visual":
        return [
            dcc.Interval(
                id="interval-visual-mode",
                interval=700,
                n_intervals=0
            ),
            html.Div(
                children=[
                    html.P(children="Confidence Level of Object Presence",
                           className='plot-title'),
                    dcc.Graph(
                        id="heatmap-confidence",
                        style={'height': '45vh', 'width': '100%'}),

                    html.P(children="Object Count",
                           className='plot-title'),
                    dcc.Graph(
                        id="pie-object-count",
                        style={'height': '40vh', 'width': '100%'}
                    )

                ]
            )
        ]
    else:
        return []


@app.callback(Output("div-detection-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_detection_mode(value):
    if value == "detection":
        return [
            dcc.Interval(
                id="interval-detection-mode",
                interval=700,
                n_intervals=0
            ),
            html.Div(
                children=[
                    html.P(children="Detection Score of Most Probable Objects",
                           className='plot-title'),
                    dcc.Graph(
                        id="bar-score-graph",
                        style={'height': '55vh'}
                    )
                ]
            )
        ]
    else:
        return []


# Updating Figures
@app.callback(Output("bar-score-graph", "figure"),
              [Input("interval-detection-mode", "n_intervals")])
def update_score_bar(n):
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        xaxis={
            'automargin': True,
        },
        yaxis={
            'title': 'Score',
            'automargin': True,
            'range': [0, 1]
        }
    )
    # print(current_time, FRAMERATE)
    # if current_time is not None:
    #     current_frame = round(current_time * FRAMERATE)

    #     if n > 0 and current_frame > 0:
    video_info_df = data_dict["teeth"]["video_info_df"]

    # Select the subset of the dataset that correspond to the current frame
    # frame_df = video_info_df[video_info_df["frame"] == current_frame]

    # Select only the frames above the threshold
    # threshold_dec = threshold / 100  # Threshold in decimal
    # frame_df = frame_df[frame_df["score"] > threshold_dec]

    # Select up to 8 frames with the highest scores
    # frame_df = frame_df[:min(8, frame_df.shape[0])]

    # Add count to object names (e.g. person --> person 1, person --> person 2)
    # objects = frame_df["class_str"].tolist()
    # object_count_dict = {x: 0 for x in set(objects)}  # Keeps count of the objects
    # objects_wc = []  # Object renamed with counts
    # for object in objects:
    #     object_count_dict[object] += 1  # Increment count
    #     objects_wc.append(f"{object} {object_count_dict[object]}")

    colors = list('rgb(250,79,86)' for i in range(2))

    # Add text information
    # y_text = [f"{round(value * 100)}% confidence" for value in video_info_df["score"].tolist()]

    figure = go.Figure({
        'data': [{'hoverinfo': 'x+text',
                    'name': 'Detection Scores',
                    'text': ["70%", "30%"],
                    'type': 'bar',
                    'x': ["teeth", "caries"],
                    'marker': {'color': colors},
                    'y': [0.70, 0.30]}],
        'layout': {'showlegend': False,
                    'autosize': False,
                    'paper_bgcolor': 'rgb(249,249,249)',
                    'plot_bgcolor': 'rgb(249,249,249)',
                    'xaxis': {'automargin': True, 'tickangle': -45},
                    'yaxis': {'automargin': True, 'range': [0, 1], 'title': {'text': 'Score'}}}
        }
    )
    return figure

    # return go.Figure(data=[go.Bar()], layout=layout)  # Returns empty bar


@app.callback(Output("pie-object-count", "figure"),
              [Input("interval-visual-mode", "n_intervals")])
def update_object_count_pie(n):
    layout = go.Layout(
        showlegend=True,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        autosize=False,
        margin=go.layout.Margin(
            l=10,
            r=10,
            t=15,
            b=15
        )
    )

    # if current_time is not None:
    #     current_frame = round(current_time * FRAMERATE)

    #     if n > 0 and current_frame > 0:
    video_info_df = data_dict["teeth"]["video_info_df"]

    # Select the subset of the dataset that correspond to the current frame
    # frame_df = video_info_df[video_info_df["frame"] == current_frame]

    # Select only the frames above the threshold
    # threshold_dec = threshold / 100  # Threshold in decimal
    # frame_df = frame_df[frame_df["score"] > threshold_dec]

    # Get the count of each object class
    class_counts = video_info_df["class"].value_counts()

    classes = class_counts.index.tolist()  # List of each class
    counts = class_counts.tolist()  # List of each count

    text = [f"{count} detected" for count in counts]

    # Set colorscale to piechart
    colorscale = ['#fa4f56', '#fe6767', '#ff7c79', '#ff908b', '#ffa39d', '#ffb6b0', '#ffc8c3', '#ffdbd7',
                    '#ffedeb', '#ffffff']

    pie = go.Pie(
        labels=classes,
        values=counts,
        text=text,
        hoverinfo="text+percent",
        textinfo="label+percent",
        marker={'colors': colorscale[:len(classes)]}
    )
    return go.Figure(data=[pie], layout=layout)

    # return go.Figure(data=[go.Pie()], layout=layout)  # Returns empty pie chart


@app.callback(Output("heatmap-confidence", "figure"),
              [Input("interval-visual-mode", "n_intervals")])
def update_heatmap_confidence(n):
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        autosize=False,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=4
        )
    )

    # if current_time is not None:
    #     current_frame = round(current_time * FRAMERATE)

    #     if n > 0 and current_frame > 0:
            # Load variables from the data dictionary
    print(data_dict.keys())
    video_info_df = data_dict["teeth"]["video_info_df"]
    classes_padded = data_dict["teeth"]["classes_padded"]
    print(classes_padded)
    root_round = data_dict["teeth"]["root_round"]
    classes_matrix = data_dict["teeth"]["classes_matrix"]

    # Select the subset of the dataset that correspond to the current frame
    # frame_df = video_info_df[video_info_df["frame"] == current_frame]

    # Select only the frames above the threshold
    # threshold_dec = threshold / 100
    # frame_df = frame_df[frame_df["score"] > threshold_dec]

    # Remove duplicate, keep the top result
    frame_no_dup = video_info_df[["class", "x_from"]].drop_duplicates("class")
    frame_no_dup.set_index("class", inplace=True)

    # The list of scores
    score_list = []
    for el in classes_padded:
        if el in frame_no_dup.index.values:
            score_list.append(frame_no_dup.loc[el][0])
        else:
            score_list.append(0)

    # Generate the score matrix, and flip it for visual
    score_matrix = np.reshape(score_list, (-1, int(root_round)))
    score_matrix = np.flip(score_matrix, axis=0)

    # We set the color scale to white if there's nothing in the frame_no_dup
    if frame_no_dup.shape != (0, 1):
        colorscale = [[0, '#f9f9f9'], [1, '#fa4f56']]
    else:
        colorscale = [[0, '#f9f9f9'], [1, '#f9f9f9']]

    hover_text = [f"{score * 100:.2f}% confidence" for score in score_list]
    hover_text = np.reshape(hover_text, (-1, int(root_round)))
    hover_text = np.flip(hover_text, axis=0)

    # Add linebreak for multi-word annotation
    classes_matrix = classes_matrix.astype(dtype='|U40')

    for index, row in enumerate(classes_matrix):
        row = list(map(lambda x: '<br>'.join(x.split()), row))
        classes_matrix[index] = row

    # Set up annotation text
    annotation = []
    for y_cord in range(int(root_round)):
        for x_cord in range(int(root_round)):
            annotation_dict = dict(
                showarrow=False,
                text=classes_matrix[y_cord][x_cord],
                xref='x',
                yref='y',
                x=x_cord,
                y=y_cord
            )
            if score_matrix[y_cord][x_cord] > 0:
                annotation_dict['font'] = {'color': '#F9F9F9', 'size': '11'}
            else:
                annotation_dict['font'] = {'color': '#606060', 'size': '11'}
            annotation.append(annotation_dict)

    # Generate heatmap figure

    figure = {
        'data': [
            {'colorscale': colorscale,
                'showscale': False,
                'hoverinfo': 'text',
                'text': hover_text,
                'type': 'heatmap',
                'zmin': 0,
                'zmax': 1,
                'xgap': 1,
                'ygap': 1,
                'z': score_matrix}],
        'layout':
            {'showlegend': False,
                'autosize': False,
                'paper_bgcolor': 'rgb(249,249,249)',
                'plot_bgcolor': 'rgb(249,249,249)',
                'margin': {'l': 10, 'r': 10, 'b': 20, 't': 20, 'pad': 2},
                'annotations': annotation,
                'xaxis': {'showticklabels': False, 'showgrid': False, 'side': 'top', 'ticks': ''},
                'yaxis': {'showticklabels': False, 'showgrid': False, 'side': 'left', 'ticks': ''}
                }
    }

    return figure

    # Returns empty figure
    # return go.Figure(data=[go.Pie()], layout=layout)


# Running the server
if __name__ == '__main__':
    app.run_server(dev_tools_hot_reload=False, debug=DEBUG, host='0.0.0.0')