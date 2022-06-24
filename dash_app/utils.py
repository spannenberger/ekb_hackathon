# import dash_core_components as dcc
# import dash_html_components as html
# import json
# import plotly.graph_objs as go
# import dash_reusable_components as drc
# from PIL import Image, ImageFilter, ImageDraw, ImageEnhance


# # [filename, image_signature, action_stack]
# STORAGE_PLACEHOLDER = json.dumps({
#     'filename': None,
#     'image_signature': None, 
#     'action_stack': []
# })

# IMAGE_STRING_PLACEHOLDER = drc.pil_to_b64(Image.open('images/default.jpg').copy(), enc_format='jpeg')

# GRAPH_PLACEHOLDER = dcc.Graph(id='interactive-image', style={'height': '80vh'})

# # Maps process name to the Image filter corresponding to that process
# FILTERS_DICT = {
#     'blur': ImageFilter.BLUR,
#     'contour': ImageFilter.CONTOUR,
#     'detail': ImageFilter.DETAIL,
#     'edge_enhance': ImageFilter.EDGE_ENHANCE,
#     'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
#     'emboss': ImageFilter.EMBOSS,
#     'find_edges': ImageFilter.FIND_EDGES,
#     'sharpen': ImageFilter.SHARPEN,
#     'smooth': ImageFilter.SMOOTH,
#     'smooth_more': ImageFilter.SMOOTH_MORE
# }

# ENHANCEMENT_DICT = {
#     'color': ImageEnhance.Color,
#     'contrast': ImageEnhance.Contrast,
#     'brightness': ImageEnhance.Brightness,
#     'sharpness': ImageEnhance.Sharpness
# }


# def generate_lasso_mask(image, selectedData):
#     """
#     Generates a polygon mask using the given lasso coordinates
#     :param selectedData: The raw coordinates selected from the data
#     :return: The polygon mask generated from the given coordinate
#     """

#     height = image.size[1]
#     y_coords = selectedData['lassoPoints']['y']
#     y_coords_corrected = [height - coord for coord in y_coords]

#     coordinates_tuple = list(zip(selectedData['lassoPoints']['x'], y_coords_corrected))
#     mask = Image.new('L', image.size)
#     draw = ImageDraw.Draw(mask)
#     draw.polygon(coordinates_tuple, fill=255)

#     return mask


# def apply_filters(image, zone, filter, mode):
#     filter_selected = FILTERS_DICT[filter]

#     if mode == 'select':
#         crop = image.crop(zone)
#         crop_mod = crop.filter(filter_selected)
#         image.paste(crop_mod, zone)

#     elif mode == 'lasso':
#         im_filtered = image.filter(filter_selected)
#         image.paste(im_filtered, mask=zone)


# def apply_enhancements(image, zone, enhancement, enhancement_factor, mode):
#     enhancement_selected = ENHANCEMENT_DICT[enhancement]
#     enhancer = enhancement_selected(image)

#     im_enhanced = enhancer.enhance(enhancement_factor)

#     if mode == 'select':
#         crop = im_enhanced.crop(zone)
#         image.paste(crop, box=zone)

#     elif mode == 'lasso':
#         image.paste(im_enhanced, mask=zone)


# def show_histogram(image):
#     def hg_trace(name, color, hg):
#         line = go.Scatter(
#             x=list(range(0, 256)),
#             y=hg,
#             name=name,
#             line=dict(color=(color)),
#             mode='lines',
#             showlegend=False
#         )
#         fill = go.Scatter(
#             x=list(range(0, 256)),
#             y=hg,
#             mode='fill',
#             name=name,
#             line=dict(color=(color)),
#             fill='tozeroy',
#             hoverinfo='none'
#         )

#         return line, fill

#     hg = image.histogram()

#     if image.mode == 'RGBA':
#         rhg = hg[0:256]
#         ghg = hg[256:512]
#         bhg = hg[512:768]
#         ahg = hg[768:]

#         data = [
#             *hg_trace('Red', '#FF4136', rhg),
#             *hg_trace('Green', '#2ECC40', ghg),
#             *hg_trace('Blue', '#0074D9', bhg),
#             *hg_trace('Alpha', 'gray', ahg)
#         ]

#         title = 'RGBA Histogram'

#     elif image.mode == 'RGB':
#         # Returns a 768 member array with counts of R, G, B values
#         rhg = hg[0:256]
#         ghg = hg[256:512]
#         bhg = hg[512:768]

#         data = [
#             *hg_trace('Red', '#FF4136', rhg),
#             *hg_trace('Green', '#2ECC40', ghg),
#             *hg_trace('Blue', '#0074D9', bhg),
#         ]

#         title = 'RGB Histogram'

#     else:
#         data = [*hg_trace('Gray', 'gray', hg)]

#         title = 'Grayscale Histogram'

#     layout = go.Layout(
#         title=title,
#         margin=go.Margin(l=35, r=35),
#         legend=dict(x=0, y=1.15, orientation="h")
#     )

#     return go.Figure(data=data, layout=layout)