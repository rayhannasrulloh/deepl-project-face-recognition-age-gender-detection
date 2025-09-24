# =============================================================================
# DASH WEB APPLICATION FOR AGE, GENDER, AND ETHNICITY DETECTION
# =============================================================================
# This application provides a user interface for the trained models.
# Features:
# 1. A modern, responsive UI using Dash Bootstrap Components.
# 2. A dropdown to select between different trained models.
# 3. An image upload feature for static prediction.
# 4. A live camera feature for real-time prediction.
# =============================================================================

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Response
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import base64
import io

# --- 1. SETUP & MODEL DEFINITIONS ---
# Define ethnicity and gender mapping
ETHNICITY_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
GENDER_MAP = {0: 'Male', 1: 'Female'}

# Re-define the model architectures exactly as in the training script
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, in_features, num_ethnicities):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.age_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.gender_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.ethnicity_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_ethnicities))
    def forward(self, x):
        features = self.base_model(x)
        return (self.age_head(features).squeeze(1), self.gender_head(features), self.ethnicity_head(features))

def get_model_architecture(model_name, num_ethnicities):
    if model_name == 'resnet50':
        base = models.resnet50(weights=None)
        in_features = base.fc.in_features; base.fc = nn.Identity()
    elif model_name == 'mobilenetv2':
        base = models.mobilenet_v2(weights=None)
        in_features = base.classifier[1].in_features; base.classifier = nn.Identity()
    elif model_name == 'efficientnet':
        base = models.efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features; base.classifier = nn.Identity()
    else: raise ValueError("Unknown model name")
    return MultiTaskModel(base, in_features, num_ethnicities)

# --- 2. LOAD MODELS ---
def load_all_models():
    print("Loading all available models...")
    device = torch.device("cpu")
    models_dict = {}
    model_names = ['resnet50', 'mobilenetv2', 'efficientnet']
    num_ethnicities = 5 # Based on the dataset

    for name in model_names:
        try:
            model_path = f"{name}_age_gender_ethnicity.pth"
            model = get_model_architecture(name, num_ethnicities)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models_dict[name] = model
            print(f"-> Weights for '{name}' loaded successfully.")
        except FileNotFoundError:
            print(f"-> Warning: '{model_path}' not found. This model will not be available.")
        except Exception as e:
            print(f"-> Warning: Failed to load '{name}'. Error: {e}")
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Model initialization complete.")
    return models_dict, face_cascade

models_dict, face_cascade = load_all_models()

# Define image transform (must be same as validation transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. DASH APP INITIALIZATION ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
server = app.server

# --- 4. APP LAYOUT ---
app.layout = dbc.Container([
    dbc.NavbarSimple(brand=" Ethnicity, Gender & Age Detection AI ", color="primary", dark=True, className="mb-4"),
    dbc.Row([
        # Left Column: Controls
        dbc.Col(md=4, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Controls", className="card-title"),
                    dbc.Label("1. Select Prediction Model:"),
                    dcc.Dropdown(
                        id='model-selector-dropdown',
                        options=[{'label': name.upper(), 'value': name} for name in models_dict.keys()],
                        value=next(iter(models_dict.keys()), None), # Default to first available model
                        clearable=False,
                    ),
                    html.Hr(),
                    dbc.Label("2. Choose Input Method:"),
                    dbc.Tabs(id="input-method-tabs", active_tab="tab-upload", children=[
                        dbc.Tab(label="Image Upload", tab_id="tab-upload"),
                        dbc.Tab(label="Live Camera", tab_id="tab-camera"),
                    ]),
                    html.Div(id="tab-content", className="pt-4")
                ])
            ]),
        ]),
        # Right Column: Output
        dbc.Col(md=8, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prediction Result", className="card-title"),
                    dbc.Spinner(html.Div(id='output-display'), color="light")
                ])
            ])
        ])
    ])
], fluid=True)

# --- 5. CALLBACKS ---
# Callback to switch between upload and camera controls
@app.callback(Output('output-display', 'children'),
              Output('tab-content', 'children'),
              Input('input-method-tabs', 'active_tab'))
def switch_tab(at):
    if at == "tab-upload":
        upload_content = dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image File')]),
            style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '20px'}
        )
        return html.P("Upload an image to see the results.", className="text-center text-muted"), upload_content
    if at == "tab-camera":
        camera_content = dbc.Button('Start / Stop Camera', id='start-camera-button', color="success", n_clicks=0, className="w-100")
        return html.P("Click the button to start the camera.", className="text-center text-muted"), camera_content
    return "Something went wrong"

# Callback for Image Upload prediction
@app.callback(
    Output('output-display', 'children', allow_duplicate=True),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def update_upload_output(contents, filename, model_name):
    if contents is None:
        return html.P("Upload an image to see the results.", className="text-center text-muted")
    
    if not model_name:
        return dbc.Alert("Please select a model first.", color="warning")

    model = models_dict[model_name]
    
    # Process image
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image_pil = Image.open(io.BytesIO(decoded)).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = image_pil.crop((x, y, x+w, y+h))
        image_tensor = transform(face_roi).unsqueeze(0)
        with torch.no_grad():
            age_pred, gender_pred, ethnicity_pred = model(image_tensor)
            gender = GENDER_MAP[torch.sigmoid(gender_pred).round().long().item()]
            ethnicity = ETHNICITY_MAP[torch.max(ethnicity_pred, 1)[1].item()]
            age = f"{age_pred.item():.1f}"
        
        text = f"{ethnicity}, {gender}, {age} yrs"
        cv2.putText(image_cv, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img_result = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img_result.save(buffer, format="PNG")
    encoded_image_string = base64.b64encode(buffer.getvalue()).decode()
    
    return html.Img(src=f'data:image/png;base64,{encoded_image_string}', style={'maxWidth': '100%', 'height': 'auto'})

# Callback for Camera Button
@app.callback(
    Output('output-display', 'children', allow_duplicate=True),
    Input('start-camera-button', 'n_clicks'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def toggle_camera(n_clicks, model_name):
    if n_clicks % 2 == 1: # Start camera
        if not model_name:
            return dbc.Alert("Please select a model first.", color="warning")
        return html.Img(src=f"/video_feed/{model_name}", style={'maxWidth': '100%', 'height': 'auto'})
    else: # Stop camera
        return html.P("Camera is off.", className="text-center text-muted")

# --- 6. VIDEO STREAMING LOGIC ---
def generate_frames(model_name):
    camera = cv2.VideoCapture(0)
    model = models_dict[model_name]
    if not camera.isOpened(): return

    while True:
        success, frame = camera.read()
        if not success: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = pil_frame.crop((x, y, x+w, y+h))
            image_tensor = transform(face_roi).unsqueeze(0)
            with torch.no_grad():
                age_pred, gender_pred, ethnicity_pred = model(image_tensor)
                gender = GENDER_MAP[torch.sigmoid(gender_pred).round().long().item()]
                ethnicity = ETHNICITY_MAP[torch.max(ethnicity_pred, 1)[1].item()]
                age = f"{age_pred.item():.1f}"
            text = f"{ethnicity}, {gender}, {age} yrs"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

@server.route('/video_feed/<model_name>')
def video_feed(model_name):
    return Response(generate_frames(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 7. RUN APP ---
if __name__ == '__main__':
    app.run(debug=True)

