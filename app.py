# FINAL APP WITH MODERN UI AND MULTI-MODEL SELECTION

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

# --- 1. Model Architecture Definitions ---

# Model 1: ResNet50
class ResNet50AgeGenderModel(nn.Module):
    def __init__(self): 
        super(ResNet50AgeGenderModel, self).__init__()
        self.base_model = models.resnet50(weights=None) # Use weights=None for compatibility
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.gender_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
        self.age_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x):
        features = self.base_model(x)
        return self.gender_head(features), self.age_head(features)

# Model 2: MobileNetV2
class MobileNetV2AgeGenderModel(nn.Module):
    def __init__(self):
        super(MobileNetV2AgeGenderModel, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.gender_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.age_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
    def forward(self, x):
        features = self.base_model(x)
        return self.gender_head(features), self.age_head(features)

# Model 3: EfficientNet-B0
class EfficientNetAgeGenderModel(nn.Module):
    def __init__(self):
        super(EfficientNetAgeGenderModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.gender_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.age_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
    def forward(self, x):
        features = self.base_model(x)
        return self.gender_head(features), self.age_head(features)

# --- 2. Load All Models and Classifiers ---
def load_models():
    print("Loading all models...")
    device = torch.device("cpu")
    models_dict = {}

    # Load ResNet50
    resnet_model = ResNet50AgeGenderModel().to(device)
    try:
        # NOTE: Make sure you have a trained 'resnet50_age_gender.pth' file
        resnet_model.load_state_dict(torch.load('resnet50_age_gender.pth', map_location=device))
        models_dict['resnet50'] = resnet_model
        print("Weights for ResNet50 loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'resnet50_age_gender.pth' not found. ResNet50 will not be available.")
    except Exception as e:
        print(f"Warning: Failed to load weights for ResNet50. Error: {e}")

    # Load MobileNetV2
    mobilenet_model = MobileNetV2AgeGenderModel().to(device)
    try:
        # NOTE: Make sure you have a trained 'mobilenetv2_age_gender.pth' file
        mobilenet_model.load_state_dict(torch.load('mobilenetv2_age_gender.pth', map_location=device))
        models_dict['mobilenetv2'] = mobilenet_model
        print("Weights for MobileNetV2 loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'mobilenetv2_age_gender.pth' not found. MobileNetV2 will not be available.")
    except Exception as e:
        print(f"Warning: Failed to load weights for MobileNetV2. Error: {e}")

    # Load EfficientNet
    efficientnet_model = EfficientNetAgeGenderModel().to(device)
    try:
        # NOTE: Make sure you have a trained 'efficientnet_age_gender.pth' file
        efficientnet_model.load_state_dict(torch.load('efficientnet_age_gender.pth', map_location=device))
        models_dict['efficientnet'] = efficientnet_model
        print("Weights for EfficientNet loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'efficientnet_age_gender.pth' not found. EfficientNet will not be available.")
    except Exception as e:
        print(f"Warning: Failed to load weights for EfficientNet. Error: {e}")
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Model initialization complete.")
    return models_dict, face_cascade

models_dict, face_cascade = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. Initialize Dash App with Bootstrap Theme ---
# You can choose a theme from: https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.CYBORG],
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
                ],)
server = app.server

# --- 4. Define the Application Layout ---
app.layout = dbc.Container([
    # Header / Navbar
    dbc.NavbarSimple(
        brand="🤖 Age & Gender Detection AI",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Main Content
    dbc.Row([
        # Left Column: Controls
        dbc.Col(md=4, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Controls", className="card-title"),
                    dbc.Label("Select Prediction Model:"),
                    dcc.Dropdown(
                        id='model-selector-dropdown',
                        options=[
                            {'label': 'ResNet50 (Accurate)', 'value': 'resnet50'},
                            {'label': 'MobileNetV2 (Fast)', 'value': 'mobilenetv2'},
                            {'label': 'EfficientNet (Balanced)', 'value': 'efficientnet'},
                        ],
                        value='resnet50',
                        clearable=False,
                    ),
                ])
            ]),
            dbc.Card(className="mt-4", children=[
                dbc.CardBody([
                     html.H4("Live Camera Mode", className="card-title"),
                     html.P("Activate your webcam for real-time detection."),
                     dbc.Button('Start / Stop Camera', id='start-camera-button', color="success", n_clicks=0, className="w-100"),
                ])
            ])
        ]),
        
        # Right Column: Output Display
        dbc.Col(md=8, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prediction Result", className="card-title"),
                    # Image Upload Area
                    dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select an Image File')]), 
                               style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '20px'}),
                    dbc.Spinner(html.Div(id='output-image-display'), color="light"),
                    # Live Camera Area
                    html.Div(id='live-camera-container', className="mt-4 text-center")
                ])
            ])
        ])
    ])
], fluid=True) # fluid=True makes the layout fill the screen width

# --- 5. Callbacks & Backend Logic ---

# Callback for Image Upload
@app.callback(
    Output('output-image-display', 'children'),
    Input('upload-image', 'contents'),
    [State('upload-image', 'filename'), State('model-selector-dropdown', 'value')]
)
def update_output(contents, filename, selected_model_name):
    if contents is None:
        return html.P("Please upload an image to see the results here.", className="text-center text-muted")
    
    if selected_model_name not in models_dict:
        return dbc.Alert(f"Error: Model '{selected_model_name}' is not available. Its weights might be missing.", color="danger")
        
    model = models_dict[selected_model_name]
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image_pil = Image.open(io.BytesIO(decoded)).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = image_pil.crop((x, y, x+w, y+h))
        image_tensor = transform(face_roi).unsqueeze(0)
        with torch.no_grad():
            gender_output, age_output = model(image_tensor)
            gender_prob = torch.sigmoid(gender_output).item()
            gender = "Male" if gender_prob > 0.5 else "Female"
            age = age_output.item()
        text = f"{gender}, {age:.1f} years"
        cv2.putText(image_cv, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img_result = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img_result.save(buffer, format="PNG")
    encoded_image_string = base64.b64encode(buffer.getvalue()).decode()
    
    return html.Div([
        html.H5(f"Results for: {filename} (using {selected_model_name})"),
        html.Img(src=f'data:image/png;base64,{encoded_image_string}', style={'maxWidth': '100%', 'height': 'auto', 'marginTop': '10px'})
    ])

# Callback for the Start/Stop Camera Button
@app.callback(
    Output('live-camera-container', 'children'),
    Input('start-camera-button', 'n_clicks'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def toggle_camera_stream(n_clicks, selected_model_name):
    # n_clicks % 2 == 1 means the button has been clicked an odd number of times (to start)
    if n_clicks % 2 == 1:
        if selected_model_name not in models_dict:
            return dbc.Alert(f"Error: Model '{selected_model_name}' is not available. Its weights might be missing.", color="danger")
        video_src = f"/video_feed/{selected_model_name}"
        return html.Img(src=video_src, style={'maxWidth': '100%', 'height': 'auto'})
    # n_clicks % 2 == 0 means the button was clicked again (even) to stop
    return html.P("Camera is currently off.", className="text-center text-muted")

# Video Streaming Logic
def generate_frames(model_name):
    camera = cv2.VideoCapture(0)
    model = models_dict[model_name]
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            continue
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = pil_frame.crop((x, y, x+w, y+h))
                image_tensor = transform(face_roi).unsqueeze(0)
                with torch.no_grad():
                    gender_output, age_output = model(image_tensor)
                    gender_prob = torch.sigmoid(gender_output).item()
                    gender = "Male" if gender_prob > 0.5 else "Female"
                    age = age_output.item()
                text = f"{gender}, {age:.1f} yrs"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # Yield the frame in the correct format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@server.route('/video_feed/<model_name>')
def video_feed(model_name):
    return Response(generate_frames(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Run the Application ---
if __name__ == '__main__':
    app.run(debug=True)

