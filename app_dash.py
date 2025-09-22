# KODE FINAL DENGAN UI MODERN (DASH BOOTSTRAP COMPONENTS)

import dash
import dash_bootstrap_components as dbc # Import library baru
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

# --- 1. Definisi Arsitektur Model (Tidak ada perubahan) ---
class ResNet50AgeGenderModel(nn.Module):
    def __init__(self): 
        super(ResNet50AgeGenderModel, self).__init__()
        self.base_model = models.resnet50(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.gender_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
        self.age_head = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x):
        features = self.base_model(x)
        return self.gender_head(features), self.age_head(features)

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

# --- 2. Muat Model dan Classifier (Tidak ada perubahan) ---
def load_models():
    print("Memuat semua model...")
    device = torch.device("cpu")
    resnet_model = ResNet50AgeGenderModel().to(device)
    try:
        resnet_model.load_state_dict(torch.load('age_gender_model.pth', map_location=device))
        print("Bobot untuk ResNet50 berhasil dimuat.")
    except Exception as e:
        print(f"Peringatan: Gagal memuat bobot untuk ResNet50. Error: {e}")
    mobilenet_model = MobileNetV2AgeGenderModel().to(device)
    efficientnet_model = EfficientNetAgeGenderModel().to(device)
    models_dict = {'resnet50': resnet_model, 'mobilenetv2': mobilenet_model, 'efficientnet': efficientnet_model}
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Semua model berhasil diinisialisasi.")
    return models_dict, face_cascade
models_dict, face_cascade = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. Inisialisasi Aplikasi Dash dengan Tema Bootstrap ---
# Pilih tema dari: https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])
server = app.server

# --- 4. Definisikan Layout Aplikasi yang Baru ---
app.layout = dbc.Container([
    # Header / Navbar
    dbc.NavbarSimple(
        brand="🤖 Deteksi Usia & Jenis Kelamin",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Konten Utama
    dbc.Row([
        # Kolom Kiri: Kontrol
        dbc.Col(md=4, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Pengaturan", className="card-title"),
                    dbc.Label("Pilih Model Prediksi:"),
                    dcc.Dropdown(
                        id='model-selector-dropdown',
                        options=[
                            {'label': 'ResNet50 (Akurat)', 'value': 'resnet50'},
                            {'label': 'MobileNetV2 (Cepat)', 'value': 'mobilenetv2'},
                            {'label': 'EfficientNet (Seimbang)', 'value': 'efficientnet'},
                        ],
                        value='resnet50',
                        clearable=False,
                    ),
                ])
            ]),
            dbc.Card(className="mt-4", children=[
                dbc.CardBody([
                     html.H4("Mode Kamera Live", className="card-title"),
                     html.P("Nyalakan kamera untuk deteksi real-time."),
                     dbc.Button('Start / Stop Kamera', id='start-camera-button', color="success", n_clicks=0, className="w-100"),
                ])
            ])
        ]),
        
        # Kolom Kanan: Tampilan Output
        dbc.Col(md=8, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Hasil Prediksi", className="card-title"),
                    
                    # Tampilan Unggah Gambar
                    html.Div(id="upload-section", children=[
                        dcc.Upload(id='upload-image', children=html.Div(['Seret dan Lepas atau ', html.A('Pilih File Gambar')]), 
                                   style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '20px'}),
                        dbc.Spinner(html.Div(id='output-image-display'), color="light"),
                    ]),
                    
                    # Tampilan Kamera Live
                    html.Div(id='live-camera-container', className="mt-4 text-center")
                ])
            ])
        ])
    ])
], fluid=True) # fluid=True agar layout memenuhi lebar layar

# --- 5. Callback & Logika Backend (Hampir tidak ada perubahan) ---

# Callback untuk Unggah Gambar
@app.callback(
    Output('output-image-display', 'children'),
    Input('upload-image', 'contents'),
    [State('upload-image', 'filename'), State('model-selector-dropdown', 'value')]
)
def update_output(contents, filename, selected_model_name):
    if contents is None:
        return html.P("Silakan unggah gambar untuk melihat hasilnya di sini.", className="text-center text-muted")
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
            gender = "Pria" if gender_prob > 0.5 else "Wanita"
            age = age_output.item()
        text = f"{gender}, {age:.1f} tahun"
        cv2.putText(image_cv, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img_result = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img_result.save(buffer, format="PNG")
    encoded_image_string = base64.b64encode(buffer.getvalue()).decode()
    return html.Div([
        html.H5(f"Hasil untuk: {filename} ({selected_model_name})"),
        html.Img(src=f'data:image/png;base64,{encoded_image_string}', style={'maxWidth': '100%', 'height': 'auto', 'marginTop': '10px'})
    ])

# Callback untuk Tombol Start/Stop Kamera
@app.callback(
    Output('live-camera-container', 'children'),
    Input('start-camera-button', 'n_clicks'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def toggle_camera_stream(n_clicks, selected_model_name):
    # n_clicks % 2 == 1 artinya tombol ditekan (ganjil)
    if n_clicks % 2 == 1:
        video_src = f"/video_feed/{selected_model_name}"
        return html.Img(src=video_src, style={'maxWidth': '100%', 'height': 'auto'})
    # n_clicks % 2 == 0 artinya tombol ditekan lagi (genap) untuk stop
    return html.P("Kamera dinonaktifkan.", className="text-center text-muted")

# Logika Video Streaming
def generate_frames(model_name):
    camera = cv2.VideoCapture(0)
    model = models_dict[model_name]
    if not camera.isOpened():
        print("Error: Kamera tidak bisa dibuka.")
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
                    gender = "Pria" if gender_prob > 0.5 else "Wanita"
                    age = age_output.item()
                text = f"{gender}, {age:.1f} thn"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release() # Tambahkan ini untuk melepaskan kamera saat loop berhenti

@server.route('/video_feed/<model_name>')
def video_feed(model_name):
    return Response(generate_frames(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)