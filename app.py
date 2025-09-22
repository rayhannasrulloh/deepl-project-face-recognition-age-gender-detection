import dash
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

app = dash.Dash(__name__,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
                ],)
server = app.server


# ---  model architecure  ---

# model 1: ResNet50
class ResNet50AgeGenderModel(nn.Module):
    def __init__(self):
        super(ResNet50AgeGenderModel, self).__init__()
        self.base_model = models.resnet50(weights=None)
        in_features = self.base_model.fc.in_features # 2048 untuk ResNet50
        self.base_model.fc = nn.Identity()
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
    def forward(self, x):
        features = self.base_model(x)
        gender_output = self.gender_head(features)
        age_output = self.age_head(features)
        return gender_output, age_output


# model 2: MobileNetV2 (model yang lebih ringan)
class MobileNetV2AgeGenderModel(nn.Module):
    def __init__(self):
        super(MobileNetV2AgeGenderModel, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        # ganti lapisan classifier terakhir dari MobileNetV2
        in_features = self.base_model.classifier[1].in_features # 1280 untuk MobileNetV2
        self.base_model.classifier = nn.Identity()
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
    def forward(self, x):
        features = self.base_model(x)
        gender_output = self.gender_head(features)
        age_output = self.age_head(features)
        return gender_output, age_output

# --- muat SEMUA Model dan Classifier ---

def start_camera_stream(n_clicks, selected_model):
    if n_clicks > 0:
        # ktika di klik, buat URL video stream yang menyertakan nama model
        video_src = f"/video_feed/{selected_model}"
        return html.Img(src=video_src, style={'maxWidth': '80%', 'height': 'auto'})
    return ""
def load_models():
    print("Memuat semua model...")
    device = torch.device("cpu")
    
    # ngeload model ResNet50
    resnet_model = ResNet50AgeGenderModel().to(device)
    # NOTE: Bobot yang dilatih untuk ResNet50 tidak kompatibel dengan MobileNetV2.
    # tpi biar aplikasi bisa jalan kita muat arsitekturnya aja
    # butuh file bobot terpisah untuk setiap model.
    try:
        resnet_model.load_state_dict(torch.load('age_gender_model.pth', map_location=device))
    except Exception as e:
        print(f"Peringatan: Tidak bisa memuat bobot untuk ResNet50. Error: {e}")

    # muat arsitektur MobileNetV2
    mobilenet_model = MobileNetV2AgeGenderModel().to(device)
    # disini Anda akan memuat file bobot untuk mobilenet, contoh:
    # mobilenet_model.load_state_dict(torch.load('mobilenet_v2_age_gender.pth', map_location=device))
    
    models_dict = {
        'resnet50': resnet_model,
        'mobilenetv2': mobilenet_model
    }
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Semua model berhasil dimuat.")
    return models_dict, face_cascade

models_dict, face_cascade = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- app layout ---
app.layout = html.Div(style={'textAlign': 'center', 'fontFamily': 'Comic Sans MS'}, children=[
    html.H1("Dashboard Deteksi Usia & Jenis Kelamin"),
    
    # dropdown untuk memilih model
    html.Div([
        html.H3("Pilih Model Prediksi:"),
        dcc.Dropdown(
            id='model-selector-dropdown',
            options=[
                {'label': 'ResNet50 (Akurat)', 'value': 'resnet50'},
                {'label': 'MobileNetV2 (Cepat)', 'value': 'mobilenetv2'},
            ],
            value='resnet50', # model default
            clearable=False,
            style={'width': '50%', 'margin': '0 auto'}
        )
    ]),
    
    html.Hr(style={'margin-block': '20px'}),

    html.H2("Mode Upload Gambar"),
    dcc.Upload(id='upload-image', children=html.Div(['Seret dan Lepas atau ', html.A('Pilih File Gambar')]), style={'width': '50%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px auto'}),
    dcc.Loading(id="loading-output", type="circle", children=html.Div(id='output-image-display')),
    
    html.Hr(style={'margin-block': '20px'}),

    html.H2("Mode Kamera Live"),
    html.Button('Start Kamera', id='start-camera-button', n_clicks=0),
    html.Div(id='live-camera-container', style={'marginTop': '20px'})
])

# --- 5. Logika untuk Unggah Gambar (Callback) ---
@app.callback(
    Output('output-image-display', 'children'),
    Input('upload-image', 'contents'),
    [State('upload-image', 'filename'),
    State('model-selector-dropdown', 'value')] # Tambahkan state dari dropdown
)
def update_output(contents, filename, selected_model_name):
    if contents is None:
        return html.P("Silakan unggah gambar untuk memulai prediksi.")
    
    # pilih model berdasarkan value dari dropdown
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
        text = f"{gender}, {age:.1f} yo"
        cv2.putText(image_cv, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img_result = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img_result.save(buffer, format="PNG")
    encoded_image_string = base64.b64encode(buffer.getvalue()).decode()
    return html.Div([
        html.H5(f"Hasil prediksi untuk: {filename} (menggunakan {selected_model_name})"),
        html.Img(src=f'data:image/png;base64,{encoded_image_string}', style={'maxWidth': '80%', 'height': 'auto'})
    ])

# --- 6. Callback untuk Tombol Start Kamera ---
@app.callback(
    Output('live-camera-container', 'children'),
    [Input('start-camera-button', 'n_clicks')],
    [State('model-selector-dropdown', 'value')] # Ambil state dari dropdown
)
def start_camera_stream(n_clicks, selected_model_name):
    if n_clicks > 0:
        # Saat tombol diklik, buat URL video stream yang menyertakan nama model
        video_src = f"/video_feed/{selected_model_name}"
        return html.Img(src=video_src, style={'maxWidth': '80%', 'height': 'auto'})
    return ""

# --- 7. Logika untuk Video Streaming (Flask Route) ---
def generate_frames(model_name): # terima nama model sebagai argumen
    camera = cv2.VideoCapture(0)
    model = models_dict[model_name] # pilih model yang sesuai
    
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
                    gender = "Male" if gender_prob > 0.5 else "Female"
                    age = age_output.item()
                text = f"{gender}, {age:.1f} thn"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@server.route('/video_feed/<model_name>') # buat URL dinamis
def video_feed(model_name):
    return Response(generate_frames(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 8. Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)