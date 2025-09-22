import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. DEFINISIKAN ULANG ARSITEKTUR MODEL ===
# pytorch perlu tahu struktur modelnya sebelum bisa memuat bobot (weights).
# copy-paste class AgeGenderModel persis sama seperti saat training.
class AgeGenderModel(nn.Module):
    def __init__(self): # hapus 'num_features', karena sudah ditentukan oleh ResNet50
        super(AgeGenderModel, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # gender branch
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # age branch
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)
        
        gender_output = self.gender_head(features)
        age_output = self.age_head(features)
        
        return gender_output, age_output

# === 2. MUAT MODEL DAN BOBOT YANG TELAH DISIMPAN ===
# Inisialisasi model dengan arsitektur yang sama
model = AgeGenderModel().to(device)

#  .pth
model_path = 'age_gender_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# Set model ke mode evaluasi
model.eval()
print(f"the model success loaded from {model_path}")

# === 3. SIAPKAN GAMBAR DAN FUNGSI PREDIKSI ===
# Transformasi untuk gambar input (harus sama dengan transformasi validasi saat training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    try:
        # buka dan ubah gambar
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Matikan perhitungan gradien untuk inferensi
        with torch.no_grad():
            gender_output, age_output = model(image_tensor)
            
            # Proses output gender
            gender_prob = torch.sigmoid(gender_output).item()
            gender = "Pria" if gender_prob > 0.5 else "Wanita"
            
            # Proses output usia
            age = age_output.item()
            
        print("\n--- Hasil Prediksi ---")
        print(f"Gambar: {image_path}")
        print(f"Jenis Kelamin: {gender} (Probabilitas Pria: {gender_prob:.2f})")
        print(f"Perkiraan Usia: {age:.1f} tahun")
        print("----------------------")

    except FileNotFoundError:
        print(f"Error: File gambar tidak ditemukan di '{image_path}'")

# === 4. JALANKAN PREDIKSI ===
# Ganti 'path/to/your/image.jpg' dengan path gambar yang ingin Anda uji
# Anda bisa download gambar wajah dari internet untuk mencobanya.
path_gambar_tes = '27.jpg'
predict_image(path_gambar_tes, model)

# Contoh lain
# path_gambar_lain = 'wajah_aktor.jpg'
# predict_image(path_gambar_lain, model)