from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import io

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega modelo e EMA
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)
ema_state = torch.load('model/model_pneumonia.pth', map_location=device)
model.load_state_dict(ema_state, strict=False)
model.to(device)
model.eval()

# Transformação da imagem
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.post("/predict-pneumonia/")
async def predict_pneumonia(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas imagens são permitidas.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao processar a imagem.")

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        classe = torch.argmax(probs, dim=1).item()
        score = probs[0][classe].item()

    return JSONResponse({
        "class": "PNEUMONIA" if classe == 1 else "NORMAL",
        "confidence": round(score * 100, 2)
    })
