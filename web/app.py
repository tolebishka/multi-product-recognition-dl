import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
PRICES_PATH = BASE_DIR / "prices" / "prices.json"

# Красивые названия моделей
MODEL_DISPLAY_NAMES = {
    "alexnet": "AlexNet",
    "vgg16": "VGG-16",
    "resnet50": "ResNet-50",
    "inception_v3": "Inception v3",
    "mobilenet_v3_large": "MobileNet v3 Large",
}

from src.models import ModelConfig, build_model
from PIL import UnidentifiedImageError

# Сообщения для разных типов ошибок
FORMAT_ERROR = "Программа не поддерживает такой формат. Загрузите изображение в формате JPG, JPEG или PNG."


def discover_models() -> list[dict]:
    """Сканирует runs/ и возвращает список доступных моделей."""
    models = []
    if not RUNS_DIR.exists():
        return models
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        best_pt = run_dir / "best.pt"
        idx_to_class = run_dir / "idx_to_class.json"
        if not best_pt.exists() or not idx_to_class.exists():
            continue
        # Читаем имя модели из configs.json
        configs_path = run_dir / "configs.json"
        model_key = run_dir.name.split("_")[0]
        if configs_path.exists():
            try:
                cfg = json.loads(configs_path.read_text())
                model_key = cfg.get("model", {}).get("name", model_key)
            except Exception:
                pass
        display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        models.append({
            "id": run_dir.name,
            "name": display,
        })
    return models


def get_model_and_transform(run_id: str):
    """Загружает модель и transform по run_id. Кэширует в памяти."""
    if not hasattr(get_model_and_transform, "_cache"):
        get_model_and_transform._cache = {}

    if run_id in get_model_and_transform._cache:
        return get_model_and_transform._cache[run_id]

    run_dir = RUNS_DIR / run_id
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"Run {run_id} not found")

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model, input_size = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    idx_to_class = json.loads((run_dir / "idx_to_class.json").read_text())

    result = (model, transform, idx_to_class)
    get_model_and_transform._cache[run_id] = result
    return result


# Предзагрузка списка моделей
AVAILABLE_MODELS = discover_models()


def _default_run_id() -> str | None:
    """ResNet-50 по умолчанию, иначе первая доступная модель."""
    for m in AVAILABLE_MODELS:
        if "resnet50" in m["id"]:
            return m["id"]
    return AVAILABLE_MODELS[0]["id"] if AVAILABLE_MODELS else None


DEFAULT_RUN_ID = _default_run_id()

prices = json.loads(PRICES_PATH.read_text())

# ===== FASTAPI =====
app = FastAPI()
templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": AVAILABLE_MODELS,
        "selected_model": DEFAULT_RUN_ID,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    weight: float = Form(1.0),
    model_id: str = Form(None),
):
    # Выбор модели
    run_id = model_id if model_id and any(m["id"] == model_id for m in AVAILABLE_MODELS) else DEFAULT_RUN_ID
    if not run_id:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": AVAILABLE_MODELS,
            "selected_model": DEFAULT_RUN_ID,
            "error": "Нет доступных моделей.",
            "top3": None,
            "result": None,
        })

    try:
        model, transform, idx_to_class = get_model_and_transform(run_id)
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": AVAILABLE_MODELS,
            "selected_model": run_id,
            "error": f"Ошибка загрузки модели: {e}",
            "top3": None,
            "result": None,
        })

    try:
        # 1) Read image safely
        try:
            image = Image.open(file.file).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "models": AVAILABLE_MODELS,
                "selected_model": run_id,
                "error": FORMAT_ERROR,
                "top3": None,
                "result": None,
            })

        # 2) Predict
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, 3)
        indices = topk.indices.tolist()
        values = topk.values.tolist()

        # 3) Build TOP-3 list for display
        top3 = []
        for idx, prob in zip(indices, values):
            label = idx_to_class[str(idx)]
            top3.append({
                "label": label,
                "prob": round(prob * 100, 2)
            })

        # 4) Use only TOP-1 for pricing
        pred_label = top3[0]["label"]
        pred_prob = top3[0]["prob"]

        if pred_label not in prices:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "models": AVAILABLE_MODELS,
                "selected_model": run_id,
                "error": f"Продукт «{pred_label}» не найден в прайсе. Попробуйте другое фото.",
                "top3": top3,
                "result": None,
            })

        price_per_kg = prices[pred_label]["price_kzt"]
        total = round(price_per_kg * weight, 2)

        result = {
            "label": pred_label,
            "prob": pred_prob,
            "price": price_per_kg,
            "weight": weight,
            "total": total
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": AVAILABLE_MODELS,
            "selected_model": run_id,
            "top3": top3,
            "result": result,
            "error": None
        })

    except Exception:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "models": AVAILABLE_MODELS,
            "selected_model": run_id,
            "error": FORMAT_ERROR,
            "top3": None,
            "result": None,
        })
