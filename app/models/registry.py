from app.models.adain import AdaINModel
from app.models.wct import WCTModel

model_map = {
    "adain": AdaINModel,
    "wct": WCTModel
}

def get_model(name: str):
    name = name.lower()
    if name in model_map:
        return model_map[name]()
    raise ValueError(f"Model '{name}' is not supported.")