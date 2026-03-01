import torch
from .agman_model import AGMAN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

agman = AGMAN()
state_dict = torch.load("models/agman_model_best.pth", map_location=DEVICE)

# Remove classifier weights if they exist (not needed for inference)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}

agman.load_state_dict(state_dict)
agman = agman.to(DEVICE)
agman.eval()

def refine_embedding(base_embedding):
    """
    base_embedding: list or np.array (2048,)
    returns: list (512,)
    """
    x = torch.tensor(base_embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        refined = agman(x)
    return refined.cpu().numpy()[0].tolist()
