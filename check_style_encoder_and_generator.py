import torch
import torch.nn.functional as F
from core.model import build_model
from torchvision import transforms
from PIL import Image
import argparse

# ---------------------
# 1. 기본 세팅
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--style_dim', type=int, default=64)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--num_domains', type=int, default=2)
parser.add_argument('--w_hpf', type=float, default=0)
args = parser.parse_args(args=[])

# Load model
nets, nets_ema = build_model(args)
ckpt = torch.load('expr/checkpoints/003000_nets_ema.ckpt', map_location='cpu')

# Key 이름 맞추기
def fix_key(state_dict):
    return {f"module.{k}" if not k.startswith("module.") else k: v for k, v in state_dict.items()}

nets_ema.style_encoder.load_state_dict(fix_key(ckpt['style_encoder']))
nets_ema.generator.load_state_dict(fix_key(ckpt['generator']))
nets_ema.mapping_network.load_state_dict(fix_key(ckpt['mapping_network']))

nets_ema.style_encoder.eval()
nets_ema.generator.eval()
nets_ema.mapping_network.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for name, module in nets_ema.items():
    nets_ema[name] = module.to(device)

# ---------------------
# 2. 이미지 불러오기
# ---------------------
def load_image(path):
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)

# 여기에 직접 경로 설정
ref1 = load_image("assets/inputs/ref/scene/8003-20-0022_1.png")  # 첫 번째 reference 이미지 경로
ref2 = load_image("assets/inputs/ref/scene/8003-20-0023_1.png")  # 두 번째 reference 이미지 경로
input_x = load_image("assets/inputs/src/register_binary/B232668.png")  # 등록된 신발 밑창 이미지

y = torch.LongTensor([0]).to(device)  # domain 0 기준

# ---------------------
# 3. StyleEncoder 다양성 확인
# ---------------------
s1 = nets_ema.style_encoder(ref1, y)
s2 = nets_ema.style_encoder(ref2, y)
dist = F.mse_loss(s1, s2).item()
print(f"[StyleEncoder] distance(s1, s2) = {dist:.6f}")

# ---------------------
# 4. Generator에 다양한 style 넣어보기
# ---------------------
print("\n[Generator - latent-guided]")
for i in range(3):
    z = torch.randn(1, args.latent_dim).to(device)
    s = nets_ema.mapping_network(z, y)
    x_fake = nets_ema.generator(input_x, s)
    save_path = f"test_out/fake_{i}.png"
    transforms.ToPILImage()(x_fake[0].cpu().clamp(0, 1)).save(save_path)
    print(f"Saved {save_path}")
