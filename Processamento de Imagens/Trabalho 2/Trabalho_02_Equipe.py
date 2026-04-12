# # UNIVERSIDADE REGIONAL DE BLUMENAU - FURB
# **DISCIPLINA**: Processamento de Imagens
# **ALUNO**: Gabriel Utyama
#
# Segmentação da pista (asfalto) na imagem de satélite do Autódromo de Interlagos
# usando operadores morfológicos (chapéu-preto, limiarização, abertura/fechamento,
# remoção de pequenos objetos) conforme o roteiro do trabalho prático.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import morphology
from skimage.morphology import disk
from skimage.morphology.binary import binary_closing, binary_opening


def imread_unicode(path: Path):
    """Lê imagem com OpenCV mesmo quando o caminho contém acentos (Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir: {path}")
    return img


def show_img(title, img, cmap=None):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap="gray")
    plt.axis("off")
    plt.show()


def excess_green_mask_bgr(bgr: np.ndarray, thresh: float = 22.0) -> np.ndarray:
    """
    Índice Excess Green (2G - R - B): vegetação apresenta valores altos;
    asfalto e áreas urbanas tendem a ficar abaixo do limiar.
    Retorna máscara booleana True onde NÃO há vegetação forte (candidatos à pista).
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    return exg < thresh


def black_hat_disk(gray: np.ndarray, radius: int) -> np.ndarray:
    """Chapéu-preto com elemento estruturante tipo disco (via elipse cheia)."""
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def largest_component(mask_bool: np.ndarray) -> np.ndarray:
    """Mantém apenas o maior componente conexo (pista principal)."""
    if not np.any(mask_bool):
        return mask_bool
    mask_u8 = mask_bool.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_bool
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


# %% [markdown]
# ## 1. Obtenção da imagem
#
# Imagem de satélite do Autódromo José Carlos Pace (Interlagos), São Paulo.

# %%
BASE = Path(__file__).resolve().parent
IMAGE_PATH = BASE / "Satélite Autódromo.jpg"

img_bgr = imread_unicode(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

h, w = img_bgr.shape[:2]
print(f"Resolução: {w}×{h}")

show_img("Imagem original (satélite)", img_bgr)

# %% [markdown]
# ## 2. Pré-processamento
#
# - Conversão para tons de cinza e leve suavização para reduzir ruído de sensor.
# - **Chapéu-preto (black-hat)**: realça estruturas mais escuras que o entorno (asfalto).
# - **CLAHE**: equalização adaptativa (melhor que `equalizeHist` global em cenas heterogêneas).

# %%
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Raio adaptativo ao tamanho da imagem (pistas largas em imagens grandes precisam de kernel maior)
bh_radius = int(max(10, min(h, w) // 55))
blackhat = black_hat_disk(gray, bh_radius)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(blackhat)

show_img(f"Após chapéu-preto (raio≈{bh_radius}) + CLAHE", enhanced, cmap="gray")

# %% [markdown]
# ## 3. Detecção e pós-processamento
#
# - Limiarização de Otsu na imagem realçada.
# - Filtro espacial por **área mínima** (`remove_small_objects`) para descartar telhados e ruídos.
# - **Fechamento morfológico** para unir falhas no asfalto (sombras, marcações).
# - **Abertura** leve para cortar pontes finas para o entorno urbano.
# - **Supressão de vegetação** (Excess Green) para reduzir manchas de grama.
# - Opcional: **remoção de buracos** internos e seleção do **maior componente** (pista principal, excluindo kartódromo fino se desconexo).

# %%
# Limiar de Otsu
otsu_val, thresh_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_bool = thresh_img > 0

# Área mínima proporcional à resolução (ajuste fino se necessário)
min_cc_area = max(800, int(h * w * 0.00012))
clean = morphology.remove_small_objects(thresh_bool, min_size=min_cc_area)

# Fechamento + abertura (elemento em cruz, como no roteiro original — com mais iterações)
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
u8 = clean.astype(np.uint8) * 255
u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cross, iterations=3)
u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cross, iterations=1)
morph_bool = u8 > 0

# Reforço com operações em booleano (disco) para continuidade da pista
se_close = disk(max(4, bh_radius // 2))
morph_bool = binary_closing(morph_bool, se_close)
se_open = disk(3)
morph_bool = binary_opening(morph_bool, se_open)

# Vegetação: restringe candidatos (asfalto não é verde)
veg_ok = excess_green_mask_bgr(img_bgr, thresh=22.0)
combined = morph_bool & veg_ok
if not np.any(combined):
    combined = morph_bool.copy()

# Buracos pequenos dentro da máscara
combined = morphology.remove_small_holes(combined, area_threshold=max(400, min_cc_area // 2))

# Maior componente = circuito principal (evita kartódromo/auxiliares menores, se desconectados)
TRACK_LARGEST_ONLY = True
if TRACK_LARGEST_ONLY:
    combined = largest_component(combined)

final_feat = (combined.astype(np.uint8)) * 255

print(f"Otsu ≈ {otsu_val:.1f} | min_cc_area={min_cc_area} | black-hat raio={bh_radius}")
show_img("Máscara da pista (binária)", final_feat, cmap="gray")

# %% [markdown]
# ## 4. Resultado (sobreposição em vermelho)

# %%
overlay = img_rgb.copy().astype(np.float32)
overlay[combined] = [255, 0, 0]
alpha = 0.55
result_rgb = (alpha * overlay + (1.0 - alpha) * img_rgb.astype(np.float32)).astype(np.uint8)

plt.figure(figsize=(14, 12))
plt.title("Sobreposição da pista detectada (vermelho)")
plt.imshow(result_rgb)
plt.axis("off")
plt.show()
