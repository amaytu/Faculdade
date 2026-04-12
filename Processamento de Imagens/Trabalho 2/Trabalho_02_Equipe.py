# # UNIVERSIDADE REGIONAL DE BLUMENAU - FURB
# **DISCIPLINA**: Processamento de Imagens
# **ALUNO**: Gabriel Utyama
#
# Roteiro (artigo / trabalho prático): imagem em tons de cinza → chapéu-preto → equalização
# → limiarização → operadores morfológicos → sobreposição. Ajustes marcam melhor a pista.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def find_satellite_jpg() -> Path:
    names = ("Satélite Autódromo.jpg", "Satelite Autodromo.jpg")
    dirs: list[Path] = []
    for d in (_script_dir(), Path.cwd().resolve()):
        if d not in dirs:
            dirs.append(d)
    repo_sub = Path.cwd() / "Processamento de Imagens" / "Trabalho 2"
    if repo_sub.is_dir():
        r = repo_sub.resolve()
        if r not in dirs:
            dirs.append(r)
    for directory in dirs:
        for name in names:
            p = (directory / name).resolve()
            if p.is_file():
                return p
    raise FileNotFoundError(
        "Imagem não encontrada. Coloque 'Satélite Autódromo.jpg' numa destas pastas:\n"
        + "\n".join(f"  - {d}" for d in dirs)
    )


def imread_bgr(path: Path) -> np.ndarray:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Arquivo vazio: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"OpenCV não decodificou a imagem (formato inválido?): {path}")
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


def black_hat_disk(gray: np.ndarray, radius: int) -> np.ndarray:
    """Chapéu-preto (elemento estruturante discoidal / elipse), como no artigo."""
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def remove_small_objects(mask_bool: np.ndarray, min_size: int) -> np.ndarray:
    if not np.any(mask_bool):
        return mask_bool
    u8 = mask_bool.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    out = np.zeros_like(mask_bool, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out |= labels == i
    return out


def largest_component(mask_bool: np.ndarray) -> np.ndarray:
    if not np.any(mask_bool):
        return mask_bool
    mask_u8 = mask_bool.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_bool
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


def score_track_shape(mask_bool: np.ndarray) -> float:
    """Quanto maior, mais 'fio' longo (pista); blocos compactos (arquibancada) têm valor menor."""
    if not np.any(mask_bool):
        return -1.0
    u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 100:
        return -1.0
    peri = cv2.arcLength(cnt, True)
    return (peri * peri) / area


def select_track_component(mask_bool: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Entre componentes com área plausível, escolhe o mais parecido com pista (perímetro²/área alto).
    Evita ficar só na arquibancada (blob compacto com maior área).
    """
    u8 = mask_bool.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    img_area = h * w
    min_a = max(800, int(0.0035 * img_area))
    max_a = int(0.48 * img_area)
    best_lab = -1
    best_sc = -1.0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_a <= area <= max_a):
            continue
        comp = labels == i
        sc = score_track_shape(comp)
        if sc > best_sc:
            best_sc = sc
            best_lab = i
    if best_lab < 0:
        return largest_component(mask_bool)
    return labels == best_lab


def morph_pipeline(thresh_bool: np.ndarray, h: int, w: int, bh_radius: int) -> np.ndarray:
    """Mesma cadeia morfológica do artigo (área → dilata/erode cruz → área → fechamento)."""
    min_area_1 = max(500, int(h * w * 0.00012))
    areaclose1 = remove_small_objects(thresh_bool, min_size=min_area_1)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    u = areaclose1.astype(np.uint8) * 255
    u = cv2.dilate(u, kernel_cross, iterations=2)
    u = cv2.erode(u, kernel_cross, iterations=1)
    eroded_bool = u > 0
    min_area_2 = max(100, int(h * w * 0.00005))
    final_bool = remove_small_objects(eroded_bool, min_size=min_area_2)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, bh_radius // 2) * 2 + 1,) * 2)
    closed_u8 = cv2.morphologyEx(final_bool.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_k, iterations=2)
    return closed_u8 > 0


def search_percentile_for_track(
    histeq: np.ndarray, h: int, w: int, bh_radius: int
) -> tuple[np.ndarray, float, str]:
    """
    Testa vários percentis (máscara mais larga que antes): inclui a pista inteira no candidato.
    Escolhe o percentil que maximiza o 'formato de pista' após a morfologia.
    """
    best_mask: np.ndarray | None = None
    best_score = -1.0
    best_p = 70
    for p in range(62, 93, 2):
        t = float(np.percentile(histeq, p))
        tb = histeq >= t
        m = morph_pipeline(tb, h, w, bh_radius)
        cand = select_track_component(m, h, w)
        if not np.any(cand):
            continue
        sc = score_track_shape(cand)
        if sc > best_score:
            best_score = sc
            best_mask = cand
            best_p = p
    if best_mask is None:
        tb, val, mode = threshold_track_mask(histeq)
        m = morph_pipeline(tb, h, w, bh_radius)
        best_mask = select_track_component(m, h, w)
        return best_mask, val, f"fallback {mode}"
    t_best = float(np.percentile(histeq, best_p))
    return best_mask, t_best, f"percentil {best_p}% (score forma={best_score:.0f})"


def threshold_track_mask(histeq: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Otsu em `histeq` costuma marcar ~90% da imagem: o fundo vira uma classe enorme.
    Escolhe entre limiar direto/inverso pela fração de pixels (~pista = minoria, tipicamente 2–20%).
    Se ambos falharem, usa percentil alto (espírito do limiar ~200 do artigo).
    """
    otsu_val, _ = cv2.threshold(histeq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = histeq > otsu_val
    lo = histeq <= otsu_val
    fh, fl = float(np.mean(hi)), float(np.mean(lo))

    def quality(frac: float) -> float:
        """0 = ótimo se fração na faixa esperada da pista; penaliza muito fundo ou máscara minúscula."""
        lo_b, hi_b = 0.015, 0.22
        if lo_b <= frac <= hi_b:
            return abs(frac - 0.07)
        if frac < lo_b:
            return 10.0 + (lo_b - frac)
        return 10.0 + (frac - hi_b)

    qh, ql = quality(fh), quality(fl)
    if qh <= ql:
        if fh <= 0.45:
            return hi, float(otsu_val), f"Otsu BINARY (frac={fh:.1%})"
        chosen = lo
        tag = f"Otsu→INV pois BINARY cobria {fh:.1%}"
    else:
        if fl <= 0.45:
            return lo, float(otsu_val), f"Otsu INV (frac={fl:.1%})"
        chosen = hi
        tag = f"Otsu→BINARY pois INV cobria {fl:.1%}"

    fc = float(np.mean(chosen))
    if fc <= 0.45:
        return chosen, float(otsu_val), tag

    # Ambos cobrem quase a imagem: cauda superior do histograma (pista realça clara)
    for p in (93, 91, 89, 87, 85, 83, 80, 78, 75):
        t = float(np.percentile(histeq, p))
        m = histeq >= t
        f = float(np.mean(m))
        if 0.012 <= f <= 0.28:
            return m, t, f"Percentil {p}% (frac={f:.1%})"

    t = float(np.percentile(histeq, 90))
    return histeq >= t, t, f"Percentil 90% fallback (frac={float(np.mean(histeq >= t)):.1%})"


def refine_track_outline(mask_bool: np.ndarray) -> np.ndarray:
    """
    Suaviza o contorno da maior região: remove serrilhado do limiar mantendo o formato do circuito.
    """
    u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask_bool
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.0012 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    out = np.zeros_like(u8)
    cv2.drawContours(out, [approx], -1, 255, thickness=-1)
    return out > 0


# %% [markdown]
# ## 1. Obtenção da imagem e conversão para tons de cinza
#
# Toda a detecção usa apenas a **imagem em escala de cinza** (como no roteiro do artigo).

# %%
IMAGE_PATH = find_satellite_jpg()
print(f"Usando imagem: {IMAGE_PATH}")

img_bgr = imread_bgr(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

h, w = img_bgr.shape[:2]
print(f"Resolução: {w}×{h}")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
show_img("Imagem original (satélite)", img_bgr)
show_img("Imagem em tons de cinza (entrada do pipeline)", img_gray, cmap="gray")

# %% [markdown]
# ## 2. Pré-processamento
#
# 1. Suavização Gaussiana (reduz ruído de alta frequência, telhados “quebram” menos o chapéu-preto).
# 2. **Chapéu-preto** (`MORPH_BLACKHAT`, elemento em disco): realça feições mais escuras que o entorno (faixa de asfalto).
# 3. **Equalização de histograma** (`equalizeHist`), como no artigo — espalha o contraste para facilitar o limiar.

# %%
# Raio do disco: artigo usa ~5; em imagens grandes, escala com a resolução (largura da pista em pixels)
bh_radius = max(5, min(h, w) // 60)
kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bh_radius + 1, 2 * bh_radius + 1))

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
closeth = cv2.morphologyEx(img_blur, cv2.MORPH_BLACKHAT, kernel_disk)
histeq = cv2.equalizeHist(closeth)

show_img(f"Após chapéu-preto (disco ≈ raio {bh_radius})", closeth, cmap="gray")
show_img("Após equalização de histograma (subm)", histeq, cmap="gray")

# %% [markdown]
# ## 3. Limiarização e pós-processamento morfológico
#
# 1. **Busca em percentis** (62–92%) em `histeq`: máscara mais ampla que a regra de “só 7% da imagem”, para incluir a faixa de asfalto.
# 2. Para cada percentil: **remove_small_objects** → **dilata/erode cruz** → **remove_small_objects** → **fechamento** (como no artigo).
# 3. **Seleção do circuito** pelo maior **perímetro²/área** entre componentes de área plausível — a pista é um anel longo; arquibancada é compacta.
# 4. Escolhe o percentil que maximiza esse “score de forma”.
# 5. **Refino do contorno** (`approxPolyDP`).

# %%
mask_bool, thr_used, thr_mode = search_percentile_for_track(histeq, h, w, bh_radius)
mask_bool = refine_track_outline(mask_bool)

final_feat = mask_bool.astype(np.uint8) * 255

print(f"Limiar: {thr_mode} | valor≈{thr_used:.1f} | disco raio={bh_radius}")
show_img("Feição detectada (máscara da pista)", final_feat, cmap="gray")

# %% [markdown]
# ## 4. Resultado — sobreposição em vermelho

# %%
overlay = img_rgb.copy().astype(np.float32)
overlay[mask_bool] = [255, 0, 0]
alpha = 0.55
result_rgb = (alpha * overlay + (1.0 - alpha) * img_rgb.astype(np.float32)).astype(np.uint8)

plt.figure(figsize=(14, 12))
plt.title("Sobreposição da pista detectada (vermelho)")
plt.imshow(result_rgb)
plt.axis("off")
plt.show()
