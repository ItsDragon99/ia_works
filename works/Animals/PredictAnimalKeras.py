import os
import json
import csv
import math
import cv2
import numpy as np
import tensorflow as tf

# MODEL_PATH         = "../models/animals/animal_cnn_keras_best.h5"
MODEL_PATH= "../models/animalsv2/animals_best_model.h5"

CLASS_IDXS_PATH = "../models/animals/class_indices.json"
IMG_SIZE = (96, 96)
INPUT_PATH = "../dataset/test"
SAVE_CSV_BATCH = False
COLLAGE_COLS = 5
TILE_SIZE = (240, 240)  
TILE_PADDING = 8
TEXT_BAR_PX = 56
MAX_DISPLAY_PX = 1400  

def load_class_names():
    if os.path.exists(CLASS_IDXS_PATH):
        with open(CLASS_IDXS_PATH, "r") as f:
            class_indices = json.load(f)
        
        
        classes = [None] * len(class_indices)
        for name, idx in class_indices.items():
            classes[idx] = name
        return classes
    else:
        print("No se encontró class_indices.json, usando DEFAULT_CLASSES.")
        return [
            "cat",
            "dog",
            "turtle",
            "ant",
            "ladybug"
        ]

def preprocess_image(img_path, target_size):
    """
    Lee y preprocesa una imagen para el modelo:
      - BGR -> RGB (porque OpenCV usa BGR)
      - resize
      - normalización [0, 1]
      - añade dimensión batch
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_resized = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)
    return img, img_batch  


def predict_image(model, img_batch, class_names, top_k=3):
    """
    Realiza la predicción y devuelve:
      - lista de (idx, nombre_clase, prob) ordenada desc por prob
    """
    pred = model.predict(img_batch, verbose=0)[0]  
    sorted_indices = np.argsort(pred)[::-1][:top_k]
    results = []
    for idx in sorted_indices:
        class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        prob = float(pred[idx])
        results.append((idx, class_name, prob))
    return results


def draw_prediction_on_image(img_bgr, best_class_name, best_prob):
    """
    Escribe el resultado principal sobre la imagen.
    """
    text = f"{best_class_name} ({best_prob*100:.1f}%)"
    cv2.putText(
        img_bgr,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    return img_bgr


def iter_image_files(dir_path):
    """
    Itera recursivamente sobre imágenes en un directorio.
    Extensiones soportadas: .jpg, .jpeg, .png, .bmp
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for root, _, files in os.walk(dir_path):
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext in exts:
                yield os.path.join(root, fname)


def _fit_into_box_keep_ar(img_bgr, box_w, box_h):
    """Redimensiona con letterbox para conservar aspecto dentro de box_w x box_h."""
    h, w = img_bgr.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y0 = (box_h - new_h) // 2
    x0 = (box_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def _draw_text_bar(tile, title, subtitle, bar_h, pad=6):
    """Dibuja una barra inferior con título y subtítulo en el tile."""
    h, w = tile.shape[:2]
    y0 = h - bar_h
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    tile = cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.5
    font_scale_sub = 0.45
    thickness = 1
    max_chars = max(8, int(w / 9))
    title_show = title[:max_chars]
    subtitle_show = subtitle[:max_chars]

    cv2.putText(tile, title_show, (pad, y0 + int(bar_h*0.55)), font, font_scale_title, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(tile, subtitle_show, (pad, y0 + bar_h - pad), font, font_scale_sub, (180, 220, 180), thickness, cv2.LINE_AA)
    return tile


def make_tile(img_bgr, title, subtitle, tile_size=TILE_SIZE, text_bar_px=TEXT_BAR_PX):
    tw, th = tile_size
    img_h = max(1, th - text_bar_px)
    
    pane = _fit_into_box_keep_ar(img_bgr, tw, img_h)
    
    tile = np.zeros((th, tw, 3), dtype=np.uint8)
    tile[:img_h, :, :] = pane
    tile = _draw_text_bar(tile, title, subtitle, text_bar_px)
    return tile


def make_collage(tiles, cols=COLLAGE_COLS, pad=TILE_PADDING, bg=(30, 30, 30)):
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    n = len(tiles)
    rows = math.ceil(n / cols)
    H = rows * th + (rows + 1) * pad
    W = cols * tw + (cols + 1) * pad
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        canvas[y:y+th, x:x+tw] = tile
    return canvas


def predict_directory(model, dir_path, class_names, top_k=3):
    """
    Predice todas las imágenes en un directorio y muestra top-k por archivo.
    Opcionalmente guarda un CSV con resultados: ruta, clase, probabilidad, top-k JSON.
    """
    rows = []
    count = 0
    tiles = []
    for img_path in iter_image_files(dir_path):
        try:
            img_bgr_for_tile = cv2.imread(img_path)  
            if img_bgr_for_tile is None:
                raise ValueError("No se pudo leer la imagen en BGR")
            _, img_batch = preprocess_image(img_path, IMG_SIZE)
            results = predict_image(model, img_batch, class_names, top_k=top_k)
            best_idx, best_name, best_prob = results[0]
            print(f"\nArchivo: {img_path}")
            for rank, (_, name, prob) in enumerate(results, start=1):
                print(f"  {rank}. {name} -> {prob*100:.2f}%")
            rows.append({
                "path": img_path,
                "top1_class": best_name,
                "top1_prob": f"{best_prob:.6f}",
                "topk": json.dumps([
                    {"rank": i+1, "class": n, "prob": float(p)} for i, (_, n, p) in enumerate(results)
                ], ensure_ascii=False)
            })
            count += 1
            title = os.path.basename(img_path)
            subtitle = f"{best_name} ({best_prob*100:.1f}%)"
            tile = make_tile(img_bgr_for_tile, title, subtitle)
            tiles.append(tile)
        except Exception as e:
            print(f"Error con {img_path}: {e}")    
    if tiles:
        collage = make_collage(tiles)
        if collage is not None:
            H, W = collage.shape[:2]
            scale = min(1.0, MAX_DISPLAY_PX / max(H, W))
            if scale < 1.0:
                collage = cv2.resize(collage, (int(W*scale), int(H*scale)))
            cv2.imshow("Collage de Predicciones", collage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\nTotal de imágenes procesadas: {count}")

if __name__ == "__main__":    
    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Modelo cargado.")
    class_names = load_class_names()
    print("Clases:", class_names)
    input_path = INPUT_PATH  
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"La ruta no existe: {input_path}")
    if os.path.isdir(input_path):
        print(f"Procesando carpeta: {input_path}")
        predict_directory(
            model,
            input_path,
            class_names,
            top_k=3,
        )
    else:
        try:
            img_bgr, img_batch = preprocess_image(input_path, IMG_SIZE)
        except ValueError as e:
            print(e)
            exit(1)
        results = predict_image(model, img_batch, class_names, top_k=3)
        print("\nPredicciones (top-3):")
        for rank, (idx, name, prob) in enumerate(results, start=1):
            print(f"{rank}. {name} -> {prob*100:.2f}%")

        best_idx, best_name, best_prob = results[0]
        img_out = draw_prediction_on_image(img_bgr, best_name, best_prob)

        cv2.imshow("Clasificador de Animales", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
