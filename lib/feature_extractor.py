import numpy as np
from PIL import ImageFilter

'''
This module serves as the single source of truth for calculating visual image features.
It extracts numerical characteristics that allow the model to classify waste based on 
texture, color, and brightness without needing to process raw images in real-time.
'''

def extract_features(img, img_size=128):
    '''
    Performs a comprehensive image analysis and returns a dictionary of numerical metrics.

    :params img: Input image object in RGB mode.
    :params img_size: Target resolution for calculation normalization (default 128x128).
    :return: A dictionary containing 15 visual features rounded for data stability.
    '''

    img_resized = img.resize((img_size, img_size))
    arr = np.asarray(img_resized).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

    gray = img_resized.convert("L")
    edge_arr = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype("float32")
    detail_arr = np.asarray(gray.filter(ImageFilter.DETAIL)).astype("float32")
    emboss_arr = np.asarray(gray.filter(ImageFilter.EMBOSS)).astype("float32")

    hist, _ = np.histogram(brightness.flatten(), bins=64, range=(0, 255))
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    ehist, _ = np.histogram(edge_arr.flatten(), bins=32, range=(0, 255))
    ehist = ehist / (ehist.sum() + 1e-8)
    edge_entropy = -np.sum(ehist[ehist > 0] * np.log2(ehist[ehist > 0]))

    return {
        "brightness": round(float(brightness.mean()), 2),
        "contrast": round(float(brightness.std()), 2),
        "saturation": round(float(saturation.mean()), 4),
        "color_uniformity": round(float(saturation.std()), 4),
        "warm_ratio": round(float((r > b + 15).mean()), 4),
        "transparency": round(float((brightness > 210).mean()), 4),
        "dark_ratio": round(float((brightness < 40).mean()), 4),
        "edge_density": round(float(edge_arr.mean()), 2),
        "edge_intensity": round(float(edge_arr.std()), 2),
        "texture_roughness": round(float(detail_arr.std()), 2),
        "smoothness": round(float(emboss_arr.std()), 2),
        "entropy": round(float(entropy), 4),
        "edge_entropy": round(float(edge_entropy), 4),
        "channel_variance": round(float(np.var([r.mean(), g.mean(), b.mean()])), 2),
        "highlights": round(float((brightness > 240).mean()), 4),
    }