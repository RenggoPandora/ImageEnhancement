import os
import uuid
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import cv2
import numpy as np
import shutil
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
RESULT_DIR = BASE_DIR / 'static' / 'results'
DATASET_DIR = BASE_DIR / 'LOLv2-real'

for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
app.secret_key = 'dev-secret'


def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXT


def read_image_from_bytes(buf):
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def calculate_channel_ssim(mu1, mu2, C1, C2):
    window_size = 11
    sigma = 1.5
    
    mu1_blur = cv2.GaussianBlur(mu1, (window_size, window_size), sigma)
    mu2_blur = cv2.GaussianBlur(mu2, (window_size, window_size), sigma)
    
    mu1_sq = mu1_blur ** 2
    mu2_sq = mu2_blur ** 2
    mu1_mu2 = mu1_blur * mu2_blur
    
    sigma1_sq = cv2.GaussianBlur(mu1 ** 2, (window_size, window_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(mu2 ** 2, (window_size, window_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(mu1 * mu2, (window_size, window_size), sigma) - mu1_mu2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if len(img1.shape) == 3:
        ssims = []
        for i in range(3):
            ssims.append(calculate_channel_ssim(img1[:,:,i], img2[:,:,i], C1, C2))
        return float(np.mean(ssims))
    else:
        return float(calculate_channel_ssim(img1, img2, C1, C2))


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def enhancement_pipeline(img, params=None):
    # Adaptive classical enhancement pipeline (brightness-aware)
    def grayworld_white_balance(bgr):
        bgr = bgr.astype(np.float32)
        avg_b = np.mean(bgr[:, :, 0])
        avg_g = np.mean(bgr[:, :, 1])
        avg_r = np.mean(bgr[:, :, 2])
        avg = (avg_b + avg_g + avg_r) / 3.0
        avg_b = max(avg_b, 1.0)
        avg_g = max(avg_g, 1.0)
        avg_r = max(avg_r, 1.0)
        bgr[:, :, 0] = np.clip(bgr[:, :, 0] * (avg / avg_b), 0, 255)
        bgr[:, :, 1] = np.clip(bgr[:, :, 1] * (avg / avg_g), 0, 255)
        bgr[:, :, 2] = np.clip(bgr[:, :, 2] * (avg / avg_r), 0, 255)
        return bgr.astype(np.uint8)

    def stretch_luminance(channel, low_pct=2.0, high_pct=98.0):
        low_val, high_val = np.percentile(channel, [low_pct, high_pct])
        if high_val <= low_val + 1e-6:
            return channel
        stretched = (channel.astype(np.float32) - low_val) * 255.0 / (high_val - low_val)
        return np.clip(stretched, 0, 255).astype(np.uint8)

    img = img.copy()
    apply_wb = False  # default to False since Grayworld desaturates low-light images
    if params:
        try:
            if 'apply_wb' in params:
                apply_wb = bool(int(params.get('apply_wb')))
        except Exception:
            pass

    if apply_wb:
        wb = grayworld_white_balance(img)
    else:
        wb = img

    # Estimate brightness and focus
    lab_tmp = cv2.cvtColor(wb, cv2.COLOR_BGR2Lab)
    L_tmp = lab_tmp[:, :, 0]
    mean_l = float(np.mean(L_tmp))
    low_light = mean_l < 90

    # This system is only intended to enhance low-light images.
    # If the input is already normal/bright, return it unchanged.
    if not low_light:
        return img

    # Set defaults adaptively based on brightness
    if low_light:
        h_color = 4
        clahe_clip = 3.0
        unsharp_amount = 1.05
        unsharp_blur_sigma = 1.0
        apply_bilateral = True
        gamma_default = 0.75
        saturation_boost = 1.8
    else:
        h_color = 1
        clahe_clip = 1.6
        unsharp_amount = 1.04
        unsharp_blur_sigma = 1.0
        apply_bilateral = False
        gamma_default = 0.92
        saturation_boost = 1.0

    # Override with user-provided params (if any)
    if params:
        try:
            if 'clahe_clip' in params:
                clahe_clip = float(params.get('clahe_clip'))
        except Exception:
            pass
        try:
            if 'denoise_h' in params:
                h_color = float(params.get('denoise_h'))
        except Exception:
            pass
        try:
            if 'unsharp_amount' in params:
                unsharp_amount = float(params.get('unsharp_amount'))
        except Exception:
            pass
        try:
            if 'gamma' in params:
                g = float(params.get('gamma'))
                # gamma <= 0 means auto
                gamma_override = g if g > 0 else None
            else:
                gamma_override = None
        except Exception:
            gamma_override = None
        try:
            if 'apply_bilateral' in params:
                apply_bilateral = bool(int(params.get('apply_bilateral')))
        except Exception:
            pass
        try:
            if 'saturation_boost' in params:
                saturation_boost = float(params.get('saturation_boost'))
        except Exception:
            pass
    else:
        gamma_override = None

    # If the user explicitly changes params, keep the adaptive gamma as fallback only
    gamma = gamma_default

    # Denoise only when needed; avoid softening bright inputs
    if h_color > 0:
        try:
            denoised = cv2.fastNlMeansDenoisingColored(wb, None, h_color, h_color, 7, 21)
        except Exception:
            denoised = cv2.bilateralFilter(wb, d=5, sigmaColor=50, sigmaSpace=50)
    else:
        denoised = wb

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    L_orig, a, b = cv2.split(lab)
    L = L_orig.copy()

    # Low-light mode benefits from luminance stretching before local contrast
    if low_light:
        L = stretch_luminance(L, low_pct=2.0, high_pct=98.5)

    # CLAHE on luminance
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    # Unsharp mask on L channel
    blur = cv2.GaussianBlur(L_clahe, (0, 0), sigmaX=unsharp_blur_sigma)
    L_unsharp = cv2.addWeighted(L_clahe, unsharp_amount, blur, 1 - unsharp_amount, 0)

    # Adaptive gamma (or override)
    if gamma_override is not None:
        gamma = gamma_override
    L_float = L_unsharp.astype(np.float32) / 255.0
    L_gamma = np.power(L_float, gamma)
    L_out = np.clip((L_gamma * 255.0), 0, 255).astype(np.uint8)

    # Proportional chroma scaling to maintain saturation when luminance increases
    if low_light:
        L_safe = np.maximum(L_orig.astype(np.float32), 1.0)
        # Compute ratio of luminance increase
        ratio = L_out.astype(np.float32) / L_safe
        
        # Moderate the ratio to prevent noise blowout (power law + clip)
        ratio = np.clip(np.power(ratio, 0.65), 1.0, 10.0)
        
        # Scale with saturation_boost slider
        chroma_scale = ratio * saturation_boost
        
        # Reshape or apply element-wise
        a_centered = a.astype(np.float32) - 128.0
        b_centered = b.astype(np.float32) - 128.0
        
        a_out = np.clip(a_centered * chroma_scale + 128.0, 0, 255).astype(np.uint8)
        b_out = np.clip(b_centered * chroma_scale + 128.0, 0, 255).astype(np.uint8)
    else:
        a_out = a
        b_out = b

    merged_lab = cv2.merge([L_out, a_out, b_out])
    result = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)

    # Edge-aware sharpening: only strengthen edges to avoid amplifying noise
    # Create sharpened version
    sharpen_strength = 1.08 if low_light else 1.04
    blurred_color = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
    sharpened_color = cv2.addWeighted(result, sharpen_strength, blurred_color, 1.0 - sharpen_strength, 0)

    # Edge mask from Laplacian of luminance
    lap = cv2.Laplacian(L_out, cv2.CV_64F)
    edge_mask = np.absolute(lap)
    # threshold adaptively
    thr = np.percentile(edge_mask, 75)
    mask = (edge_mask > thr).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.expand_dims(np.clip(mask, 0.0, 1.0), axis=2)

    result = (result.astype(np.float32) * (1.0 - mask) + sharpened_color.astype(np.float32) * mask)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mild final smoothing only for very dark images to suppress residual noise
    if apply_bilateral and low_light:
        result = cv2.bilateralFilter(result, d=5, sigmaColor=40, sigmaSpace=40)

    return result


def save_image(path: Path, img):
    # cv2.imwrite expects BGR
    cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))


def cleanup_old_files(folder: Path, max_age_seconds=3600):
    now = time.time()
    for f in folder.glob('*'):
        try:
            if now - f.stat().st_mtime > max_age_seconds:
                f.unlink()
        except Exception:
            pass


@app.route('/')
def index():
    # List example images from dataset (safe limited listing)
    examples = []
    for sub in ('test/low', 'train/low'):
        p = DATASET_DIR / sub
        if p.exists():
            for f in sorted(p.glob('*'))[:20]:
                if f.suffix.lower() in ALLOWED_EXT:
                    examples.append(str(f.relative_to(BASE_DIR)).replace('\\', '/'))
    return render_template('index.html', examples=examples)


@app.route('/upload', methods=['POST'])
def upload():
    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(RESULT_DIR)

    file = request.files.get('file')
    orig_name = request.form.get('orig')

    if (not file or file.filename == '') and not orig_name:
        flash('No file selected')
        return redirect(url_for('index'))

    if file and file.filename != '':
        if not allowed_file(file.filename):
            flash('Unsupported file type')
            return redirect(url_for('index'))

        file.stream.seek(0, os.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_FILE_SIZE:
            flash('File too large (max 5 MB)')
            return redirect(url_for('index'))

        data = file.read()
        img = read_image_from_bytes(data)
        if img is None:
            flash('Failed to read image')
            return redirect(url_for('index'))

        uid = uuid.uuid4().hex
        orig_name = f'{uid}_orig.jpg'
        res_name = f'{uid}_res.jpg'
        orig_path = UPLOAD_DIR / orig_name
        res_path = RESULT_DIR / res_name
        save_image(orig_path, img)
    else:
        orig_path = UPLOAD_DIR / orig_name
        if not orig_path.exists():
            flash('Original image not found')
            return redirect(url_for('index'))
        img = cv2.imread(str(orig_path))
        if img is None:
            flash('Failed to read original image')
            return redirect(url_for('index'))
        uid = orig_name.split('_')[0]
        res_name = f'{uid}_res.jpg'
        res_path = RESULT_DIR / res_name
    # collect tuning params from form
    params = {}
    for key in ('clahe_clip', 'denoise_h', 'unsharp_amount', 'gamma', 'apply_bilateral', 'saturation_boost', 'apply_wb', 'gt_path'):
        if key in request.form:
            params[key] = request.form.get(key)

    out = enhancement_pipeline(img, params=params)
    save_image(res_path, out)

    gt_path_val = params.get('gt_path', '')
    return redirect(url_for('view_result', orig=orig_name, res=res_name, gt_path=gt_path_val))


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """AJAX endpoint: upload original file for interactive preview. Returns orig filename and preview URL placeholder."""
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported type'}), 400

    data = file.read()
    img = read_image_from_bytes(data)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 400

    uid = uuid.uuid4().hex
    orig_name = f'{uid}_orig.jpg'
    orig_path = UPLOAD_DIR / orig_name
    save_image(orig_path, img)

    return jsonify({'orig': orig_name, 'orig_url': f'/static/uploads/{orig_name}'})


@app.route('/start_example', methods=['POST'])
def start_example():
    rel = request.form.get('path')
    if not rel:
        return jsonify({'error': 'No path'}), 400
    full = (BASE_DIR / rel).resolve()
    try:
        if not str(full).startswith(str(DATASET_DIR.resolve())):
            return jsonify({'error': 'Invalid path'}), 400
    except Exception:
        return jsonify({'error': 'Invalid path'}), 400
    if not full.exists() or full.suffix.lower() not in ALLOWED_EXT:
        return jsonify({'error': 'Example not available'}), 400

    uid = uuid.uuid4().hex
    orig_name = f'{uid}_orig.jpg'
    orig_path = UPLOAD_DIR / orig_name
    try:
        shutil.copyfile(str(full), str(orig_path))
    except Exception:
        return jsonify({'error': 'copy failed'}), 500

    # Determine ground truth path
    gt_path_rel = None
    if 'low' in full.name:
        new_name = full.name.replace('low', 'normal')
        gt_path = full.parent.parent / 'high' / new_name
        if gt_path.exists():
            gt_path_rel = str(gt_path.relative_to(BASE_DIR)).replace('\\', '/')

    return jsonify({
        'orig': orig_name, 
        'orig_url': f'/static/uploads/{orig_name}',
        'gt_path': gt_path_rel
    })


@app.route('/preview', methods=['POST'])
def preview():
    data = request.get_json() or {}
    orig = data.get('orig')
    if not orig:
        return jsonify({'error': 'orig missing'}), 400
    orig_path = UPLOAD_DIR / orig
    if not orig_path.exists():
        return jsonify({'error': 'orig not found'}), 404

    # read image
    img = cv2.imread(str(orig_path))
    if img is None:
        return jsonify({'error': 'cannot read orig'}), 500

    params = data.get('params', {})
    out = enhancement_pipeline(img, params=params)

    uid = orig.split('_')[0]
    preview_name = f'{uid}_preview.jpg'
    preview_path = RESULT_DIR / preview_name
    save_image(preview_path, out)

    # Calculate accuracy if gt_path is provided
    gt_path_rel = data.get('gt_path')
    psnr_val = None
    ssim_val = None
    if gt_path_rel:
        gt_full = (BASE_DIR / gt_path_rel).resolve()
        if gt_full.exists():
            gt_img = cv2.imread(str(gt_full))
            if gt_img is not None:
                if gt_img.shape != out.shape:
                    gt_img = cv2.resize(gt_img, (out.shape[1], out.shape[0]))
                psnr_val = calculate_psnr(out, gt_img)
                ssim_val = calculate_ssim(out, gt_img)

    return jsonify({
        'res_url': f'/static/results/{preview_name}?t={int(time.time())}',
        'psnr': psnr_val,
        'ssim': ssim_val
    })


@app.route('/process_example')
def process_example():
    # path is relative to BASE_DIR, but only allow inside dataset dir
    rel = request.args.get('path')
    if not rel:
        flash('No example selected')
        return redirect(url_for('index'))

    full = (BASE_DIR / rel).resolve()
    try:
        if not str(full).startswith(str(DATASET_DIR.resolve())):
            flash('Invalid example path')
            return redirect(url_for('index'))
    except Exception:
        flash('Invalid path')
        return redirect(url_for('index'))

    if not full.exists() or full.suffix.lower() not in ALLOWED_EXT:
        flash('Example not available')
        return redirect(url_for('index'))

    img = cv2.imread(str(full))
    if img is None:
        flash('Failed to read example image')
        return redirect(url_for('index'))

    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(RESULT_DIR)

    uid = uuid.uuid4().hex
    orig_name = f'{uid}_orig.jpg'
    res_name = f'{uid}_res.jpg'
    orig_path = UPLOAD_DIR / orig_name
    res_path = RESULT_DIR / res_name

    save_image(orig_path, img)
    # Determine ground truth path
    gt_path_rel = None
    if 'low' in full.name:
        new_name = full.name.replace('low', 'normal')
        gt_path = full.parent.parent / 'high' / new_name
        if gt_path.exists():
            gt_path_rel = str(gt_path.relative_to(BASE_DIR)).replace('\\', '/')

    # collect tuning params from query string
    params = {}
    for key in ('clahe_clip', 'denoise_h', 'unsharp_amount', 'gamma', 'apply_bilateral', 'saturation_boost', 'apply_wb'):
        if key in request.args:
            params[key] = request.args.get(key)

    out = enhancement_pipeline(img, params=params)
    save_image(res_path, out)

    return redirect(url_for('view_result', orig=orig_name, res=res_name, gt_path=gt_path_rel))


@app.route('/view')
def view_result():
    orig = request.args.get('orig')
    res = request.args.get('res')
    gt_path = request.args.get('gt_path')
    if not orig or not res:
        flash('Result not found')
        return redirect(url_for('index'))

    psnr_val = None
    ssim_val = None
    if gt_path:
        gt_full = (BASE_DIR / gt_path).resolve()
        if gt_full.exists():
            res_img = cv2.imread(str(RESULT_DIR / res))
            gt_img = cv2.imread(str(gt_full))
            if res_img is not None and gt_img is not None:
                if gt_img.shape != res_img.shape:
                    gt_img = cv2.resize(gt_img, (res_img.shape[1], res_img.shape[0]))
                psnr_val = calculate_psnr(res_img, gt_img)
                ssim_val = calculate_ssim(res_img, gt_img)

    return render_template('view.html', orig=orig, res=res, gt_path=gt_path, psnr=psnr_val, ssim=ssim_val)


@app.route('/static/uploads/<path:fname>')
def uploaded_file(fname):
    return send_from_directory(UPLOAD_DIR, fname)


@app.route('/static/results/<path:fname>')
def result_file(fname):
    return send_from_directory(RESULT_DIR, fname)


@app.route('/download/<kind>/<fname>')
def download(kind, fname):
    if kind == 'orig':
        folder = UPLOAD_DIR
    else:
        folder = RESULT_DIR
    return send_from_directory(folder, fname, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
