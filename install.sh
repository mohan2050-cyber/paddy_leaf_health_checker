#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HOME}/paddy_leaf_health_checker"
VENV_NAME="paddy_app"
RAW_BASE="https://raw.githubusercontent.com/mohan2050-cyber/paddy_leaf_health_checker/main"
RELEASE_BASE="https://github.com/mohan2050-cyber/paddy_leaf_health_checker/releases/download/v0.1.0"

echo "==> Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y curl unzip python3-venv python3-pip

echo "==> Creating project directory structure at ${APP_DIR}..."
mkdir -p "${APP_DIR}/static" "${APP_DIR}/templates"
cd "${APP_DIR}"

echo "==> Downloading application files from GitHub (main)..."
for file in app.py class_names.json knowledge_base.py paddy_regional_advisories_india.json translator.py; do
    echo "    -> ${file}"
    curl -4 -fL "${RAW_BASE}/${file}" -o "${file}"
done

echo "==> Downloading static assets..."
curl -4 -fL "${RAW_BASE}/app.js" -o "static/app.js"
curl -4 -fL "${RAW_BASE}/style.css" -o "static/style.css"

echo "==> Downloading HTML template..."
curl -4 -fL "${RAW_BASE}/index.html" -o "templates/index.html"

echo "==> Downloading model files from Release v0.1.0..."
echo "    -> efficientnet_b4_paddy.pte"
curl -4 -fL --progress-bar "${RELEASE_BASE}/efficientnet_b4_paddy.pte" -o "efficientnet_b4_paddy.pte"

echo "    -> llama32_1b_et.zip"
curl -4 -fL --progress-bar "${RELEASE_BASE}/llama32_1b_et.zip" -o "llama32_1b_et.zip"

echo "==> Extracting llama32_1b_et.zip..."
rm -rf llama32_1b_et llama32_1b_et_tmp
unzip -q llama32_1b_et.zip -d llama32_1b_et_tmp
if [ -d "llama32_1b_et_tmp/llama32_1b_et" ]; then
    mv llama32_1b_et_tmp/llama32_1b_et ./llama32_1b_et
    rm -rf llama32_1b_et_tmp
else
    mv llama32_1b_et_tmp ./llama32_1b_et
fi
rm -f llama32_1b_et.zip

echo "==> Creating Python virtual environment '${VENV_NAME}'..."
python3 -m venv "${VENV_NAME}"
source "${VENV_NAME}/bin/activate"

echo "==> Installing Python packages..."
pip install --retries 10 --timeout 120 --upgrade pip setuptools wheel
pip install \
    --resume-retries 10 \
    --timeout 120 \
    --retries 10 \
    flask \
    numpy \
    pillow \
    torch \
    executorch \
    transformers \
    optimum-executorch \
    argostranslate

echo "==> Installing Argos Translate language models..."
argospm update
argospm install translate-en_bn
argospm install translate-bn_en
argospm install translate-en_hi
argospm install translate-hi_en

echo "==> Warming up Argos Translate models for offline use..."
python - << 'EOF'
import socket
import urllib3.util.connection as urllib3_cn
import argostranslate.translate

urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

test_pairs = [
    ("en", "bn", "Hello"),
    ("bn", "en", "হ্যালো"),
    ("en", "hi", "Hello"),
    ("hi", "en", "नमस्ते"),
]

for from_code, to_code, sample in test_pairs:
    translated = argostranslate.translate.translate(sample, from_code, to_code)
    assert translated and translated.strip(), f"Translation failed for {from_code} -> {to_code}"
    print(f"    -> Verified {from_code} -> {to_code}: {translated.strip()}")
EOF

echo ""
echo "==> Setup complete!"
echo "Run the app with:"
echo "  cd ${APP_DIR}"
echo "  source ${VENV_NAME}/bin/activate"
echo "  python app.py"
