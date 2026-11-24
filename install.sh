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
    curl -fL "${RAW_BASE}/${file}" -o "${file}"
done

echo "==> Downloading static assets..."
curl -fL "${RAW_BASE}/static/app.js" -o "static/app.js"
curl -fL "${RAW_BASE}/static/style.css" -o "static/style.css"

echo "==> Downloading HTML template..."
curl -fL "${RAW_BASE}/templates/index.html" -o "templates/index.html"

echo "==> Downloading model files from Release v0.1.0..."
echo "    -> efficientnet_b4_paddy.pte"
curl -fL --progress-bar "${RELEASE_BASE}/efficientnet_b4_paddy.pte" -o "efficientnet_b4_paddy.pte"

echo "    -> llama32_1b_et.zip"
curl -fL --progress-bar "${RELEASE_BASE}/llama32_1b_et.zip" -o "llama32_1b_et.zip"

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

echo ""
echo "==> Setup complete!"
echo "Run the app with:"
echo "  cd ${APP_DIR}"
echo "  source ${VENV_NAME}/bin/activate"
echo "  python app.py"
