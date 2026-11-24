# Paddy Leaf Health Checker (Tested and run on Raspberry Pi 5 using Raspberry Pi OS 64-bit)

## IMPORTANT - Notes for visitors

NOTE 1/3: AI assistance was used for coding, debugging, documentation and drafting the initial proof-of-concept knowledge base for this project.

NOTE 2/3: This repository was created for a hackathon submission.

**NOTE 3/3: AI tools were used to rapidly build this feasibility prototype and prove the concept works, rather than to create a finished product. This project (including code, logic and documentation) may contain errors. A clean, stable version written entirely by me will follow in a future release.**

## External Libraries

* `executorch`
* `torch`
* `optimum-executorch`
* `transformers`
* `argostranslate`
* `flask`
* `numpy`
* `pillow`

## Setup
Run the `install.sh` script on your Raspberry Pi 5 to download the required files and install the required packages.

Make sure the files and folders are arranged in the following manner:

<img width="364" height="291" alt="image" src="https://github.com/user-attachments/assets/a5af8c43-f890-40d7-b2d8-38ee3c348b8f" />


Once the files have been downloaded and the packages have been installed, run the following:
```
cd ~/paddy_leaf_health_checker
source paddy_app/bin/activate
python app.py
```
It will run the server.

On a web browser, open the following link: `http://127.0.0.1:5000`

## Training & Model Conversion Notebook

The Jupyter notebook included in this repository is provided **for reference only** to show the workflow used for model training and ExecuTorch (`.pte`) conversion. If you want to use it for training or conversion, you will need to modify the file paths, dataset locations, and configuration parameters accordingly.

## License & Third-Party Attributions

- **Built with Llama (Meta Llama 3.2):** This project uses Meta Llama 3.2 1B Instruct (`llama32_1b_et`). Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
- **External Libraries, Datasets & References:** All third-party libraries (`executorch`, `torch`, `optimum-executorch`, `transformers`, `argostranslate`, `flask`, `numpy`, `pillow`, etc.), datasets (including the Kaggle *Paddy Doctor: Paddy Disease Classification* dataset), pre-trained base weights, and cited research papers remain the intellectual property of their respective owners and authors, and are subject to their original licenses and terms of use. No ownership is claimed over any third-party materials, and the training dataset is not redistributed in this repository.
