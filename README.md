# Paddy Leaf Health Checker (Tested and run on Raspberry Pi 5 using Raspberry Pi os 64-bit)

## IMPORTANT - Notes for visitors

NOTE 1/3: AI assistance was used for coding, debugging, documentation and drafting the initial proof-of-concept knowledge base for this project.

NOTE 2/3: This repository was created for a hackathon submission.

**NOTE 3/3: AI tools were used to rapidly build this feasibility prototype and prove the concept works, rather than to create a finished product. A clean, stable version written entirely by me will follow in a future release.**

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
`cd ~/paddy_leaf_health_checker`
`source paddy_app/bin/activate`
`python app.py`
It will run the server.

On a web browser, open the following link: `http://127.0.0.1:5000`
