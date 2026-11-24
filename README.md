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

<img width="366" height="343" alt="image" src="https://github.com/user-attachments/assets/7cc91c78-24df-467c-ac27-118bfe3cc3fa" />

Once the files have been downloaded and the packages have been installed, run `run.sh` to run the server.

On a web browser, open the following link: `http://127.0.0.1:5000`
