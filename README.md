# Interactive Event Sifting using Bayesian Graph Neural Networks

This repository contains the official implementation of the paper:

**Interactive Event Sifting using Bayesian Graph Neural Networks**  
*José Nascimento, Nathan Jacobs, Anderson Rocha*  
Published in the **2024 IEEE International Workshop on Information Forensics and Security (WIFS)**  
[📄 IEEE Xplore Link](https://ieeexplore.ieee.org/document/10810718)  
*December 2–5, 2024, Rome, Italy*

---

## 📜 Abstract
This work introduces a novel Bayesian Graph Neural Network (BGNN) framework for interactive event sifting. By leveraging Bayesian inference within a graph-based learning paradigm, the model effectively handles uncertainty in event prediction tasks. The approach was benchmarked on several disaster-related datasets, demonstrating state-of-the-art performance in identifying informative events.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository

### 2️⃣ Set Up the Environment
Install the required dependencies using pip:

```bash
pip install -r requirements.txt

📂 Dataset Preparation
Download and process the dataset splits by running:

```bash
bash process.sh

This script automates downloading the datasets and preparing them for training and evaluation.


markdown
Copiar
Editar
# Interactive Event Sifting using Bayesian Graph Neural Networks

This repository contains the official implementation of the paper:

**Interactive Event Sifting using Bayesian Graph Neural Networks**  
*José Nascimento, Nathan Jacobs, Anderson Rocha*  
Published in the **2024 IEEE International Workshop on Information Forensics and Security (WIFS)**  
[📄 IEEE Xplore Link](https://ieeexplore.ieee.org/document/10810718)  
*December 2–5, 2024, Rome, Italy*

---

## 📜 Abstract
This work introduces a novel Bayesian Graph Neural Network (BGNN) framework for interactive event sifting. By leveraging Bayesian inference within a graph-based learning paradigm, the model effectively handles uncertainty in event prediction tasks. The approach was benchmarked on several disaster-related datasets, demonstrating state-of-the-art performance in identifying informative events.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jdnascim/bnn-al
cd bnn-al

### 2️⃣ Set Up the Environment
Install the required dependencies using pip:

```bash
pip install -r requirements.txt

### 📂 Dataset Preparation
Download and process the dataset splits by running:

```bash
bash process.sh

This script automates downloading the datasets and preparing them for training and evaluation.

###🚀 Running Experiments
Available Scripts
You can explore two main directories for experiments:

scripts/ablation: Scripts for ablation studies.
scripts/diff_models: Scripts to test different models.
Example Execution
To run an experiment, use the following command:

```bash
bash exp_2.sh 18 0 1 mexico_earthquake

Explanation of Parameters
2: Experiment number (e.g., exp_2.sh).
18: Initial size of the labeled set.
0: Run ID (useful for running the same experiment multiple times).
1: GPU ID for execution.
mexico_earthquake: Dataset name to use for the experiment.

###🧪 Results
The results of our experiments highlight the efficacy of our Bayesian Graph Neural Network framework. For detailed results and analysis, please refer to the paper.

###📚 Citation
If you use this code or our work in your research, please consider citing:

```bibtex
@inproceedings{nascimento2024bgnn,
  title={Interactive Event Sifting using Bayesian Graph Neural Networks},
  author={Nascimento, Jos{\'e} and Jacobs, Nathan and Rocha, Anderson},
  booktitle={2024 IEEE International Workshop on Information Forensics and Security (WIFS)},
  year={2024},
  location={Rome, Italy},
  publisher={IEEE},
  doi={10.1109/WIFS2024.10810718}
}

### 🤝 Acknowledgments
This research was supported by [mention any funding or institutions here if applicable].

### 🛡️ License
This repository is released under the MIT License.

### 📬 Contact
For any inquiries or questions, please contact:

José Nascimento: jose.nascimento@ic.unicamp.br