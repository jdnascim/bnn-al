# Interactive Event Sifting using Bayesian Graph Neural Networks

This repository contains the official implementation of the paper:

**Interactive Event Sifting using Bayesian Graph Neural Networks**  
*José Nascimento, Nathan Jacobs, Anderson Rocha*  
Published in the **2024 IEEE International Workshop on Information Forensics and Security (WIFS)**  
[📄 IEEE Xplore Link](https://ieeexplore.ieee.org/document/10810718)  

---

## 📜 Abstract
Forensic analysts often use social media imagery
and texts to understand important events. A primary challenge
is the initial sifting of irrelevant posts. This work introduces
an interactive process for training an event-centric, learningbased multimodal classification model that automates sanitization. We propose a method based on Bayesian Graph Neural
Networks (BGNNs) and evaluate active learning and pseudolabeling formulations to reduce the number of posts the analyst
must manually annotate. Our results indicate that BGNNs are
useful for social-media data sifting for forensics investigations of
events of interest, the value of active learning and pseudo-labeling
varies based on the setting, and incorporating unlabelled data
from other events improves performance

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jdnascim/bnn-al
cd bnn-al
```

### 2️⃣ Set Up the Environment
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 📂 Dataset Preparation
Download and process the dataset splits by running:

```bash
bash process.sh
```

This script automates downloading the datasets and preparing them for training and evaluation.

### 🚀 Running Experiments
Available Scripts
You can explore two main directories for experiments:

**scripts/ablation**: Scripts for ablation studies.

**scripts/diff_models**: Scripts to test different models.

Example Execution:
To run an experiment, use the following command:

```bash
bash exp_2.sh 18 0 1 mexico_earthquake
```

**Explanation of Parameters**

**2**: Experiment number (e.g., exp_2.sh).

**18**: Initial size of the labeled set.

**0**: Run ID (useful for running the same experiment multiple times).

**1**: GPU ID for execution.

**mexico_earthquake**: Dataset name to use for the experiment.

### 🧪 Results

The results of our experiments highlight the efficacy of our Bayesian Graph Neural Network framework. For detailed results and analysis, please refer to the paper.

### 📚 Citation

If you use this code or our work in your research, please consider citing:

```bibtex
@INPROCEEDINGS{10810718,
  author={Nascimento, José and Jacobs, Nathan and Rocha, Anderson},
  booktitle={2024 IEEE International Workshop on Information Forensics and Security (WIFS)}, 
  title={Interactive Event Sifting using Bayesian Graph Neural Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Social networking (online);Forensics;Conferences;Active learning;Graph neural networks;Bayes methods;Security;Bayesian Graph Neural Networks;forensic event analysis;human-in-the-loop;few-shot learning},
  doi={10.1109/WIFS61860.2024.10810718}
  }
```

### 🤝 Acknowledgments

We thank the McDonnell International Scholars Academy
at Washington University in St. Louis and the Sao˜
Paulo Research Foundation (FAPESP) Horus project (Grant
#2023/12865-8) for supporting this work.

### 🛡️ License

This repository is released under the MIT License.

### 📬 Contact

For any inquiries or questions, please contact:

José Nascimento: jose.nascimento@ic.unicamp.br