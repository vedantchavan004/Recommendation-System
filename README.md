# Recommendation System

This repository implements a recommendation system using a Graph Neural Network (GNN) based on a simplified [LightGCN](https://arxiv.org/abs/2002.02126) model. The system is trained on the **MovieLens 100K** dataset and uses Bayesian Personalized Ranking (BPR) loss for training.

## 📂 Repository Structure

```
LightGCN-Recommendation/
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── data/                   # Directory for downloaded data (created at runtime)
├── notebooks/
│   └── LightGCN_Recommender.ipynb   # Optional Colab/Jupyter notebook with full pipeline
└── src/
    ├── data.py             # Data download & processing
    ├── model.py            # LightGCN model definition
    ├── train.py            # Training script using BPR loss
    └── recommend.py        # Script for generating recommendations
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/LightGCN-Recommendation.git
cd LightGCN-Recommendation
```

### 2️⃣ Set Up a Virtual Environment

Create and activate a virtual environment:

- **On Windows:**
  ```bash
  python -m venv env
  env\Scripts\activate
  ```

- **On macOS/Linux:**
  ```bash
  python -m venv env
  source env/bin/activate
  ```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏗 Running the Project

### ✅ a. Download & Process Data

The dataset will be automatically downloaded and extracted into the `data/` folder. Run:

```bash
python src/data.py
```

This script will:
- Download the **MovieLens 100K** dataset.
- Extract and preprocess user-item interactions.
- Convert it into a graph-friendly format.

### 🏋️ b. Train the Model

To train the **LightGCN** model, run:

```bash
python src/train.py
```

This script:
- Loads the processed dataset.
- Constructs a **normalized adjacency matrix**.
- Trains the model using **Bayesian Personalized Ranking (BPR) loss**.
- Saves the trained model as `model.pth`.

### 🎯 c. Generate Recommendations

Once the model is trained, generate recommendations for a specific user by running:

```bash
python src/recommend.py
```

This script:
- Loads the trained **LightGCN** model.
- Computes user and item embeddings.
- Prints the **Top 10 recommended items** for a sample user.

---

## 🧪 Jupyter Notebook (Optional)

A **Jupyter/Colab Notebook** is available in the `notebooks/` folder. Open **`LightGCN_Recommender.ipynb`** to run the full pipeline interactively.

To launch Jupyter:
```bash
jupyter notebook
```

---

## 📦 Dependencies & Requirements

This project requires **Python 3.7+** and the following libraries:

```txt
torch==2.0.1
torchvision==0.15.2
torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
torch-geometric
pandas
```

If using **CPU only**, replace `+cu117` with `+cpu`.

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments

- **[LightGCN Paper](https://arxiv.org/abs/2002.02126)** – Model inspiration.
- **[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)** – Dataset source.
- **PyTorch Geometric** – Graph learning framework.

---

### 🎯 **Next Steps**
✔️ Train on **larger datasets** (MovieLens 1M, Amazon Reviews, etc.).  
✔️ Implement **hyperparameter tuning**.  
✔️ Experiment with **Graph Attention Networks (GAT)**.  

---


