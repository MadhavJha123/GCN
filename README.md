# 📜 Node Classification using Graph Neural Networks (GCN) on Citation Networks

## 📌 Project Overview
This project implements a **Graph Convolutional Network (GCN)** using **PyTorch Geometric** to classify research papers in the **Cora citation dataset**. The model leverages **message passing** to learn meaningful node embeddings and predict paper categories based on citation relationships.

---

## 📂 Dataset: Cora Citation Network
The **Cora dataset** is a commonly used benchmark for node classification tasks. It represents a citation network where:
- **Nodes (2,708)** represent research papers.
- **Edges (5,429)** represent citation links between papers.
- Each node has a **1,433-dimensional feature vector** representing the content of the paper.
- Papers belong to **one of 7 categories** based on their topic.

### 🔹 Dataset Structure:

| Feature | Value |
|---------|-------|
| Nodes   | 2,708 |
| Edges   | 5,429 |
| Features per Node | 1,433 |
| Classes | 7 |
| Graph Type | Undirected |

---

## 🎯 Objective
The goal is to classify each paper into one of **7 categories** using graph-based learning. Traditional machine learning models treat data as tabular, but **GNNs utilize the graph structure**, capturing relationships between nodes through message passing.

---

## 🏗 Model Architecture
The **Graph Convolutional Network (GCN)** consists of:
1. **First Graph Convolution Layer:** Extracts features from neighboring nodes.
2. **ReLU Activation:** Adds non-linearity to improve learning.
3. **Dropout Layer:** Reduces overfitting by randomly disabling neurons.
4. **Second Graph Convolution Layer:** Further refines node embeddings.
5. **Log Softmax Output:** Predicts the probability distribution over 7 classes.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## 🚀 Installation & Setup
### 🔹 Prerequisites
Ensure you have **Python 3.8+** installed. Then, install the necessary dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install matplotlib networkx numpy scikit-learn
```

---

## ⚙️ How to Run the Project
1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-repo/graph-gcn-classification.git
   cd graph-gcn-classification
   ```

2. **Run the GCN model:**
   ```bash
   python gcn_cora.py
   ```

3. **Expected Output:**
   - Training logs with loss reduction over **200 epochs**.
   - **Final test accuracy** (typically **80%+**).

---

## 📊 Evaluation Metrics
The model is evaluated using **classification accuracy**:
- **Accuracy = (Correct Predictions) / (Total Test Samples)**
- Expected accuracy: **~80%**

---

## 🛠 Future Improvements
- Experiment with **Graph Attention Networks (GAT)** for better performance.
- Use **larger graph datasets** like PubMed or Reddit.
- Apply **semi-supervised learning** for better generalization.

---

## 📜 References
- **Original Paper:** [Thomas Kipf & Max Welling (2016)](https://arxiv.org/abs/1609.02907)
- **PyTorch Geometric Documentation:** [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

---

## 🤝 Contributing
Feel free to open issues, suggest improvements, or fork the repository to contribute! 🚀
