# Petal-FL 🌸
*A Modular Federated Learning Framework for Experimenting With Client Heterogeneity*

---

## 📖 Overview

Petal-FL is a federated learning framework built with PyTorch and gRPC that simulates multiple clients training models locally and a central server aggregating their updates asynchronously using FedAvg. The system is fully containerized with Docker for easy deployment and scalability.

---

## 📚 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- Distributed federated learning simulation with multiple clients
- Asynchronous FedAvg aggregation of models
- Modular design to swap out model architectures easily
- gRPC-based communication for scalability and efficiency
- Docker Compose support for easy multi-container deployment

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- `pip` package manager

### Installation

```bash
git clone https://github.com/ethanchilds/Petal-FL.git
cd Petal-FL
pip install -r requirements.txt
