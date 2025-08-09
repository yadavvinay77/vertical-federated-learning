
# Vertical Federated Learning Demo

This repository demonstrates a basic vertical federated learning (VFL) setup using the UCI Adult Census dataset. The dataset is split vertically between two clients, each holding a distinct set of features. A simple neural network model is trained collaboratively without sharing raw data, preserving data privacy.

## Features

- Vertical data splitting of the UCI Adult dataset into two clients
- Simple client-server model architecture using numpy
- Training with forward and backward passes distributed between clients and server
- Model evaluation with accuracy and loss metrics
- Save/load model parameters functionality
- Training loss visualization
- CI/CD workflow for automated data preparation and training demo

## Project Structure

```
vfl_project/
├── clients/               # Client-side model code
├── coordinator/           # Server-side (coordinator) model code
├── data/                  # Dataset files and preprocessing script
├── encryption/            # Mock/homomorphic encryption modules (optional)
├── models/                # Model implementations (ClientModel, ServerModel)
├── tests/                 # Unit and integration tests
├── utils/                 # Helper utilities (data loading, config)
├── run_all.py             # Script to run the full VFL training demo
├── requirements.txt       # Python dependencies
└── .github/workflows/     # GitHub Actions CI/CD workflow
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yadavvinay77/vertical-federated-learning.git
cd vertical-federated-learning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data:

```bash
python data/prepare_data.py
```

4. Run the VFL training demo:

```bash
python run_all.py
```

5. (Optional) View training loss plot and saved model files.

## Usage

The demo trains two client models and one server model collaboratively on vertically split data. The client models process their respective feature sets and send intermediate results to the server model for final prediction.

## Contributing

Contributions are welcome! Please fork the repo and open pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Contact

Vinaykumar Yadav  
[yadavvinay77](https://github.com/yadavvinay77)  
Email: yadavvinay77@gmail.com
