# Math Reasoning Benchmark Experiment

Comprehensive benchmarking of GPT-4o-mini and open-source LLMs on math reasoning tasks.

## 📋 Overview

This project evaluates 15 different language models across 4 math reasoning benchmarks:

### Models Tested
- **GPT-4o-mini** (OpenAI API)
- **Llama-3.1-8B** (Base + Instruct)
- **Llama-4-Scout-17B-16E** (Base + Instruct)
- **Llama-4-Maverick-17B-128E** (Base + Instruct)
- **Qwen-2.5-7B** (Base + Instruct)
- **Qwen-2.5-Math-7B** (Base + Instruct)
- **Qwen-3-4B** (Base + Instruct + Thinking)

### Datasets
- **GSM8K**: Grade school math problems
- **MATH-500**: Competition-level mathematics
- **AIME-25**: American Invitational Mathematics Examination
- **Omni-MATH**: Comprehensive math reasoning benchmark

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

### 3. Run Experiments

```bash
# Run all models on all datasets
python run_benchmark.py --all

# Run specific model on specific dataset
python run_benchmark.py --models gpt-4o-mini --datasets gsm8k

# Run multiple models
python run_benchmark.py --models gpt-4o-mini llama-3.1-8b-instruct --datasets gsm8k math-500
```

### 4. Analyze Results

```bash
# Analyze results and generate visualizations
python analyze_results.py --results results/processed/summary_*.json
```

## 📁 Project Structure

```
math_benchmark_experiment/
├── config.yaml                 # Experiment configuration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
│
├── models/                   # Model loading and inference
│   ├── __init__.py
│   └── model_loader.py
│
├── prompts/                  # Prompt templates
│   ├── __init__.py
│   └── prompt_templates.py
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── answer_parser.py
│   ├── dataset_loader.py
│   └── metrics.py
│
├── evaluators/               # Evaluation logic
│   ├── __init__.py
│   └── base_evaluator.py
│
├── data/                     # Downloaded datasets
│   ├── gsm8k/
│   ├── math500/
│   ├── aime25/
│   └── omni_math/
│
├── results/                  # Experiment results
│   ├── raw/                  # Individual results
│   ├── processed/            # Summary results
│   └── visualizations/       # Plots and charts
│
├── run_benchmark.py          # Main experiment script
└── analyze_results.py        # Results analysis script
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Model parameters (temperature, max_tokens, etc.)
- Dataset sampling (full dataset or sample size)
- Prompt templates
- Evaluation settings
- Output directories

## 📊 Output

### Results Files
- `results/raw/`: Individual model-dataset evaluation JSON files
- `results/processed/summary_*.json`: Aggregated results
- `results/visualizations/`: Generated plots

### Visualizations
- Accuracy heatmap across models and datasets
- Model comparison bar charts
- Latency vs accuracy trade-off plot
- Summary statistics table (CSV)

## 🔧 Advanced Usage

### Custom Prompts

Edit prompt templates in `config.yaml`:

```yaml
prompts:
  base_models:
    template: "Question: {question}\n\nAnswer:"

  instruct_models:
    template: |
      You are a helpful math tutor. Solve the following math problem step by step.

      Problem: {question}

      Solution:
```

### Adding New Models

1. Add model configuration to `config.yaml`:

```yaml
models:
  your-model-name:
    type: "huggingface"
    model_id: "org/model-name"
    dtype: "float16"
    device_map: "auto"
```

2. Run benchmark with new model:

```bash
python run_benchmark.py --models your-model-name --datasets gsm8k
```

### Memory Management

For large models on limited GPU memory:

```yaml
models:
  large-model:
    load_in_8bit: true  # Enable 8-bit quantization
    device_map: "auto"  # Automatic device placement
```

## 📈 Evaluation Metrics

- **Accuracy**: Percentage of correct answers
- **Latency**: Average time per problem (seconds)
- **Token Usage**: Total and average tokens generated
- **Pass@1**: Correctness on first attempt

## 🐛 Troubleshooting

### CUDA Out of Memory
- Enable 8-bit quantization: `load_in_8bit: true`
- Reduce batch size in config
- Use smaller models
- Unload models between evaluations

### HuggingFace Model Access
- Some models require authentication
- Set `HF_TOKEN` in `.env`
- Accept model license on HuggingFace website

### API Rate Limits
- Adjust temperature and max_tokens
- Add delays between API calls
- Monitor OpenAI usage dashboard

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{math_reasoning_benchmark,
  title = {Math Reasoning Benchmark},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/math_benchmark_experiment}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.
