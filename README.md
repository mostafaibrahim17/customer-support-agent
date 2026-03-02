# customer-support-agent
A production-ready customer support agent using Weave and W&B Inference.

![Python](https://img.shields.io/badge/language-Python-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview
This project implements a customer support agent that leverages Weave and OpenAI's capabilities to provide efficient and safe customer interactions. It includes guardrails to ensure account and content safety, making it suitable for production use.

## Tutorial Reference
This repository accompanies a tutorial article. You can read it [here](URL_PLACEHOLDER).

## Prerequisites
- Python 3.8 or higher
- Weave library
- OpenAI API key (create an account at [OpenAI](https://openai.com/))
- W&B account (create an account at [Weights & Biases](https://wandb.ai/))

## Quick Start
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/customer-support-agent.git
   cd customer-support-agent
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```bash
   echo "WANDB_API_KEY=your-api-key" >> .env
   ```
4. **Run the application**:
   ```bash
   python src/main.py
   ```

## Project Structure
```
customer-support-agent/
├── .gitignore
├── requirements.txt
└── src/
    ├── agent.py
    ├── guardrails.py
    └── main.py
```

## Key Concepts
- **Weave**: A framework for building and deploying machine learning applications.
- **Guardrails**: Safety checks implemented to ensure that the agent operates within defined safety parameters.

## Code Highlights
- **Guardrail Implementation**: The `FlaggedAccountGuardrail` class in `guardrails.py` demonstrates how to implement safety checks for flagged accounts, ensuring that sensitive operations are restricted.
- **Asynchronous Operations**: The use of `async` in `run_guarded_agent` in `agent.py` allows for non-blocking execution, improving the responsiveness of the customer support agent.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.