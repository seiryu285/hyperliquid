# HyperLiquid Trading Agent

An advanced trading agent for the HyperLiquid platform, featuring anomaly detection, risk management, and reinforcement learning capabilities.

## Features

### Core Components

1. **Caching System**
   - Redis-based caching with in-memory fallback
   - Thread-safe operations
   - Bulk operations optimization
   - Automatic retry and error handling

2. **Anomaly Detection**
   - Market data anomaly detection using Isolation Forest
   - Model persistence and automatic updates
   - Real-time data processing
   - Alert notification system

3. **Risk Management**
   - Kelly Criterion position sizing
   - Risk metrics calculation
   - Automated loss limitation
   - Portfolio optimization

4. **Reinforcement Learning**
   - Custom PPO implementation
   - GRPO extensions
   - Custom loss functions
   - Market state representation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hyperliquid-agent.git
cd hyperliquid-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start Redis server:
```bash
redis-server
```

## Usage

1. Prepare backtest data:
```bash
python scripts/prepare_backtest_data.py
```

2. Train the anomaly detection model:
```bash
python scripts/train_anomaly_detector.py
```

3. Start the trading agent:
```bash
python main.py
```

## Project Structure

```
project_root/
├── core/                 # Core functionality
│   ├── cache.py         # Caching system
│   └── config.py        # Configuration management
├── ml/                  # Machine learning components
│   └── anomaly_detection.py
├── risk_management/     # Risk management tools
│   └── position_sizing/
├── rl_agent/           # Reinforcement learning
│   └── algorithms/
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── main.py            # Application entry point
```

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Check code style:
```bash
black .
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HyperLiquid team for their excellent platform
- Contributors and maintainers of the open-source libraries used in this project