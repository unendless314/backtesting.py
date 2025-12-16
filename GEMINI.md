## Project Overview

This project, `backtesting.py`, is a Python library for backtesting trading strategies. It is designed to be lightweight, fast, and easy to use. It allows users to test their trading strategies on historical OHLC (Open, High, Low, Close) data for any financial instrument.

The core of the library consists of the `Strategy` class, which users subclass to define their trading logic, and the `Backtest` class, which is used to run the backtests, optimizations, and visualizations.

The project uses `pandas` for data manipulation, `numpy` for numerical operations, and `bokeh` for creating interactive plots.

## Building and Running

### Dependencies

The project requires Python 3.9+. The core dependencies are `numpy`, `pandas`, and `bokeh`. Additional dependencies for documentation, testing, and development are specified in `setup.py`.

It is highly recommended to use a Python virtual environment for development.

To install all dependencies, you can use pip:

```bash
# Install core dependencies
pip install .

# Install test dependencies
pip install .[test]

# Install documentation dependencies
pip install .[doc]

# Install development dependencies
pip install .[dev]
```

### Running Tests

The test suite is built on Python's `unittest` framework. The tests can be run using the following command from the root of the project:

```bash
python -m backtesting.test
```

This command is also used in the CI pipeline (`.github/workflows/ci.yml`) to ensure code quality and correctness.

### Linting and Type Checking

The project uses `flake8` for linting and `mypy` for static type checking. These checks are also part of the CI pipeline.

To run them locally:

```bash
flake8 backtesting setup.py
mypy --no-warn-unused-ignores backtesting
```

The configuration for the `ruff` linter is also present in `pyproject.toml`.

### Building Documentation

The documentation is built using `pdoc3`. The build process is orchestrated by a shell script.

To build the documentation:
```bash
./doc/build.sh
```

## Development Conventions

*   **Code Style:** The code follows standard Python conventions (PEP 8), enforced by `flake8` and `ruff`.
*   **Testing:** New features and bug fixes should be accompanied by tests. The tests are located in the `backtesting/test/` directory. The main test file is `_test.py`.
*   **API Design:** The library exposes a simple and intuitive API. The main entry points are the `Backtest` and `Strategy` classes.
*   **Strategies:** Users define their own strategies by subclassing the `Strategy` class and implementing the `init()` and `next()` methods.
    *   `init()`: Used for one-time setup and to pre-compute indicators in a vectorized way. Indicators are declared using the `self.I()` method.
    *   `next()`: Called for each data point (candlestick bar) to execute the trading logic.
*   **Optimization:** Strategy parameters can be optimized using the `Backtest.optimize()` method. Parameters to be optimized are declared as class variables in the `Strategy` subclass.
*   **Data:** The library expects data as a `pandas.DataFrame` with `Open`, `High`, `Low`, `Close`, and optional `Volume` columns.

A simple example of a strategy:

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.n1)
        self.ma2 = self.I(SMA, price, self.n2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

# Run a backtest
bt = Backtest(GOOG, SmaCross, commission=.002)
stats = bt.run()
bt.plot()

# Optimize the strategy
stats = bt.optimize(n1=range(5, 30, 5),
                    n2=range(10, 70, 5),
                    maximize='Equity Final [$]',
                    constraint=lambda param: param.n1 < param.n2)
```
