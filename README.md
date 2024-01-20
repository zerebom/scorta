# Scorta

Scorta is a versatile library designed to facilitate the reuse of machine learning pipelines across multiple projects, including but not limited to 2-stage recommender systems. It offers a streamlined and straightforward interface that can be easily adapted for various machine learning competitions.

## Features

- **Reusability**: Crafted with the aim to be reused in multiple ML pipelines.
- **Compatibility**: Suitable for use in 2-stage recommender systems and more.
- **Examples Included**: The `samples` directory contains real-world code examples from ML competitions.
- **Simplicity and Flexibility**: Maintains simplicity while being adaptable for use in different competition contexts.

## Installation

You can install Scorta directly from the repository using pip:

```bash
pip3 install git+https://github.com/zerebom/scorta.git
```

## Directory Structure

Scorta comes with a well-organized directory structure:

```
scorta/
├── eda/
├── feature/
├── model/
├── recsys/
├── testing/
└── utils/
samples/
├── 001_binary_classification.ipynb
├── atma-16/
└── atma15/
```

Within the `samples` directory, you can find notebooks (`*.ipynb`) demonstrating how `scorta` was utilized in ML competitions, providing a practical reference for your projects.

## Usage Example

Here's a quick example of how to extend the `Candidate` class from the `scorta.recsys` module for a competition:

```python
from scorta.recsys.candidate_generate import Candidate

class Atma16Candidate(Candidate):
    # initialization and method definitions
    ...

class LastSeenedCandidate(Atma16Candidate):
    # A specific strategy for scoring within the same session
    def generate(self) -> pl.DataFrame:
        # Logic to generate candidate DataFrame
        ...
```

With Scorta, you can define your own candidates and strategies to fit the needs of your recommender system, ensuring high reusability and easy experimentation.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.


## Contact

If you have any questions or suggestions, please reach out to us at the repository on GitHub.
