# Climate Science Fact-Checking System

This repository contains an implementation of a fact-checking system for climate science claims using the Qwen3-1.8B model.

## Overview

The system performs the following tasks:
1. Retrieves relevant evidence from a knowledge source for a given claim
2. Classifies the claim as SUPPORTS, REFUTES, NOT_ENOUGH_INFO, or DISPUTED based on the retrieved evidence

## System Architecture

The system consists of two main components:

1. **Hybrid Retrieval Component**:
   - TF-IDF for keyword matching
   - Semantic search using Qwen3-1.8B embeddings
   - Combined scoring for optimal retrieval

2. **Few-shot Classification Component**:
   - In-context learning with Qwen3-1.8B
   - Evidence conflict detection for disputed claims
   - Label similarity comparison

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from optimized_fact_checking import OptimizedFactCheckingSystem

# Initialize the system
system = OptimizedFactCheckingSystem()

# Load evidence corpus
system.load_evidence_corpus("path/to/evidence.json")

# Compute evidence vectors (this may take a while)
system.compute_evidence_vectors()

# Process claims
system.process_claims("path/to/claims.json", "output.json")

# Evaluate the system
system.evaluate("output.json", "path/to/groundtruth.json")
```

### Running on the COMP90042 Dataset

```bash
python optimized_fact_checking.py
```

This will:
1. Initialize the system
2. Process the development set
3. Evaluate performance
4. Process the test set for leaderboard submission

## Model Details

- **Model**: Qwen3-1.8B
- **Memory Usage**: ~8GB
- **Performance**: See evaluation results in the system output

## Future Improvements

1. Fine-tuning the model for the specific task (not done here due to project constraints)
2. More sophisticated claim-evidence pair encoding
3. Cross-attention between claim and evidence
4. Ensemble methods for classification 