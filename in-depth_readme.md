<p align="center">
  <img src="sage_k_logo.png" alt="SAGE-K Logo" width="200"/>
</p>

<h1 align="center">SAGE-K: Python Implementation</h1>
<h3 align="center"><i>S-Index for Advanced Graph-based Engineering of Knowledge</i></h3>
<h4 align="center"><i>An intelligent framework for dynamic, graph-based scientific knowledge integration and impact scoring, implemented in Python.</i></h4>

---

## Overview

This repository contains the Python implementation for SAGE-K, a framework designed to compute the **S-Index**. The S-Index is a dynamic, real-time metric that quantifies the scholarly impact of researchers and their contributions. It uniquely integrates structural, semantic, temporal, and data-sharing dimensions into a composite score.

A core design principle of the S-Index is the **"Carrot Effect"** 🥕: it explicitly rewards researchers for open science practices, particularly the sharing of datasets. This Python script provides the tools to construct a knowledge graph from scholarly data, calculate the necessary metrics, and compute the S-Index.

## Core Concept: The S-Index

The S-Index is a composite score, scaled from 0 to 100, that reflects a researcher's scholarly contributions. It is calculated as a weighted sum of four key categories:

1.  📐 **Structural Metrics (40%):** Derived from graph-theoretic features of the citation and collaboration network. This includes metrics like connectivity, centrality (degree), PageRank, betweenness, and clustering coefficients.
2.  🧠 **Semantic Metrics (30%):** Assesses the content and meaning within scholarly works. This implementation uses BioMedLM (specifically, a PubMedBERT variant) to generate embeddings for abstracts and calculate semantic similarity between connected works. It also includes a measure of knowledge completeness.
3.  ⏱️ **Knowledge Update Metrics (15%):** Measures the recency and volume of contributions to the knowledge graph. This involves tracking new node additions (integration rate) and shifts in graph centrality, reflecting the evolving nature of the knowledge base.
4.  🥕 **Data Sharing Metrics (15%):** Explicitly rewards open science practices. This score increases with the number of datasets openly shared by a researcher, normalized against a target count. This is the "carrot incentive" designed to encourage data sharing.

## Key Features of this Python Implementation

* **Knowledge Graph Construction:** Dynamically builds and updates a directed, multi-edge knowledge graph using the `networkx` library. Nodes can represent papers, datasets, and potentially researchers, while edges represent citations, authorship, data sharing links, etc.
* **Advanced Semantic Analysis:** Leverages state-of-the-art **BioMedLM (PubMedBERT)** via the `transformers` library to generate rich, domain-specific embeddings from abstracts and other textual content. These embeddings are then used for calculating semantic similarity.
* **Modular Metric Calculation:** Each of the four S-Index components has dedicated Python functions for clarity and maintainability (`calculate_structural_metrics`, `calculate_semantic_metrics`, `calculate_knowledge_update_metrics`, `calculate_data_sharing_metrics`).
* **Dynamic S-Index Computation:** The `calculate_s_index` function aggregates the component scores based on their defined weights. The script demonstrates how the S-Index can be recalculated as the graph evolves with new papers and datasets.
* **Extensibility for Relation Extraction:** Includes a placeholder mechanism for extracting relationships (e.g., "SUPPORTS," "DISPUTES") from abstracts using `spaCy` if available. This feature is designed for future enhancement.
* **Graph Transformer Components:** The script includes PyTorch-based classes for `GraphTransformer`, `TransformerEncoder`, and `GraphAttentionNetwork`. While not used in the basic S-Index calculation loop in the example, these components are provided for advanced downstream tasks such as link prediction, node classification, or predicting the impact of future updates.

## Architecture Overview (Python Script Focused)

The Python script follows a logical flow:

1.  **Model Initialization:** Loads pretrained models like BioMedLM (via `transformers`) and optionally `spaCy` at startup.
2.  **Data Input (Simulated):** The example usage (`if __name__ == "__main__":`) simulates the addition of paper and dataset information. In a production system, this data would come from APIs like Semantic Scholar or Figshare.
3.  **Knowledge Graph Construction (`KnowledgeGraphBuilder`):**
    * Papers and datasets are added as nodes. Textual information (e.g., abstracts, descriptions) is converted into BioMedLM embeddings and stored as node attributes.
    * Relationships (citations, data links) are added as edges.
    * The graph maintains a `previous_graph` state to compute update metrics.
4.  **Metric Calculation:** Individual functions process the current graph (and `previous_graph` where applicable) to compute the raw values for structural, semantic, knowledge update, and data sharing metrics.
5.  **S-Index Aggregation:** The `calculate_s_index` function takes the (averaged and normalized where necessary) component scores and applies the predefined weights to output the final S-Index (0-100).
6.  **Output:** The script prints the S-Index and component scores at various stages of graph evolution.

For a visual representation of this flow, please see the [Python Script Execution Architecture Diagram](sage_k_script_architecture.txt).

## Code Structure (`s_index.py`)

The `s_index.py` script is organized into several key sections:

* **Global Setup & Model Loading:** Initializes BioMedLM tokenizer/model and the `spaCy` NLP model. Defines global constants for S-Index weights and normalization targets.
* **S-Index Calculation Functions:**
    * `get_biomedlm_embedding(text)`: Generates embedding for a given text using the loaded BioMedLM model.
    * `calculate_structural_metrics(graph)`: Computes graph-based structural scores.
    * `calculate_semantic_similarity_from_embeddings(embedding1, embedding2)`: Calculates cosine similarity between two BioMedLM embeddings.
    * `calculate_semantic_metrics(graph)`: Computes semantic scores using node embeddings.
    * `calculate_knowledge_update_metrics(graph, previous_graph)`: Calculates scores based on graph evolution.
    * `calculate_data_sharing_metrics(graph, researcher_id)`: Calculates the data sharing incentive score.
    * `calculate_s_index(...)`: Aggregates component scores into the final S-Index.
* **`KnowledgeGraphBuilder` Class:**
    * Manages the creation and modification of the knowledge graph (`self.graph`, `self.previous_graph`).
    * `add_paper(paper_data)`: Adds paper nodes and citation edges, generates embeddings.
    * `add_dataset(dataset_data)`: Adds dataset nodes, generates embeddings from descriptions.
    * `create_relationship(source, target, relation_type)`: Adds edges to the graph.
    * `update_node(node_id, new_data)`: Modifies existing nodes.
    * `extract_relations_from_abstract_spacy(abstract, paper_id)`: Placeholder for `spaCy`-based relation extraction.
* **Graph Transformer Components (PyTorch `nn.Module`s):**
    * `GraphTransformer`, `TransformerEncoder`, `GraphAttentionNetwork`: Classes defining neural network architectures for graph learning tasks.
* **Example Usage (`if __name__ == "__main__":`)**
    * Demonstrates how to use `KnowledgeGraphBuilder` to create a graph and how the S-Index is calculated and updated incrementally.

## Setup and Dependencies

This script is written in Python 3. Key libraries include:

* `networkx`: For graph creation and manipulation.
* `numpy`: For numerical operations, especially with embeddings.
* `spacy`: For NLP tasks, particularly the placeholder relation extraction. (Optional if relation extraction is not used).
    * You'll need to download a `spaCy` model: `python -m spacy download en_core_web_md`
* `torch`: The PyTorch library, core for the Graph Transformer components and `transformers`.
* `torch_geometric`: For Graph Neural Network layers like `GATConv`.
* `transformers`: From Hugging Face, for loading and using pretrained models like BioMedLM (PubMedBERT).
* `scikit-learn`: For utility functions like `cosine_similarity`.

You can typically install these using pip:
```bash
pip install networkx numpy spacy torch torch_geometric transformers scikit-learn
# Don't forget the spaCy model download (see above)
