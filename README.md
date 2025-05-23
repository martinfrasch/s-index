<p align="center">
  <img src="sage_k_logo.png" alt="SAGE-K Logo" width="200"/>
</p>

<h2 align="center">SAGE-K</h2>
<h4 align="center"><i>S-Index for Advanced Graph-based Engineering of Knowledge</i></h4>
<h4 align="center"><i>An intelligent framework for dynamic, graph-based scientific knowledge integration and impact scoring</i></h4>

---


📊 **S Index: A Quantifiable Metric of Scholarly Impact with Data Sharing Incentives**

This repository provides an executable, extensible framework to compute the S Index — a composite, real-time metric for quantifying researcher impact. It draws on structured metadata from Semantic Scholar, Figshare, and other repositories, integrating structural, semantic, temporal, and data-sharing dimensions.

🔍 **What is the S Index?**

The S Index is a weighted composite score (scaled 0–100) that captures the scholarly contribution of a researcher by combining four key categories:

- 📐 **Structural Metrics (40%)**: Graph-theoretic features from citation networks (connectivity, centrality, clustering, PageRank).
- 🧠 **Semantic Metrics (30%)**: Embedding-based similarity of abstracts, knowledge completeness, ontological diversity.
- 📦 **Data Sharing Metrics (15%)**: Open science participation via dataset uploads (e.g., on Figshare).
- ⏱️ **Knowledge Update Metrics (15%)**: Frequency and volume of recent knowledge additions to the graph.

🥕 **The Carrot Effect: Why Researchers Should Care**

The S Index rewards data sharing explicitly:

📈 Each dataset openly shared increases the S Index, providing researchers with tangible incentives for open science.

**Benefits include:**
- Increased visibility and academic reputation.
- Better chances for grants, collaboration, and promotion.
- Real-time feedback via dashboards showcasing how actions (like dataset sharing) improve the score.

⚙️ **Architecture Overview**

**Data Sources → Unified Graph → Metrics Engine → S Index → Dashboards**

- **Data Ingestion**: Pulls papers and citations via the Semantic Scholar API, and datasets via the Figshare API.
- **Graph Builder**: Constructs a timestamped, directed citation graph with embedded text.
- **Metric Engine**: Computes individual scores using normalized mathematical formulations.
- **S Index Calculator**: Applies fixed weights to produce a single interpretable score.
- **Carrot Effect Logic**: Tracks and rewards new shared datasets to boost the score transparently.

📦 **Features**
- Full knowledge graph construction with metadata  
- Abstract embedding and similarity scoring
- Builds upon BioMedLM
- Explicit timestamp-based knowledge update scoring  
- Objective reward mechanism for data sharing  
- Extensible to additional APIs and ontologies  

🚀 **Getting Started**
```bash
pip install -r requirements.txt
python s_index.py
```

🧾 **You’ll need**:
- A Semantic Scholar author ID
- Researcher name string (to query Figshare)


📈 **Example Output**
	- Author’s S Index: 68.75
 
	- Data Sharing Score: 0.90 (6 datasets shared)
 
	- Structural Score: 0.65
 
	- Semantic Score: 0.72
 
	- Knowledge Update Score: 0.45


📜 **License** 

MIT License. Feel free to build on this framework for academic, research, or open science projects.
