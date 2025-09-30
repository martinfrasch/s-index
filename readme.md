# **QIC-Index: A Data-Centric Metric for Research Data Sharing**

This repository contains the supplemental materials for the manuscript, **"The QIC-Index: A Novel, Data-Centric Metric for Quantifying the Impact of Research Data Sharing."**

## **Abstract**

Modern science faces an "incentive gap" where traditional, publication-centric metrics like the h-index fail to value the critical contribution of research data sharing. This discourages open science practices and hinders collaborative progress. To address this, we propose the QIC-Index, a novel metric designed to quantify and reward the sharing of high-quality research data. The QIC-Index moves beyond publications to assess the value of individual data objects based on their **Quality (Q)**, **Impact (I)**, and **Collaboration (C)**. This framework aligns individual rewards with the collective goals of open science, fostering a more transparent, efficient, and collaborative research culture.

## **The QIC-Index Framework**

The score for an individual data object (sⱼ) is calculated as:

sⱼ \= Qⱼ × Iⱼ × Cⱼ

An author's total QIC-Index (Sᵢ) is the sum of these scores:

Sᵢ \= ∑ sⱼ

This model directly evaluates the data object itself, a significant conceptual advance over indirect, publication-centric metrics.

## **System Architecture**

The proposed system for calculating the QIC-Index is based on a data pipeline that ingests data from repositories, analyzes it against FAIR principles, and tracks its reuse over time within a knowledge graph.

**\[Insert** Figure 1: QIC-Index System Architecture **Image Here\]**

*Figure 1: The system architecture for the QIC-Index, showing the flow from data ingestion to the final aggregated score.*

## **Manuscript and Citation**

For a full description of the methodology, comparative analysis, and conceptual framework, please refer to the final manuscript.

**Link** to Your Final Published Manuscript or **Pre-print: \[Insert Link Here\]**

### **A Note on the Original S-Index Concept (h/√n)**

The initial proof-of-concept for this project explored a different metric, defined as S-Index \= h-index / √n\_co-authors. The code and live demo in this repository reflect that **deprecated, early-stage idea**.

Through rigorous review, this initial concept was superseded by the far more robust and conceptually sound **QIC-Index** described in the final manuscript. The original framework was found to be misaligned to promote data sharing and collaboration. The QIC-Index, by contrast, directly addresses the "incentive gap" by focusing on the data object itself.

The original code is retained here for archival purposes.