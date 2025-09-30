## **Architectural Outline for Future Development**

This is the complete strategic roadmap for building a full-scale, living knowledge graph to power the QIC-Index.

\# Architectural Outline: A Living Knowledge Graph for the QIC-Index

This document outlines the high-level architecture for a scalable platform designed to ingest scholarly data, construct a living knowledge graph, and calculate the QIC-Index in a continuous, automated fashion.

\#\# Guiding Principles

\* \*\*Modularity:\*\* The system will be composed of independent microservices for easy development, scaling, and maintenance.  
\* \*\*Scalability:\*\* The architecture must handle a growing volume of scholarly outputs from millions of researchers worldwide.  
\* \*\*Extensibility:\*\* The system should be designed to easily incorporate new data sources and scoring algorithms in the future.  
\* \*\*Transparency:\*\* The processes for data ingestion and score calculation should be well-documented and auditable.

\---

\#\# Phase 1: The Foundational Layer \- Data & Knowledge Graph Core

This phase focuses on building the core infrastructure to ingest, normalize, and store the interconnected data that forms the basis of the QIC-Index.

\#\#\# 1.1. Data Ingestion & Integration Service

This service is the gateway for all external data. It will be a collection of connectors responsible for fetching data from various sources.

\* \*\*Data Sources:\*\*  
    \* \*\*Researcher Profiles:\*\* ORCID  
    \* \*\*Institutional Identifiers:\*\* Research Organization Registry (ROR)  
    \* \*\*Data Repositories:\*\* APIs from Zenodo, Dryad, Figshare, etc.  
    \* \*\*Publication & Citation Data:\*\* Crossref, DataCite, OpenAlex, Semantic Scholar  
    \* \*\*Grant & Funding Data:\*\* APIs from funders like NIH, NSF, etc.  
\* \*\*Mechanism:\*\*  
    \* \*\*API Connectors:\*\* Custom connectors for each data source.  
    \* \*\*Webhooks:\*\* For real-time updates.  
    \* \*\*Batch Processing:\*\* For initial data loading and periodic reconciliation.  
\* \*\*Data Normalization:\*\* A crucial step to transform heterogeneous source data into a standardized internal JSON-LD format before it enters the knowledge graph.

\#\#\# 1.2. Knowledge Graph Core

This is the heart of the system, storing entities and their complex relationships.

\* \*\*Technology:\*\* A native graph database is essential.  
    \* \*\*Recommended:\*\* Neo4j, Amazon Neptune, or TigerGraph.  
\* \*\*Core Ontology (Schema):\*\* A well-defined schema is critical for consistency.

    \*\*Nodes:\*\*  
    \`\`\`  
    \- Person (Properties: orcid, name, affiliations)  
    \- Institution (Properties: ror\_id, name, location)  
    \- Dataset (Properties: doi, title, metadata\_url, license)  
    \- Publication (Properties: doi, title, journal)  
    \- Grant (Properties: grant\_id, funder, title)  
    \`\`\`

    \*\*Edges:\*\*  
    \`\`\`  
    \- AUTHORED\_BY  
    \- AFFILIATED\_WITH  
    \- CITES  
    \- REUSES  
    \- FUNDED\_BY  
    \- VERSION\_OF  
    \`\`\`

\#\#\# 1.3. Entity Resolution Service

This service ensures data integrity by preventing duplicate entities.

\* \*\*Primary Mechanism:\*\* Leverage Persistent Identifiers (PIDs) like DOIs, ORCIDs, and ROR IDs.  
\* \*\*Secondary Mechanism:\*\* Use fuzzy matching algorithms and machine learning models for data lacking PIDs.

\---

\#\# Phase 2: The Enrichment Layer \- QIC Calculation Engines

This layer consists of specialized microservices that run continuously to analyze the data in the knowledge graph and calculate the component scores.

\#\#\# 2.1. Quality (Q) Engine

This service analyzes new and updated \`Dataset\` nodes to calculate their Quality score.

\* \*\*Process:\*\*  
    1\.  Fetches metadata for a given dataset DOI.  
    2\.  Applies a rule-based scoring rubric based on FAIR principles.  
    3\.  Writes the calculated \`q\_score\` as a property on the \`Dataset\` node.  
\* \*\*Technology:\*\* A containerized Python service.

\#\#\# 2.2. Impact (I) Engine

A continuously running service that scours the scholarly web for evidence of data reuse.

\* \*\*Process:\*\*  
    1\.  Regularly queries sources for mentions of dataset DOIs.  
    2\.  When a reuse event is found, it creates a \`REUSES\` edge in the knowledge graph.  
    3\.  The \`i\_score\` is then calculated dynamically by querying the number of incoming \`REUSES\` edges.

\#\#\# 2.3. Collaboration (C) & QIC Aggregator Service

This service performs the final calculations.

\* \*\*C-Score Calculation:\*\* For any given \`Dataset\`, this service runs a graph traversal query to find all connected \`Person\` and \`Institution\` nodes and calculates the \`c\_score\`.  
\* \*\*Final Aggregation:\*\* To calculate an author's total QIC-Index, this service:  
    1\.  Finds the target \`Person\` node.  
    2\.  Traverses the graph to find all \`Dataset\` nodes they \`AUTHORED\_BY\`.  
    3\.  For each dataset, it retrieves or calculates the Q, I, and C scores.  
    4\.  It computes \`s\_j \= Q × I × C\` for each dataset and sums them to get the final \`S\_i\`.

\---

\#\# Phase 3: The Access Layer \- API & Applications

This layer exposes the calculated scores and underlying data to end-users and other systems.

\#\#\# 3.1. Public API

A well-documented public API is crucial for adoption.

\* \*\*Technology:\*\* A GraphQL API is highly recommended as it is a natural fit for querying graph data.  
\* \*\*Example Endpoints:\*\*  
    \* \`getAuthorQIC(orcid: "...")\`: Returns the total QIC-Index for an author.  
    \* \`getDataObject(doi: "...")\`: Returns the full QIC breakdown for a dataset.

\#\#\# 3.2. Web Portal / User Interface

A user-facing web application that consumes the public API.

\* \*\*Features:\*\*  
    \* \*\*Researcher Profiles:\*\* Allow researchers to log in with ORCID and view their QIC-Index.  
    \* \*\*Institutional Dashboards:\*\* Provide analytics for administrators.  
    \* \*\*Graph Explorer:\*\* A visualization tool for users to explore connections within the knowledge graph.

\---

\#\# Cross-Cutting Concerns

\* \*\*Infrastructure:\*\* A cloud-native approach (AWS, GCP, Azure) using a container orchestration platform like Kubernetes is essential for scalability.  
\* \*\*Monitoring & Logging:\*\* Comprehensive monitoring of all services, API usage, and data ingestion pipelines.  
\* \*\*Data Governance & Ethics:\*\* Establish clear policies for data privacy, provide mechanisms for users to correct inaccuracies, and ensure the QIC algorithm is transparent and publicly documented.

