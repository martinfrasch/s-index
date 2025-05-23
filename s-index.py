# s_index.py

import networkx as nx # [1, 12]
import numpy as np # [1, 12]
import spacy # [1, 12, 13]
import torch # [1, 12]
import torch.nn as nn # [1]
from torch_geometric.nn import GATConv, GCNConv # [1, 14]

# Load a suitable spaCy model for semantic analysis [13]
# NOTE: You need to download this model separately after installing spacy:
# python -m spacy download en_core_web_md
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("SpaCy model 'en_core_web_md' not found. Please download it by running:")
    print("python -m spacy download en_core_web_md")
    # Exit or handle the error appropriately if the model is essential for execution
    # For this script, we'll allow it to continue, but semantic functions will fail.
    nlp = None
    print("Semantic metrics calculation may not work without the spaCy model.")


# --- 1. S Index Calculation Functions --- [13, 15-19]

def calculate_structural_metrics(graph):
    """
    Calculates various structural metrics for the graph. [13, 15]
    Returns a dictionary of structural metrics.
    """
    metrics = {}
    if not graph or graph.number_of_nodes() == 0:
        # Return default or zero metrics for empty graphs
        return {
            'connectivity': 0, # [13]
            'centrality': 0, # [13]
            'pagerank': 0, # [13]
            'betweenness': 0, # [15]
            'clustering': 0 # [15]
        }

    try:
        metrics['connectivity'] = nx.edge_connectivity(graph) # [13]
    except nx.NetworkXNoPath:
         metrics['connectivity'] = 0 # Handle disconnected graph
    except nx.NetworkXPointlessConcept:
         metrics['connectivity'] = 0 # Handle graphs with too few nodes

    # Centrality measures might fail on empty graphs, handled above
    metrics['centrality'] = np.mean(list(nx.degree_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0 # [13]
    metrics['pagerank'] = np.mean(list(nx.pagerank(graph).values())) if graph.number_of_nodes() > 0 else 0 # Added PageRank [13]
    metrics['betweenness'] = np.mean(list(nx.betweenness_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0 # Added betweenness [15]
    metrics['clustering'] = np.mean(list(nx.clustering(graph).values())) if graph.number_of_nodes() > 0 else 0 # Added clustering coefficient [15]

    return metrics

def calculate_semantic_similarity(node1_text, node2_text):
    """
    Calculates the semantic similarity between two text nodes using spaCy. [15]
    Returns a similarity score between 0 and 1.
    """
    if nlp is None:
        print("SpaCy model not loaded. Cannot calculate semantic similarity.")
        return 0.0

    if not node1_text or not node2_text:
        return 0.0 # Return 0 if either text is missing [15]

    try:
        doc1 = nlp(node1_text)
        doc2 = nlp(node2_text)
        # Using .similarity() requires vector models like 'md' or 'lg' [15]
        return doc1.similarity(doc2)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_semantic_metrics(graph):
    """
    Calculates semantic metrics based on text within the graph nodes. [16]
    Currently focuses on average semantic similarity of connected nodes.
    Returns a dictionary of semantic metrics.
    """
    semantic_scores = []
    if nlp is None:
        print("SpaCy model not loaded. Cannot calculate semantic metrics.")
        return {'completeness': len(graph.nodes) / 80000000, 'similarity': 0.0} # [18]

    # Calculate average semantic similarity for edges [16]
    # This might be computationally expensive for large graphs
    for u, v in graph.edges():
        node1_text = graph.nodes[u].get('abstract', '') # Assumes abstract text is stored in node attributes [16, 20]
        node2_text = graph.nodes[v].get('abstract', '') # [16, 20]
        similarity = calculate_semantic_similarity(node1_text, node2_text) # [16]
        semantic_scores.append(similarity)

    # Other semantic metrics mentioned but not explicitly coded: [5, 21]
    # - Ontological Coverage
    # - Relationship Diversity
    # - Knowledge Completeness (approximated here by node count vs target) [18]

    avg_similarity = np.mean(semantic_scores) if semantic_scores else 0.0 # return 0 if there are no edges [16]

    metrics = {
        'completeness': len(graph.nodes) / 80000000, # Example using the target node count [18] - TARGET_NODE_COUNT should be a constant
        'similarity': avg_similarity
    }

    return metrics

def calculate_update_metrics(graph, previous_graph):
    """
    Calculates update metrics based on changes to the graph. [16-18]
    Compares the current graph to a previous version.
    Returns a dictionary of update metrics.
    """
    metrics = {}

    if not previous_graph or previous_graph.number_of_nodes() == 0:
        # If no previous graph, assume this is the initial state or full update
        # Integration rate is 1, centrality shift is 0, temporal consistency is 1
        return {'integration_rate':1.0, 'centrality_shift': 0.0, 'temporal_consistency':1.0} [17]

    added_nodes = set(graph.nodes) - set(previous_graph.nodes) [17]
    # added_edges = set(graph.edges) - set(previous_graph.edges) # Could also track edge changes

    if not added_nodes:
        # If no new nodes, integration rate is 0 [17]
        # Centrality shift can still occur from changes to existing nodes/edges, but simplify for now
        return {'integration_rate':0.0, 'centrality_shift': 0.0, 'temporal_consistency':1.0} [17]

    # Integration Rate: Percentage of new nodes in the current graph [17]
    integration_rate = len(added_nodes) / len(graph.nodes) if graph.number_of_nodes() > 0 else 0.0 [17]

    # Centrality Shift: Change in average centrality [17]
    centrality_before = np.mean(list(nx.degree_centrality(previous_graph).values())) if previous_graph.number_of_nodes() > 0 else 0.0 [17]
    centrality_after = np.mean(list(nx.degree_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0.0 [17]
    centrality_shift = abs(centrality_after - centrality_before) [17] # Absolute difference

    # Temporal Consistency: Placeholder for future implementation [12, 18]
    # This would likely involve checking time-based relationships or publication dates
    temporal_consistency = 1.0 # Placeholder [18]

    metrics['integration_rate'] = integration_rate
    metrics['centrality_shift'] = centrality_shift
    metrics['temporal_consistency'] = temporal_consistency

    return metrics


# Define default weights [5, 18, 22-24]
STRUCTURAL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.3
UPDATE_WEIGHT = 0.2
TARGET_NODE_COUNT = 80000000 # Example using the target node count [18, 25-27] - Should be defined elsewhere

def calculate_s_index(graph, previous_graph=None):
    """
    Calculates the S Index based on structural, semantic, and update metrics. [5, 8, 18, 19, 28]
    Takes an optional previous_graph argument to calculate update metrics. [18]
    Returns a single S Index score.
    """
    # Get metrics [19]
    structural_metrics = calculate_structural_metrics(graph) [18]
    semantic_metrics = calculate_semantic_metrics(graph) [18]
    update_metrics = calculate_update_metrics(graph, previous_graph) [19]

    # Calculate weighted averages [19]
    # NOTE: The source formula S = sum(w_i * (x_i - mu_i) / sigma_i) suggests normalization [22].
    # The code snippet [29] and the `calculate_s_index` function here [18]
    # use a simpler weighted sum of the raw metric values or means.
    # This implementation follows the simpler code snippet structure. [19]

    # Calculate mean of metrics within each category for the weighted sum [19]
    avg_structural = np.mean(list(structural_metrics.values())) if structural_metrics else 0.0
    avg_semantic = np.mean(list(semantic_metrics.values())) if semantic_metrics else 0.0
    avg_update = np.mean(list(update_metrics.values())) if update_metrics else 0.0

    weighted_score = (STRUCTURAL_WEIGHT * avg_structural +
                      SEMANTIC_WEIGHT * avg_semantic +
                      UPDATE_WEIGHT * avg_update) [19]

    # The S Index is described as being on a 0-100 scale [25-27].
    # The current calculation does not guarantee this scale.
    # A mapping or normalization step would be needed here to achieve a 0-100 scale.
    # For this implementation, we return the raw weighted score.
    s_index_score = weighted_score

    return s_index_score

# --- 2. Knowledge Graph Construction Class --- [6, 10, 19]

class KnowledgeGraphBuilder:
    """
    Builds and updates the knowledge graph. [19]
    Uses a NetworkX MultiDiGraph to allow multiple directed edges between nodes. [6, 10, 30]
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph() # [19, 30]
        # You might store a reference to the previous state for update metrics
        self.previous_graph = nx.MultiDiGraph() # To store a copy for update calculations

    def _update_previous_graph(self):
        """Copies the current graph to the previous state."""
        self.previous_graph = self.graph.copy()


    def add_paper(self, paper_data):
        """
        Adds a paper (as a node) and its citations (as edges) to the knowledge graph. [6, 20, 30]
        paper_data is expected to be a dictionary with 'id', 'abstract', and 'citations' keys.
        """
        # Before adding, update the previous graph state
        self._update_previous_graph()

        paper_id = paper_data.get('id')
        if paper_id is None:
            print("Warning: Paper data missing 'id'. Skipping.")
            return

        abstract_text = paper_data.get('abstract', '') # [20]
        abstract_embedding = self.embed_text(abstract_text) # [20] - Uses spaCy

        # Add node for the paper [20, 30]
        # Store relevant data as node attributes [20, 30]
        node_attributes = {'type': 'paper', 'abstract_embedding': abstract_embedding}
        node_attributes.update(paper_data) # Add all data from the dict
        self.graph.add_node(paper_id, **node_attributes) [20, 30]

        # Add citation edges [6, 10, 20, 30]
        citations = paper_data.get('citations', []) [20]
        for citation_id in citations:
            # Citation_id should be the ID of the paper being cited.
            # We add an edge from the current paper to the cited paper.
            self.create_relationship(paper_id, citation_id, 'CITES') [20, 30]

        # Use NLP to extract other relations and create relationships [6, 10, 20, 31]
        # This calls the placeholder method
        for relation in self.extract_relations_from_abstract(abstract_text, paper_id): [31]
             self.create_relationship(relation['source'], relation['target'], relation['type']) [31]


    def create_relationship(self, source, target, relation_type):
        """
        Creates a relationship (edge) between two nodes in the knowledge graph. [30, 31]
        Checks if nodes exist before creating the edge.
        """
        if source in self.graph.nodes and target in self.graph.nodes:
             self.graph.add_edge(source, target, type=relation_type) [30, 31]
        else:
             # Handle cases where target node doesn't exist yet (e.g., cited paper not added)
             # Depending on requirements, you might add the target node as a stub here
             print(f"Warning: Skipping edge {source} -> {target} ({relation_type}). One or both nodes not found.")


    def embed_text(self, text):
        """
        Embeds text using spaCy. [20, 31]
        Returns a NumPy array representing the text embedding, or None if spaCy not loaded or text is empty.
        """
        if nlp is None or not text: [31]
            return None
        try:
            # .vector property provides the embedding [31]
            return nlp(text).vector [31]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None

    def extract_relations_from_abstract(self, abstract, paper_id):
        """
        Extracts relationships from the abstract using NLP techniques (placeholder). [12, 31, 32]
        This is a placeholder implementation and needs significant enhancement. [12]
        Returns a list of dictionaries, e.g., [{'source': id1, 'target': id2, 'type': 'SUPPORTS'}].
        """
        # Placeholder: actual implementation will depend on the NLP model used and logic. [12, 32]
        # The provided example code looks for 'supports' or 'disputes' and attempts to find a target. [32-34]
        # A real system would need more sophisticated NLP/NER and linking to existing graph entities. [12, 32]

        relations = []
        if nlp is None or not abstract: [32]
            return relations

        try:
            doc = nlp(abstract) [32]
            for sent in doc.sents: [32]
                for token in sent: [32]
                    # Example logic based on provided snippet [32-34]
                    if token.text.lower() == 'supports' and token.head.i != token.i: [32]
                        target_node_id = None # Placeholder [33]
                        for possible_target in sent: [33]
                             if possible_target.i > token.i and possible_target.dep_ == 'pobj': [33]
                                 # This will likely involve some kind of lookup against the graph [33, 34]
                                 # to verify the target mentioned in text corresponds to an existing node ID. [34]
                                 # The find_node_id method below is also a placeholder. [34]
                                 target_node_id = self.find_node_id(possible_target.text) [33, 34]
                                 if target_node_id:
                                     break # Found a potential target, break inner loop
                        if target_node_id: [33]
                            relations.append({'source': paper_id, 'target': target_node_id, 'type': 'SUPPORTS'}) [33]

                    elif token.text.lower() == 'disputes' and token.head.i != token.i: [33]
                        target_node_id = None # Placeholder [34]
                        for possible_target in sent: [34]
                            if possible_target.i > token.i and possible_target.dep_ == 'pobj': [34]
                                # This will likely involve some kind of lookup against the graph [34]
                                target_node_id = self.find_node_id(possible_target.text) [34]
                                if target_node_id:
                                    break # Found a potential target, break inner loop
                        if target_node_id: [34]
                            relations.append({'source': paper_id, 'target': target_node_id, 'type': 'DISPUTES'}) [34]
        except Exception as e:
             print(f"Error during relation extraction: {e}")
             # Continue with empty relations list

        return relations [34]

    def find_node_id(self, node_text):
        """
        Placeholder method to search the existing graph for a node that matches a text. [12, 34, 35]
        In a real system, you would need a search index like Elasticsearch or a dedicated KG search method. [34]
        Returns the node ID if found, otherwise None.
        """
        # This is a very inefficient placeholder searching through all node attributes. [12, 35]
        # A real system needs an efficient lookup.
        for node_id, node_data in self.graph.nodes(data=True): [35]
            # Example: searching by 'title' attribute [35]
            if node_data.get('title', '').lower() == node_text.lower(): [35]
                return node_id
            # Add other potential matching criteria (e.g., abstract snippet, concept IDs)
        return None [35]


    def update_node(self, paper_id, new_paper_data):
        """
        Updates an existing node with new information. [35, 36]
        Updates node attributes, including re-embedding the abstract if present. [35]
        Re-extracts relations from the updated abstract. [36]
        """
        if paper_id in self.graph.nodes: [35]
            # Before updating, update the previous graph state
            self._update_previous_graph()

            abstract_text = new_paper_data.get('abstract','') [35]
            # Re-embed the abstract [35]
            abstract_embedding = self.embed_text(abstract_text) [35]

            # Update node attributes [35]
            self.graph.nodes[paper_id].update(new_paper_data) [35]
            self.graph.nodes[paper_id]['abstract_embedding'] = abstract_embedding # update embeddings [35]

            # Add any other necessary update logic here [36]
            # For example, update or re-evaluate existing edges related to this node

            # Re-extract and potentially add new relations from the updated abstract [36]
            # Note: This doesn't handle *removing* old relations if they are no longer valid.
            for relation in self.extract_relations_from_abstract(abstract_text, paper_id): [36]
                 self.create_relationship(relation['source'], relation['target'], relation['type']) [36]

        else:
            print(f"Warning: Node with ID {paper_id} not found. Cannot update.")
            # Depending on logic, you might call add_paper here if the node should be added
            # self.add_paper(new_paper_data) # Example: add if not exists


# --- 3. Graph Transformer Architecture Components --- [4, 7, 11, 14, 21, 36-38]
# These components are part of the larger model that *uses* the graph and S-index,
# but the S-index calculation itself relies primarily on the graph structure and node data.
# They are included here as they are part of the described codebase architecture. [4, 11, 21, 36]

class GraphTransformer(nn.Module):
    """
    Graph Transformer Model combining a Transformer Encoder and a Graph Attention Network. [4, 11, 21, 36, 38]
    """
    def __init__(self, dim=512, heads=8, depth=6):
        super().__init__()
        self.encoder = TransformerEncoder(dim, heads, depth) [21, 36]
        self.decoder = GraphAttentionNetwork(dim, heads) # Using custom decoder [21, 36, 39]

    def forward(self, node_features, edge_index):
        """
        Forward pass through the Graph Transformer. [21, 36]
        node_features: Tensor of node features (e.g., abstract embeddings). [36]
        edge_index: Tensor representing graph connectivity in PyTorch Geometric format. [12, 37]
        """
        # Process node features with the Transformer Encoder [21, 36]
        x = self.encoder(node_features) [21, 36]

        # Process graph structure and encoded features with the GAT decoder [21, 36]
        x = self.decoder(x, edge_index) [21, 36]

        # The output 'x' would typically be used for downstream tasks
        # like node classification, link prediction, or feeding into the prediction engine. [4, 7, 9]
        # For the S-index calculation itself, these outputs might contribute indirectly
        # (e.g., to derive certain semantic or structural scores if not calculated directly).
        # However, the core calculate_s_index function above operates directly on the graph.
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Component for processing node features. [4, 11, 14, 21, 38]
    """
    def __init__(self, dim, heads, depth):
        super().__init__()
        # Using standard PyTorch TransformerEncoderLayer [14]
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads) for _ in range(depth)]) [14]

    def forward(self, x):
        """
        Forward pass through the encoder layers. [14]
        x: Input tensor of node features. [14]
        Returns the encoded node features. [14]
        """
        for layer in self.layers: [14]
            x = layer(x) [14]
        return x [14]

class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) layer as a decoder. [4, 11, 14, 21, 37-39]
    Leverages PyTorch Geometric's GATConv. [14, 39]
    """
    def __init__(self, dim, heads=8):
        super(GraphAttentionNetwork, self).__init__()
        # GATConv requires in_channels and out_channels [14, 37]
        # Assuming input and output dimensions are the same 'dim' for simplicity
        # concat=False means the output features from heads are averaged, not concatenated [37]
        self.gat_layer = GATConv(in_channels=dim, out_channels=dim, heads=heads, concat=False) [37]

    def forward(self, x, edge_index):
        """
        Forward pass through the GAT layer. [37]
        x: Input node features tensor. [37]
        edge_index: Tensor representing graph connectivity in PyTorch Geometric format (COO format). [12, 37]
        NOTE: Converting the NetworkX graph's edges to this format is required before calling this forward method. [12]
        """
        # Apply the GAT convolution [37]
        x = self.gat_layer(x, edge_index) [37]
        return x [37]


# --- Example Usage ---
if __name__ == "__main__":
    print("Demonstrating Knowledge Graph construction and S-index calculation:")

    # 1. Initialize the graph builder [19]
    kg_builder = KnowledgeGraphBuilder()

    # 2. Add some example paper data [20, 30]
    # Simulate data fetched from an API like Semantic Scholar [3, 9, 22]
    paper1_data = {
        'id': 'paper1',
        'title': 'First Paper on Scientific Concepts',
        'abstract': 'This paper introduces novel concepts in data sharing and scientific metrics. It supports previous work on the h-index.',
        'citations': ['paperA', 'paperB'], # Cites existing papers
        'year': 2020
    }
    paper2_data = {
        'id': 'paper2',
        'title': 'Advancing Data Sharing Metrics',
        'abstract': 'Building on the ideas from "First Paper on Scientific Concepts", this work proposes a new S-index. It cites paper1.',
        'citations': ['paper1'], # Cites paper1
        'year': 2021
    }
    paper3_data = {
         'id': 'paper3',
         'title': 'Application of the S-Index',
         'abstract': 'We apply the S-index metric to evaluate data sharing practices in biomedical research. This study supports the utility of the S-index proposed in "Advancing Data Sharing Metrics".',
         'citations': ['paper2'], # Cites paper2
         'year': 2022
    }
    # Simulate an existing paper that paper1 cites (as a stub)
    paperA_data = {
        'id': 'paperA',
        'title': 'Previous Work on H-Index',
        'abstract': 'Discusses the h-index.',
        'citations': [],
        'year': 2010
    }


    print("\nAdding initial papers...")
    kg_builder.add_paper(paperA_data) # Add cited papers first if possible to avoid warnings
    kg_builder.add_paper(paper1_data)
    kg_builder.add_paper(paper2_data)
    kg_builder.add_paper(paper3_data) # Adding paper3 as a new update

    current_graph = kg_builder.graph
    # To calculate update metrics, we need a snapshot *before* the latest update.
    # The builder saves the state *before* the most recent add/update call.
    # Let's simulate calculating S-index after adding paper2, then after adding paper3.

    # Re-initialize to show graph evolution steps
    kg_builder_sim = KnowledgeGraphBuilder()

    print("\nSimulation step 1: Adding paperA and paper1")
    kg_builder_sim.add_paper(paperA_data)
    kg_builder_sim.add_paper(paper1_data)
    graph_step1 = kg_builder_sim.graph.copy() # Snapshot after step 1
    s_index_step1 = calculate_s_index(graph_step1)
    print(f"S-index after step 1 (paperA, paper1 added): {s_index_step1:.4f}")

    print("\nSimulation step 2: Adding paper2")
    kg_builder_sim.add_paper(paper2_data)
    graph_step2 = kg_builder_sim.graph.copy() # Snapshot after step 2
    # Calculate S-index including update metrics by passing the previous graph
    s_index_step2 = calculate_s_index(graph_step2, previous_graph=graph_step1)
    print(f"S-index after step 2 (paper2 added): {s_index_step2:.4f}")

    print("\nSimulation step 3: Adding paper3")
    kg_builder_sim.add_paper(paper3_data)
    graph_step3 = kg_builder_sim.graph.copy() # Snapshot after step 3
    # Calculate S-index including update metrics by passing the previous graph
    s_index_step3 = calculate_s_index(graph_step3, previous_graph=graph_step2)
    print(f"S-index after step 3 (paper3 added): {s_index_step3:.4f}")


    # You can inspect the graph
    print(f"\nFinal graph has {current_graph.number_of_nodes()} nodes and {current_graph.number_of_edges()} edges.")
    print("Nodes:", current_graph.nodes(data=True))
    print("Edges:", current_graph.edges(data=True))

    # Example of calculating S-index on the final graph state without update metrics
    final_s_index_static = calculate_s_index(current_graph, previous_graph=None)
    print(f"\nFinal S-index (static calculation): {final_s_index_static:.4f}")

    # Example of using the Update Impact Prediction (requires GraphTransformer)
    # This part is conceptual as the GraphTransformer forward requires edge_index tensor,
    # which isn't automatically generated here from networkx, and the model isn't trained.
    # print("\nDemonstrating conceptual Update Impact Prediction:")
    # try:
    #     # Simulate creating a model (needs appropriate dimensions matching embeddings)
    #     sample_dim = 300 if nlp else 512 # SpaCy 'md' default vector size is 300
    #     dummy_model = GraphTransformer(dim=sample_dim)
    #
    #     # You would need to get node features and convert graph edges to PyG format
    #     # For example:
    #     # node_features_tensor = torch.tensor([data['abstract_embedding'] for node_id, data in graph_step2.nodes(data=True) if data.get('abstract_embedding') is not None], dtype=torch.float)
    #     # edge_list = list(graph_step2.edges())
    #     # edge_index_tensor = torch.tensor(edge_list).t().contiguous()
    #
    #     # This requires significant setup beyond this basic script
    #     print("Conceptual prediction requires converting graph to tensors and a trained model.")
    #     # impact = predict_impact(dummy_model, new_data_causing_graph_step3) # This function isn't fully provided
    #     # print(f"Predicted impact of update: {impact:.4f}")
    # except Exception as e:
    #     print(f"Could not demonstrate impact prediction: {e}")
    #     print("Requires PyTorch, PyTorch Geometric, and conversion of graph data to tensors.")

    print("\nCode execution finished.")
    print("NOTE: Placeholder functions (e.g., for advanced relation extraction) need further implementation.")
    print("NOTE: S-index normalization to 0-100 scale is not implemented in calculate_s_index.")
