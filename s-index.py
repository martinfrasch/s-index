# s_index.py

import networkx as nx
import numpy as np
import spacy
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- BioMedLM (PubMedBERT) and spaCy Model Loading ---
BIOMEDLM_MODEL_NAME = "microsoft/BioMedLM-PubMedBERT-base-uncased-abstract-fulltext" #

try:
    biomedlm_tokenizer = AutoTokenizer.from_pretrained(BIOMEDLM_MODEL_NAME)
    biomedlm_model = AutoModel.from_pretrained(BIOMEDLM_MODEL_NAME)
    print(f"BioMedLM model '{BIOMEDLM_MODEL_NAME}' loaded successfully.")
    BIOMEDLM_EMBEDDING_DIM = biomedlm_model.config.hidden_size # Typically 768
except Exception as e:
    print(f"Error loading BioMedLM model '{BIOMEDLM_MODEL_NAME}': {e}")
    print("Semantic similarity and BioMedLM-based embeddings will not be available.")
    biomedlm_tokenizer = None
    biomedlm_model = None
    BIOMEDLM_EMBEDDING_DIM = 768 # Default fallback

# Load spaCy for tasks like relation extraction (if still needed)
try:
    nlp = spacy.load("en_core_web_md")
    print("SpaCy model 'en_core_web_md' loaded successfully (for relation extraction, etc.).")
except OSError:
    print("SpaCy model 'en_core_web_md' not found. Please download it by running:")
    print("python -m spacy download en_core_web_md")
    nlp = None
    print("SpaCy-dependent features like relation extraction may not work.")


# --- 1. S Index Calculation Functions ---

STRUCTURAL_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.30
KNOWLEDGE_UPDATE_WEIGHT = 0.15
DATA_SHARING_WEIGHT = 0.15
TARGET_NODE_COUNT = 80000000
TARGET_DATASET_COUNT_FOR_MAX_SCORE = 10

def get_biomedlm_embedding(text):
    """
    Generates an embedding for the given text using BioMedLM.
    Uses mean pooling of the last hidden states.
    """
    if biomedlm_tokenizer is None or biomedlm_model is None or not text:
        return None
    try:
        inputs = biomedlm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = biomedlm_model(**inputs)
        # Mean pooling: average the token embeddings from the last hidden layer
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error generating BioMedLM embedding: {e}")
        return None

def calculate_structural_metrics(graph):
    """
    Calculates various structural metrics for the graph.
    Returns a dictionary of structural metrics.
    """
    metrics = {}
    if not graph or graph.number_of_nodes() == 0:
        return {
            'connectivity': 0, 'centrality': 0, 'pagerank': 0,
            'betweenness': 0, 'clustering': 0
        }
    try:
        if nx.is_strongly_connected(graph):
             metrics['connectivity'] = nx.average_node_connectivity(graph)
        elif nx.is_weakly_connected(graph):
             metrics['connectivity'] = nx.average_node_connectivity(graph.to_undirected())
        else:
            metrics['connectivity'] = 0
    except Exception:
         metrics['connectivity'] = 0

    metrics['centrality'] = np.mean(list(nx.degree_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0
    metrics['pagerank'] = np.mean(list(nx.pagerank(graph, max_iter=500).values())) if graph.number_of_nodes() > 0 else 0 # Added max_iter for convergence
    metrics['betweenness'] = np.mean(list(nx.betweenness_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0
    metrics['clustering'] = np.mean(list(nx.clustering(graph).values())) if graph.number_of_nodes() > 0 else 0
    return metrics

def calculate_semantic_similarity_from_embeddings(embedding1, embedding2):
    """
    Calculates the cosine similarity between two pre-computed embeddings.
    Returns a similarity score between 0 and 1.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    try:
        # Reshape for cosine_similarity function (expects 2D arrays)
        sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0, 0]
        return (sim + 1) / 2 # Normalize from [-1, 1] to [0, 1] if needed, or ensure embeddings are positive.
                               # BERT embeddings are not guaranteed to be positive. Cosine similarity is typically -1 to 1.
                               # For S-Index, a 0-1 range is usually expected for component scores.
                               # Let's assume direct cosine similarity, and normalization happens at the S-index avg_semantic level.
                               # Or, ensure this score is within a reasonable positive range for averaging.
                               # Standard cosine similarity is often used directly. Forcing to 0-1 if it becomes negative.
        return max(0, sim) # Ensure score is non-negative
    except Exception as e:
        print(f"Error calculating semantic similarity from embeddings: {e}")
        return 0.0

def calculate_semantic_metrics(graph):
    """
    Calculates semantic metrics using BioMedLM embeddings stored in graph nodes.
    Returns a dictionary of semantic metrics.
    """
    semantic_scores = []
    if biomedlm_model is None: # Check if BioMedLM is available
        completeness = len(graph.nodes) / TARGET_NODE_COUNT if TARGET_NODE_COUNT > 0 else 0
        return {'completeness': min(completeness, 1.0), 'similarity': 0.0}

    for u, v in graph.edges():
        # Assumes 'biomedlm_embedding' is stored in node attributes
        embedding1 = graph.nodes[u].get('biomedlm_embedding')
        embedding2 = graph.nodes[v].get('biomedlm_embedding')
        
        if embedding1 is not None and embedding2 is not None:
            similarity = calculate_semantic_similarity_from_embeddings(embedding1, embedding2)
            semantic_scores.append(similarity)

    avg_similarity = np.mean(semantic_scores) if semantic_scores else 0.0
    completeness_raw = len(graph.nodes) / TARGET_NODE_COUNT if TARGET_NODE_COUNT > 0 else 0
    metrics = {
        'completeness': min(completeness_raw, 1.0),
        'similarity': avg_similarity
    }
    return metrics

def calculate_knowledge_update_metrics(graph, previous_graph):
    """
    Calculates knowledge update metrics based on changes to the graph.
    Returns a dictionary of update metrics.
    """
    if not previous_graph or previous_graph.number_of_nodes() == 0:
        return {'integration_rate': 1.0, 'centrality_shift': 0.0, 'temporal_consistency': 1.0}

    added_nodes = set(graph.nodes) - set(previous_graph.nodes)
    if not added_nodes and graph.number_of_nodes() == previous_graph.number_of_nodes():
        integration_rate = 0.0
    elif graph.number_of_nodes() > 0:
        integration_rate = len(added_nodes) / graph.number_of_nodes()
    else:
        integration_rate = 0.0
        
    centrality_before = np.mean(list(nx.degree_centrality(previous_graph).values())) if previous_graph.number_of_nodes() > 0 else 0.0
    centrality_after = np.mean(list(nx.degree_centrality(graph).values())) if graph.number_of_nodes() > 0 else 0.0
    centrality_shift = abs(centrality_after - centrality_before)
    normalized_centrality_shift = min(centrality_shift, 1.0)
    temporal_consistency = 1.0 # Placeholder

    metrics = {
        'integration_rate': integration_rate,
        'centrality_shift': 1.0 - normalized_centrality_shift,
        'temporal_consistency': temporal_consistency
    }
    return metrics

def calculate_data_sharing_metrics(graph, researcher_id=None):
    """
    Calculates data sharing metrics.
    Returns a dictionary including normalized score and dataset count.
    """
    num_shared_datasets = 0
    if researcher_id:
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'dataset' and data.get('researcher_id') == researcher_id:
                num_shared_datasets += 1
    else:
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'dataset':
                num_shared_datasets += 1
    
    normalized_score = min(num_shared_datasets / TARGET_DATASET_COUNT_FOR_MAX_SCORE, 1.0) if TARGET_DATASET_COUNT_FOR_MAX_SCORE > 0 else 0.0
    return {
        'data_sharing_score': normalized_score,
        'dataset_count': num_shared_datasets
    }

def calculate_s_index(structural_score, semantic_score, knowledge_update_score, data_sharing_score_value):
    """
    Calculates the S Index (0-100).
    """
    weighted_score = (STRUCTURAL_WEIGHT * structural_score +
                      SEMANTIC_WEIGHT * semantic_score +
                      KNOWLEDGE_UPDATE_WEIGHT * knowledge_update_score +
                      DATA_SHARING_WEIGHT * data_sharing_score_value)
    return weighted_score * 100

# --- 2. Knowledge Graph Construction Class ---

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.previous_graph = nx.MultiDiGraph()

    def _update_previous_graph(self):
        self.previous_graph = self.graph.copy()

    def add_paper(self, paper_data):
        self._update_previous_graph()
        paper_id = paper_data.get('id')
        if paper_id is None: return

        abstract_text = paper_data.get('abstract', '')
        # Generate and store BioMedLM embedding
        embedding = get_biomedlm_embedding(abstract_text) #

        node_attributes = {'type': 'paper', 'biomedlm_embedding': embedding}
        node_attributes.update(paper_data)
        self.graph.add_node(paper_id, **node_attributes)

        for citation_id in paper_data.get('citations', []):
            self.create_relationship(paper_id, citation_id, 'CITES')

        # Relation extraction can still use spaCy if nlp model is loaded
        if nlp and abstract_text:
            for relation in self.extract_relations_from_abstract_spacy(abstract_text, paper_id):
                 self.create_relationship(relation['source'], relation['target'], relation['type'])

    def add_dataset(self, dataset_data):
        self._update_previous_graph()
        dataset_id = dataset_data.get('id')
        if dataset_id is None: return
        
        node_attributes = {'type': 'dataset'}
        node_attributes.update(dataset_data)
        # Datasets typically don't have abstracts for embedding, but could have descriptions
        description = dataset_data.get('description', '')
        if description:
            node_attributes['biomedlm_embedding'] = get_biomedlm_embedding(description)

        self.graph.add_node(dataset_id, **node_attributes)

    def create_relationship(self, source, target, relation_type):
        if relation_type == 'CITES' and not self.graph.has_node(target):
            self.graph.add_node(target, type='paper_stub', title=f"Cited Paper {target}")

        if source in self.graph.nodes and target in self.graph.nodes:
             self.graph.add_edge(source, target, type=relation_type)
        else:
             print(f"Warning: Skipping edge {source} -> {target} ({relation_type}). Node(s) not found.")

    def extract_relations_from_abstract_spacy(self, abstract, paper_id):
        """
        Placeholder for spaCy-based relation extraction.
        Requires spaCy 'nlp' model to be loaded.
        """
        relations = []
        if nlp is None or not abstract: # Check if spaCy is available
            return relations
        try:
            doc = nlp(abstract)
            # Simplified example logic (remains placeholder)
            for sent in doc.sents:
                for token in sent:
                    if token.text.lower() == 'supports' and token.head.i != token.i: # Basic check
                        target_node_id = self.find_node_id_by_title_globally(token.head.text) # Highly simplified target finding
                        if target_node_id:
                            relations.append({'source': paper_id, 'target': target_node_id, 'type': 'SUPPORTS'})
        except Exception as e:
             print(f"Error during spaCy relation extraction: {e}")
        return relations
    
    def find_node_id_by_title_globally(self, title_text_fragment):
        """ Very basic search for a node by title fragment. Needs improvement. """
        for node_id, data in self.graph.nodes(data=True):
            if data.get('title') and title_text_fragment.lower() in data.get('title').lower():
                return node_id
        return None

    def update_node(self, node_id, new_data):
        if node_id in self.graph.nodes:
            self._update_previous_graph()
            current_node_data = self.graph.nodes[node_id]
            current_node_data.update(new_data)

            if 'abstract' in new_data:
                abstract_text = new_data.get('abstract','')
                current_node_data['biomedlm_embedding'] = get_biomedlm_embedding(abstract_text) #
                if nlp and abstract_text: # Re-extract relations if spaCy is available
                    # Consider logic for removing old relations
                    for relation in self.extract_relations_from_abstract_spacy(abstract_text, node_id):
                         self.create_relationship(relation['source'], relation['target'], relation['type'])
        else:
            print(f"Warning: Node {node_id} not found. Cannot update.")


# --- 3. Graph Transformer Architecture Components ---
# (Kept as per original structure, uses PyTorch)

class GraphTransformer(nn.Module):
    def __init__(self, dim=768, heads=8, depth=6): # Default dim to BioMedLM's typical output
        super().__init__()
        self.encoder = TransformerEncoder(dim, heads, depth)
        self.decoder = GraphAttentionNetwork(dim, heads)

    def forward(self, node_features, edge_index):
        x = self.encoder(node_features)
        x = self.decoder(x, edge_index)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True) for _ in range(depth)])

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        for layer in self.layers: x = layer(x)
        if x.ndim == 3 and x.shape[1] == 1: x = x.squeeze(1)
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, dim, heads=8):
        super(GraphAttentionNetwork, self).__init__()
        self.gat_layer = GATConv(in_channels=dim, out_channels=dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        return self.gat_layer(x, edge_index)

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- SAGE-K S-Index Calculation Demo (with BioMedLM focus) ---")

    kg_builder = KnowledgeGraphBuilder()
    researcher1_id = "researcher_Alpha"

    # Sample data
    paperA_data = {'id': 'paperA', 'title': 'Foundations of H-Index', 'abstract': 'This work explores the foundational concepts of the h-index.', 'citations': [], 'year': 2010, 'researcher_id': researcher1_id}
    paper1_data = {'id': 'paper1', 'title': 'Novel Data Sharing Paradigms', 'abstract': 'We introduce innovative paradigms for scientific data sharing, building upon h-index concepts. This research supports open science.', 'citations': ['paperA'], 'year': 2020, 'researcher_id': researcher1_id}
    dataset1_data = {'id': 'dataset1', 'title': 'Data for Sharing Paradigms Paper', 'researcher_id': researcher1_id, 'year': 2020, 'description': 'Raw and processed data for the 2020 paper on data sharing paradigms.'}
    
    print("\nStep 1: Adding initial paper (PaperA)")
    kg_builder.add_paper(paperA_data)
    graph_s0 = kg_builder.graph.copy()
    prev_graph_s0 = None

    structural_s0 = np.mean(list(calculate_structural_metrics(graph_s0).values()))
    semantic_s0 = np.mean(list(calculate_semantic_metrics(graph_s0).values()))
    update_s0 = np.mean(list(calculate_knowledge_update_metrics(graph_s0, prev_graph_s0).values()))
    ds_info_s0 = calculate_data_sharing_metrics(graph_s0, researcher1_id)
    s_index_s0 = calculate_s_index(structural_s0, semantic_s0, update_s0, ds_info_s0['data_sharing_score'])
    print(f"S-Index (PaperA): {s_index_s0:.2f} (DataSets: {ds_info_s0['dataset_count']})")
    print(f"  Scores (Struct:{structural_s0:.2f} Sem:{semantic_s0:.2f} Upd:{update_s0:.2f} DS:{ds_info_s0['data_sharing_score']:.2f})")

    prev_graph_s1 = graph_s0.copy()
    print("\nStep 2: Adding Paper1 and Dataset1")
    kg_builder.add_paper(paper1_data)
    kg_builder.add_dataset(dataset1_data)
    graph_s1 = kg_builder.graph.copy()

    structural_s1 = np.mean(list(calculate_structural_metrics(graph_s1).values()))
    semantic_s1 = np.mean(list(calculate_semantic_metrics(graph_s1).values())) # Will use BioMedLM embeddings
    update_s1 = np.mean(list(calculate_knowledge_update_metrics(graph_s1, prev_graph_s1).values()))
    ds_info_s1 = calculate_data_sharing_metrics(graph_s1, researcher1_id)
    s_index_s1 = calculate_s_index(structural_s1, semantic_s1, update_s1, ds_info_s1['data_sharing_score'])
    print(f"S-Index (Paper1, Dataset1): {s_index_s1:.2f} (DataSets: {ds_info_s1['dataset_count']})")
    print(f"  Scores (Struct:{structural_s1:.2f} Sem:{semantic_s1:.2f} Upd:{update_s1:.2f} DS:{ds_info_s1['data_sharing_score']:.2f})")


    print(f"\nFinal graph: {kg_builder.graph.number_of_nodes()} nodes, {kg_builder.graph.number_of_edges()} edges.")
    if biomedlm_model is None:
        print("\nNOTE: BioMedLM model was not loaded. Semantic scores are based on fallbacks.")
    if nlp is None:
        print("NOTE: SpaCy model was not loaded. Relation extraction features are disabled.")
    
    print("\nCode execution finished.")
    print("This demo uses BioMedLM for embeddings if available. Ensure 'transformers' and 'sklearn' are installed.")
