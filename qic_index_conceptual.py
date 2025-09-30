import math

def calculate_quality_score(metadata):
    """
    Calculates the Quality (Q) score based on FAIR principles.
    This is a conceptual demonstration. A real implementation would parse rich metadata.
    
    Args:
        metadata (dict): A dictionary containing FAIR-related metadata.
        
    Returns:
        float: The calculated Q score (0-10).
    """
    # Weights for each FAIR component
    weights = {'findable': 0.3, 'accessible': 0.3, 'interoperable': 0.2, 'reusable': 0.2}
    
    q_score = 0.0
    
    # Example: Score for Findability (0-10)
    # A real system would check for DOI, rich metadata, etc.
    q_score += metadata.get('findability_score', 0) * weights['findable']
    
    # Example: Score for Accessibility (0-10)
    # Checks for open protocols, no access barriers.
    q_score += metadata.get('accessibility_score', 0) * weights['accessible']
    
    # Example: Score for Interoperability (0-10)
    # Checks for standard formats, vocabularies.
    q_score += metadata.get('interoperability_score', 0) * weights['interoperable']

    # Example: Score for Reusability (0-10)
    # Checks for a clear, machine-readable license.
    q_score += metadata.get('reusability_score', 0) * weights['reusable']
    
    return q_score

def calculate_impact_score(reuse_events):
    """
    Calculates the Impact (I) score based on reuse.
    I = 1 + ln(1 + number_of_reuse_events)
    
    Args:
        reuse_events (int): The number of times the data has been cited or reused.
        
    Returns:
        float: The calculated I score.
    """
    if reuse_events < 0:
        return 1.0 # Reuse cannot be negative
        
    # The formula from the manuscript, simplified for this demonstration.
    # A full implementation would have weighted reuse events.
    return 1 + math.log(1 + reuse_events)

def calculate_collaboration_score(num_authors, num_institutions):
    """
    Calculates the Collaboration (C) score.
    C = (1 + ln(N_authors)) * (1 + 0.5 * ln(N_institutions))

    Args:
        num_authors (int): The number of contributing authors.
        num_institutions (int): The number of unique contributing institutions.
        
    Returns:
        float: The calculated C score.
    """
    if num_authors < 1 or num_institutions < 1:
        return 1.0 # Default score if inputs are invalid
        
    author_component = 1 + math.log(num_authors)
    institution_component = 1 + (0.5 * math.log(num_institutions))
    
    return author_component * institution_component

def calculate_qic_score(data_object):
    """
    Calculates the final QIC-Index score for a single data object.
    
    Args:
        data_object (dict): A dictionary representing the data object with all necessary info.
        
    Returns:
        float: The final QIC score (s_j).
    """
    quality_score = calculate_quality_score(data_object['metadata'])
    impact_score = calculate_impact_score(data_object['reuse_events'])
    collaboration_score = calculate_collaboration_score(data_object['num_authors'], data_object['num_institutions'])
    
    final_score = quality_score * impact_score * collaboration_score
    
    print(f"--- Calculating Score for: {data_object['name']} ---")
    print(f"  Quality (Q) Score: {quality_score:.2f}")
    print(f"  Impact (I) Score: {impact_score:.2f}")
    print(f"  Collaboration (C) Score: {collaboration_score:.2f}")
    print(f"  Final QIC Score (s_j): {final_score:.2f}")
    print("-------------------------------------------------")
    
    return final_score


if __name__ == '__main__':
    # --- Example Demonstrations ---
    
    # Persona 1: Dr. Al-Jamil, The Early-Career Collaborator
    # A dataset from a large, multi-institution consortium. High quality and collaboration, but still new.
    al_jamil_dataset = {
        'name': "Consortium Genomics Dataset",
        'metadata': {
            'findability_score': 9.0,
            'accessibility_score': 8.5,
            'interoperability_score': 7.0,
            'reusability_score': 9.0
        },
        'reuse_events': 15,
        'num_authors': 25,
        'num_institutions': 10
    }
    
    # Persona 2: Dr. Singh, The Traditional PI
    # A meticulously curated dataset from a single lab that has been very widely used.
    singh_dataset = {
        'name': "Longitudinal Cohort Study Data",
        'metadata': {
            'findability_score': 9.5,
            'accessibility_score': 9.0,
            'interoperability_score': 8.0,
            'reusability_score': 9.5
        },
        'reuse_events': 350,
        'num_authors': 5,
        'num_institutions': 1
    }
    
    # Calculate the total QIC-Index for an author by summing the scores of their data objects
    author_total_qic_index = 0
    
    # In this example, let's say Dr. Al-Jamil has one primary dataset contribution
    author_total_qic_index += calculate_qic_score(al_jamil_dataset)
    
    print(f"\nDr. Al-Jamil's Total QIC-Index (S_i): {author_total_qic_index:.2f}\n")
    
    # Reset for Dr. Singh
    author_total_qic_index = 0
    author_total_qic_index += calculate_qic_score(singh_dataset)
    print(f"\nDr. Singh's Total QIC-Index (S_i): {author_total_qic_index:.2f}\n")
