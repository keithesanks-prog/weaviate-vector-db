#!/usr/bin/env python3
"""
NLM (National Library of Medicine) Integration Module
Enriches client experience data with authoritative health information from NLM services.

References:
- MedlinePlus Web Service: https://medlineplus.gov/about/developers/webservices/
- PubMed API: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- UMLS API: https://www.nlm.nih.gov/research/umls/index.html
"""

import requests
import json
import re
from typing import Dict, List, Optional, Any
from urllib.parse import quote

# NLM API Base URLs
MEDLINEPLUS_BASE = "https://wsearch.nlm.nih.gov/ws/query"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE}/efetch.fcgi"

# Health-related keywords that might appear in client experiences
HEALTH_KEYWORDS = [
    'anxiety', 'depression', 'stress', 'pain', 'illness', 'disease', 'medication',
    'doctor', 'hospital', 'clinic', 'treatment', 'therapy', 'mental health',
    'chronic', 'diabetes', 'hypertension', 'asthma', 'cancer', 'heart',
    'sleep', 'fatigue', 'headache', 'migraine', 'arthritis', 'obesity',
    'substance', 'addiction', 'trauma', 'ptsd', 'bipolar', 'schizophrenia'
]

def extract_health_keywords(text: str) -> List[str]:
    """
    Extract health-related keywords from text that might be relevant for NLM queries.
    
    Args:
        text: The text snippet to analyze
        
    Returns:
        List of potential health-related keywords found in the text
    """
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in HEALTH_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Also look for common health phrases
    health_phrases = [
        r'\b(mental health|physical health|health care|healthcare)\b',
        r'\b(medical|medication|prescription|diagnosis)\b',
        r'\b(symptom|condition|disorder|syndrome)\b'
    ]
    
    for pattern in health_phrases:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_keywords.extend(matches)
    
    return list(set(found_keywords))  # Remove duplicates

def query_medlineplus(topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Query MedlinePlus for consumer health information.
    Note: MedlinePlus doesn't have a public REST API, so we provide direct links.
    
    Args:
        topic: Health topic or keyword to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing MedlinePlus article information
    """
    try:
        # Since MedlinePlus doesn't have a public API, we'll provide direct search links
        # and common health topic links based on keywords
        results = []
        
        # Map common keywords to MedlinePlus topics
        topic_mapping = {
            'anxiety': 'anxiety',
            'depression': 'depression',
            'stress': 'stress',
            'mental health': 'mentalhealth',
            'pain': 'pain',
            'sleep': 'sleepdisorders',
            'diabetes': 'diabetes',
            'hypertension': 'highbloodpressure',
            'asthma': 'asthma',
            'heart': 'heartdiseases',
            'fatigue': 'fatigue',
            'headache': 'headache',
            'migraine': 'migraine',
            'arthritis': 'arthritis',
            'obesity': 'obesity',
            'addiction': 'substanceabuse',
            'trauma': 'trauma',
            'ptsd': 'ptsd'
        }
        
        # Find matching topic
        topic_lower = topic.lower()
        mapped_topic = None
        for key, value in topic_mapping.items():
            if key in topic_lower:
                mapped_topic = value
                break
        
        if mapped_topic:
            results.append({
                'title': f'MedlinePlus: {topic.title()}',
                'url': f'https://medlineplus.gov/{mapped_topic}.html',
                'source': 'MedlinePlus',
                'type': 'Consumer Health Information'
            })
        
        # Always provide a search link
        search_url = f"https://medlineplus.gov/search/?query={quote(topic)}"
        results.append({
            'title': f'Search MedlinePlus for: {topic}',
            'url': search_url,
            'source': 'MedlinePlus',
            'type': 'Search Results'
        })
        
        return results[:max_results]
    except Exception as e:
        print(f"Warning: MedlinePlus query failed for '{topic}': {e}")
        return []

def query_pubmed(keywords: List[str], max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Query PubMed for relevant research articles using the E-utilities API.
    
    Args:
        keywords: List of keywords to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing PubMed article information
    """
    if not keywords:
        return []
    
    try:
        # Build search query - combine keywords with AND for more focused results
        # Focus on social determinants of health, mental health, poverty-related research
        query_terms = []
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            # Add context for social services relevance
            if keyword in ['anxiety', 'stress', 'depression', 'mental health']:
                query_terms.append(f'({keyword}[Title/Abstract] AND (poverty OR "social determinants" OR "health disparities"))')
            else:
                query_terms.append(f'{keyword}[Title/Abstract]')
        
        query = ' OR '.join(query_terms)
        
        # Step 1: Search PubMed
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        search_response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=10)
        search_response.raise_for_status()
        
        search_data = search_response.json()
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            # If no results, try a simpler search
            simple_query = ' OR '.join(keywords[:2])
            search_params['term'] = simple_query
            search_response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=10)
            search_data = search_response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return []
        
        # Step 2: Fetch article details
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids[:max_results]),
            'retmode': 'xml'  # Use XML for better parsing
        }
        
        fetch_response = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=10)
        fetch_response.raise_for_status()
        
        # Parse XML response (simplified)
        xml_content = fetch_response.text
        articles = []
        
        # Extract article information using regex (in production, use proper XML parser)
        pmid_pattern = r'<PMID[^>]*>(\d+)</PMID>'
        title_pattern = r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>'
        author_pattern = r'<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>'
        abstract_pattern = r'<AbstractText[^>]*>(.*?)</AbstractText>'
        
        pmids_found = re.findall(pmid_pattern, xml_content)
        titles = re.findall(title_pattern, xml_content, re.DOTALL)
        authors_list = re.findall(author_pattern, xml_content)
        abstracts = re.findall(abstract_pattern, xml_content, re.DOTALL)
        
        for i, pmid in enumerate(pmids_found[:max_results]):
            title = titles[i] if i < len(titles) else 'No title available'
            # Clean up title
            title = re.sub(r'<[^>]+>', '', title).strip()
            
            # Get authors
            author_names = []
            if authors_list:
                start_idx = i * 3  # Approximate author grouping
                for j in range(start_idx, min(start_idx + 3, len(authors_list))):
                    last, first = authors_list[j]
                    author_names.append(f"{last}, {first}")
            
            abstract = abstracts[i] if i < len(abstracts) else ''
            abstract = re.sub(r'<[^>]+>', '', abstract).strip()
            if len(abstract) > 500:
                abstract = abstract[:500] + '...'
            
            articles.append({
                'pmid': pmid,
                'title': title,
                'authors': ', '.join(author_names) if author_names else 'Unknown',
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'source': 'PubMed',
                'type': 'Research Article'
            })
        
        return articles
    except Exception as e:
        print(f"Warning: PubMed query failed: {e}")
        # Return a search link as fallback
        if keywords:
            search_query = '+'.join(keywords[:2])
            return [{
                'title': f'Search PubMed for: {", ".join(keywords[:2])}',
                'url': f'https://pubmed.ncbi.nlm.nih.gov/?term={quote(search_query)}',
                'source': 'PubMed',
                'type': 'Search Results',
                'authors': '',
                'abstract': ''
            }]
        return []

def enrich_experience_with_nlm(text_snippet: str, tag_abstract: str = None, 
                               survey_anxiety: int = None) -> Dict[str, Any]:
    """
    Enrich a client experience entry with relevant NLM information.
    
    Args:
        text_snippet: The text snippet from the experience
        tag_abstract: The abstract tag (e.g., "Cognitive Load", "Emotional Exhaustion")
        survey_anxiety: Self-reported anxiety level (1-5)
        
    Returns:
        Dictionary containing enriched information from NLM sources
    """
    enrichment = {
        'health_keywords': [],
        'medlineplus_articles': [],
        'pubmed_articles': [],
        'suggested_topics': []
    }
    
    # Extract health keywords
    health_keywords = extract_health_keywords(text_snippet)
    enrichment['health_keywords'] = health_keywords
    
    # If anxiety is mentioned or survey shows high anxiety, add mental health topics
    if survey_anxiety and survey_anxiety >= 4:
        if 'anxiety' not in health_keywords:
            health_keywords.append('anxiety')
        if 'stress' not in health_keywords:
            health_keywords.append('stress management')
    
    # Map abstract tags to health topics
    tag_to_topic = {
        'Cognitive Load': 'cognitive health',
        'Emotional Exhaustion': 'stress management',
        'Systemic Frustration': 'mental health',
        'Hopelessness': 'depression',
        'Social Isolation': 'mental health',
        'Time Poverty': 'stress management'
    }
    
    if tag_abstract:
        for tag, topic in tag_to_topic.items():
            if tag in tag_abstract:
                if topic not in health_keywords:
                    health_keywords.append(topic)
    
    # Query MedlinePlus for consumer health information
    if health_keywords:
        # Use the first relevant keyword
        primary_keyword = health_keywords[0]
        medline_results = query_medlineplus(primary_keyword, max_results=2)
        enrichment['medlineplus_articles'] = medline_results
    
    # Query PubMed for research articles (use broader search)
    if health_keywords:
        pubmed_results = query_pubmed(health_keywords, max_results=2)
        enrichment['pubmed_articles'] = pubmed_results
    
    # Generate suggested topics
    enrichment['suggested_topics'] = list(set(health_keywords[:5]))
    
    return enrichment

def get_nlm_enrichment_for_experience(experience_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get NLM enrichment for a complete experience object.
    
    Args:
        experience_data: Dictionary containing experience properties
        
    Returns:
        Enriched experience data with NLM information
    """
    text = experience_data.get('text', '') or experience_data.get('text_snippet', '')
    tag = experience_data.get('tag_abstract', '')
    anxiety = experience_data.get('survey_anxiety')
    
    enrichment = enrich_experience_with_nlm(text, tag, anxiety)
    
    return {
        'original_data': experience_data,
        'nlm_enrichment': enrichment
    }

# Example usage
if __name__ == "__main__":
    # Test with a sample experience
    sample_experience = {
        'text': 'I feel completely alone in this. I have no one to talk to about money, shame keeps me silent.',
        'tag_abstract': 'Social Isolation',
        'survey_anxiety': 5
    }
    
    print("Testing NLM Integration...")
    print("=" * 60)
    
    enriched = get_nlm_enrichment_for_experience(sample_experience)
    
    print(f"Health Keywords Found: {enriched['nlm_enrichment']['health_keywords']}")
    print(f"\nMedlinePlus Articles: {len(enriched['nlm_enrichment']['medlineplus_articles'])}")
    for article in enriched['nlm_enrichment']['medlineplus_articles']:
        print(f"  - {article['title']}")
        print(f"    URL: {article['url']}")
    
    print(f"\nPubMed Articles: {len(enriched['nlm_enrichment']['pubmed_articles'])}")
    for article in enriched['nlm_enrichment']['pubmed_articles']:
        print(f"  - {article['title']}")
        print(f"    Authors: {article['authors']}")
        print(f"    URL: {article['url']}")

