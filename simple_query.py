#!/usr/bin/env python3
"""
Simple query examples for Social Services Experience Analytics Platform
Run this after ingesting data to test queries
"""

from query_demo import (
    connect_to_weaviate,
    abstract_text_search,
    filtered_search,
    survey_filtered_search,
    advanced_multi_modal_query
)

def main():
    print("="*80)
    print("SIMPLE QUERY EXAMPLES")
    print("="*80)
    print()
    
    # Connect to Weaviate
    try:
        client = connect_to_weaviate()
        print("✓ Connected to Weaviate\n")
    except Exception as e:
        print(f"✗ Connection error: {e}")
        print("\nMake sure Weaviate is running:")
        print("  docker-compose up -d")
        return
    
    try:
        # Example 1: Simple semantic search
        print("Example 1: Simple Semantic Search")
        print("-" * 80)
        results = abstract_text_search(
            client,
            "hope and resilience in difficult times",
            limit=3
        )
        print()
        
        # Example 2: Filter by tag
        print("Example 2: Filter by Tag")
        print("-" * 80)
        results = filtered_search(
            client,
            "community support and strength",
            filter_property="tag_abstract",
            filter_value="Spiritual Resilience",
            limit=3
        )
        print()
        
        # Example 3: Filter by survey rating
        print("Example 3: Filter by Survey Rating")
        print("-" * 80)
        results = survey_filtered_search(
            client,
            tag="Spiritual Resilience",
            survey_anxiety_max=2,  # Low anxiety
            limit=3
        )
        print()
        
        # Example 4: Advanced query
        print("Example 4: Advanced Multi-Modal Query")
        print("-" * 80)
        results = advanced_multi_modal_query(
            client,
            query_text="financial instability and uncertainty",
            high_volatility=True,
            limit=3
        )
        print()
        
    except Exception as e:
        print(f"Error during queries: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        print("="*80)
        print("Query examples complete!")
        print("="*80)

if __name__ == "__main__":
    main()

