"""
Debug script to test if memories are properly stored and retrievable
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env file with OPENAI_API_KEY

from mem0 import Memory
from util import USER_ID
import os

# Test 1: Check if memories exist in checkpoint
print("=" * 60)
print("TEST 1: Checking checkpoint memories")
print("=" * 60)

checkpoint_path = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/test/test_checkpoints/checkpoint_1"

if os.path.exists(checkpoint_path):
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "memories",
                "path": checkpoint_path
            }
        }
    }

    memory = Memory.from_config(config)

    # Try to get all memories for user_alpha
    print(f"\nSearching for memories with USER_ID: {USER_ID}")

    # Test with a very generic query
    results = memory.search("therapy session patient", user_id=USER_ID)

    print(f"\nSearch results:")
    print(f"Number of results: {len(results.get('results', []))}")

    if results.get('results'):
        print("\nFound memories:")
        for i, mem in enumerate(results['results'][:5]):  # Show first 5
            print(f"\n{i+1}. {mem.get('memory', 'N/A')}")
            print(f"   Score: {mem.get('score', 'N/A')}")
    else:
        print("\n⚠️ NO MEMORIES FOUND!")
        print("\nPossible reasons:")
        print("1. Data was stored with different user_id")
        print("2. Embedder configuration mismatch")
        print("3. Data wasn't actually stored")

    # Try to get ALL memories (no filtering)
    print("\n" + "=" * 60)
    print("TEST 2: Attempting to retrieve ALL memories")
    print("=" * 60)

    try:
        all_memories = memory.get_all(user_id=USER_ID)
        print(f"\nTotal memories for {USER_ID}: {len(all_memories.get('results', []))}")

        if all_memories.get('results'):
            print("\nFirst 3 memories:")
            for i, mem in enumerate(all_memories['results'][:3]):
                print(f"\n{i+1}. {mem}")
    except Exception as e:
        print(f"Error getting all memories: {e}")

else:
    print(f"❌ Checkpoint path not found: {checkpoint_path}")

print("\n" + "=" * 60)
print("TEST 3: Check main memory database")
print("=" * 60)

main_mem_path = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/Mem0_Logic/chroma_db"

if os.path.exists(main_mem_path):
    config_main = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "memories",
                "path": main_mem_path
            }
        }
    }

    memory_main = Memory.from_config(config_main)

    print(f"\nSearching main database with USER_ID: {USER_ID}")
    results_main = memory_main.search("therapy session", user_id=USER_ID)

    print(f"Number of results in main DB: {len(results_main.get('results', []))}")

    if results_main.get('results'):
        print("\nFound memories in main DB:")
        for i, mem in enumerate(results_main['results'][:3]):
            print(f"\n{i+1}. {mem.get('memory', 'N/A')[:100]}...")

    # Get all
    try:
        all_main = memory_main.get_all(user_id=USER_ID)
        print(f"\nTotal memories in main DB for {USER_ID}: {len(all_main.get('results', []))}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"❌ Main memory path not found: {main_mem_path}")
