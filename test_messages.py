"""
Test message format with mem0
"""
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from util import USER_ID

memory = Memory()

print("=" * 60)
print("Testing Message Format")
print("=" * 60)

# Test with conversation messages (correct format)
messages = [
    {"role": "user", "content": "Hi, my name is John and I'm a software engineer at Google."},
    {"role": "assistant", "content": "Nice to meet you John! How do you like working at Google?"},
    {"role": "user", "content": "I love it! I've been there for 5 years. I mainly work on search algorithms."}
]

print(f"\nAdding {len(messages)} messages...")
print(f"User ID: {USER_ID}")

result = memory.add(messages, user_id=USER_ID)

print(f"\nResult: {result}")
print(f"Number of memories created: {len(result.get('results', []))}")

if result.get('results'):
    print("\nMemories created:")
    for i, mem in enumerate(result['results']):
        print(f"{i+1}. {mem.get('memory', 'N/A')} (Event: {mem.get('event', 'N/A')})")

# Retrieve
print("\n" + "=" * 60)
print("Retrieving All Memories")
print("=" * 60)

all_mems = memory.get_all(user_id=USER_ID)
print(f"\nTotal memories: {len(all_mems.get('results', []))}")

# Search
print("\n" + "=" * 60)
print("Searching Memories")
print("=" * 60)

search_result = memory.search("What does John do for work?", user_id=USER_ID)
print(f"\nSearch query: 'What does John do for work?'")
print(f"Results found: {len(search_result.get('results', []))}")

if search_result.get('results'):
    print("\nSearch results:")
    for i, mem in enumerate(search_result['results']):
        print(f"{i+1}. {mem.get('memory', 'N/A')}")
