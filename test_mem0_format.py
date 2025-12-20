"""
Test what format mem0 expects
"""
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from util import USER_ID

memory = Memory()

print("=" * 60)
print("Testing Different Input Formats for mem0")
print("=" * 60)

# Test 1: Simple string
print("\n1. Testing with simple string...")
result1 = memory.add("I love pizza and hate broccoli", user_id=USER_ID)
print(f"Result: {result1}")
print(f"Number of memories created: {len(result1.get('results', []))}")

# Test 2: List of strings
print("\n2. Testing with list of strings...")
result2 = memory.add(["My favorite color is blue", "I work as a software engineer"], user_id=USER_ID)
print(f"Result: {result2}")
print(f"Number of memories created: {len(result2.get('results', []))}")

# Test 3: Message format (what we're currently using)
print("\n3. Testing with message format (current approach)...")
messages = [
    {"role": "user", "content": "I have a dog named Max"},
    {"role": "assistant", "content": "That's great! Dogs are wonderful pets."}
]
result3 = memory.add(messages, user_id=USER_ID)
print(f"Result: {result3}")
print(f"Number of memories created: {len(result3.get('results', []))}")

# Now try to retrieve
print("\n" + "=" * 60)
print("Retrieving Memories")
print("=" * 60)

all_mems = memory.get_all(user_id=USER_ID)
print(f"\nTotal memories stored: {len(all_mems.get('results', []))}")

if all_mems.get('results'):
    print("\nMemories:")
    for i, mem in enumerate(all_mems['results']):
        print(f"\n{i+1}. {mem.get('memory', 'N/A')}")
