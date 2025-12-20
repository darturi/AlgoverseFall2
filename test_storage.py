"""
Test memory storage to see what's happening
"""
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from process_data.transcript_to_msg import load_and_split
from util import USER_ID, BASE_MEM_CONFIG

print("=" * 60)
print("Testing Memory Storage")
print("=" * 60)

# Parse the transcript
transcript_path = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/test/test_raw/10_1000060894.txt"

print(f"\n1. Loading and parsing transcript...")
messages = load_and_split(transcript_path)

print(f"\nParsed {len(messages)} messages")
print(f"Message type: {type(messages)}")

if messages:
    print(f"\nFirst 3 messages:")
    for i, msg in enumerate(messages[:3]):
        print(f"\n{i+1}. Role: {msg['role']}")
        print(f"   Content: {msg['content'][:100]}...")

# Now try to add to memory
print("\n" + "=" * 60)
print("2. Adding to Memory")
print("=" * 60)

print(f"\nCreating memory with config:")
print(f"  Path: {BASE_MEM_CONFIG['vector_store']['config']['path']}")
print(f"  Collection: {BASE_MEM_CONFIG['vector_store']['config']['collection_name']}")
print(f"  User ID: {USER_ID}")

memory = Memory.from_config(BASE_MEM_CONFIG)

print(f"\nAdding messages to memory...")

try:
    result = memory.add(messages, user_id=USER_ID)
    print(f"\n✅ Successfully added!")
    print(f"Result: {result}")
except Exception as e:
    print(f"\n❌ Error adding to memory: {e}")
    import traceback
    traceback.print_exc()

# Try to retrieve
print("\n" + "=" * 60)
print("3. Trying to Retrieve Memories")
print("=" * 60)

print(f"\nSearching for 'therapy session'...")
search_results = memory.search("therapy session", user_id=USER_ID)

print(f"Found {len(search_results.get('results', []))} results")

if search_results.get('results'):
    print("\nFirst result:")
    print(search_results['results'][0])
else:
    print("\n⚠️ No results found!")

# Get all memories
print(f"\nGetting ALL memories for {USER_ID}...")
all_memories = memory.get_all(user_id=USER_ID)
print(f"Total memories: {len(all_memories.get('results', []))}")

if all_memories.get('results'):
    print("\nFirst 2 memories:")
    for i, mem in enumerate(all_memories['results'][:2]):
        print(f"\n{i+1}. {mem}")
