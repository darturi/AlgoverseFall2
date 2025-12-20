"""
Test the full workflow: add memories in batches, export checkpoint, query from checkpoint
"""
from dotenv import load_dotenv
load_dotenv()

import shutil
from pathlib import Path
from mem0 import Memory
from util import USER_ID
from process_data.transcript_to_msg import load_and_split

# Use absolute paths
project_root = Path.cwd()
chroma_path = project_root / "test_chroma_db"
checkpoint_path = project_root / "data/test/test_checkpoints/checkpoint_test"

print(f"Project root: {project_root}")
print(f"Chroma path: {chroma_path}")
print(f"Checkpoint path: {checkpoint_path}")

# Clean slate
if chroma_path.exists():
    shutil.rmtree(chroma_path)
if checkpoint_path.exists():
    shutil.rmtree(checkpoint_path)

# Step 1: Add memories in batches with explicit path
print('\nSTEP 1: Adding memories in batches')
print('=' * 60)

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": str(chroma_path)
        }
    }
}

memory = Memory.from_config(config)
messages = load_and_split('data/test/test_raw/10_1000060894.txt')

total = 0
for i in range(0, len(messages), 20):
    batch = messages[i:i+20]
    result = memory.add(batch, user_id=USER_ID)
    count = len(result.get('results', []))
    total += count
    print(f'  Batch {i//20+1}: {count} memories')

print(f'\nTotal memories added: {total}')
print(f'Database exists: {chroma_path.exists()}')

# Step 2: Export checkpoint
print('\nSTEP 2: Exporting checkpoint')
print('=' * 60)

shutil.copytree(chroma_path, checkpoint_path)
print(f'✅ Checkpoint exported to: {checkpoint_path}')

# Step 3: Query from checkpoint
print('\nSTEP 3: Querying from checkpoint')
print('=' * 60)

checkpoint_config = {
    'vector_store': {
        'provider': 'chroma',
        'config': {
            'collection_name': 'memories',
            'path': str(checkpoint_path)
        }
    }
}

memory_from_checkpoint = Memory.from_config(checkpoint_config)

# Test query
query = 'What issues is the user dealing with?'
print(f'\nQuery: "{query}"')
results = memory_from_checkpoint.search(query, user_id=USER_ID)

print(f'\nResults found: {len(results.get("results", []))}')

if results.get('results'):
    print('\nTop 5 relevant memories:')
    for i, mem in enumerate(results['results'][:5]):
        print(f'  {i+1}. {mem.get("memory")}')
        print(f'     Score: {mem.get("score", "N/A"):.3f}')

print('\n' + '=' * 60)
print('✅ SUCCESS! Full workflow completed:')
print(f'   - Added {total} memories in batches')
print(f'   - Exported checkpoint successfully')
print(f'   - Queried {len(results.get("results", []))} memories from checkpoint')
print('=' * 60)
