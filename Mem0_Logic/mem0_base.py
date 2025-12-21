import os
from dotenv import load_dotenv
from mem0 import Memory
from process_data.transcript_to_msg import load_and_split
from util import *

class Mem0Base():
    load_dotenv()

    def __init__(self, subdir_name, mem_config=BASE_MEM_CONFIG, **kwargs):
        self.mem_config = mem_config

        self.memory = Memory.from_config(self.mem_config)
        self.subdir_name = subdir_name
        self.subdir_path = os.path.join(DATA_DIR, self.subdir_name)

        self.checkpoint_path = os.path.join(self.subdir_path, f'{self.subdir_name}_checkpoints')
        self.raw_path = os.path.join(self.subdir_path, f'{self.subdir_name}_raw')

        self.checkpoint_counter = 1

    def add_mem_batch(self, mem_batch, batch_size=10):
        """
        Add memories in batches to avoid mem0 extraction issues with large conversations.

        Args:
            mem_batch: Either a file path (str) or list of message dicts
            batch_size: Number of messages to process at once (default: 20)
                       Empirically determined - too large (>50) may result in 0 extractions

        Returns:
            total_memories: Total number of memories extracted
        """
        if type(mem_batch) != list:
            mem_batch = load_and_split(mem_batch)

        print(f"Processing {len(mem_batch)} messages in batches of {batch_size}...")

        total_memories = 0
        all_results = []

        # Process in batches to avoid mem0 limitations
        for i in range(0, len(mem_batch), batch_size):
            batch = mem_batch[i:i+batch_size]
            result = self.memory.add(batch, user_id=USER_ID)
            count = len(result.get('results', []))
            total_memories += count
            all_results.extend(result.get('results', []))
            print(f"  Batch {i//batch_size + 1} (messages {i}-{i+len(batch)}): {count} memories extracted")

        print(f"Total memories extracted: {total_memories}")

        print(all_results)
        return {'total': total_memories, 'results': all_results}

    def reset_memory(self, del_checkpoints=True):
        self.memory = Memory()

        # Delete Main Memory
        delete_directory(MEM_LOCATION)

        # Delete all checkpoints
        delete_directory(self.checkpoint_path)
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)



    def export_checkpoint(self):
        destination = os.path.join(self.checkpoint_path, f'checkpoint_{self.checkpoint_counter}')
        copy_directory(MEM_LOCATION, destination)
        self.checkpoint_counter += 1

    def process_all_raw(self):
        dir_path = Path(self.raw_path)

        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        for file_path in dir_path.iterdir():
            if file_path.is_file():
                self.add_mem_batch(file_path)
                self.export_checkpoint()
                # print(file_path)

if __name__ == '__main__':
    test = Mem0Base('0518-014')

    # transcript_path = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/test/test_raw/10_1000060894.txt"

    #test.add_mem_batch(transcript_path)

    #test.export_checkpoint()
    test.process_all_raw()



