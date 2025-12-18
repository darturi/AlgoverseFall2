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

    def add_mem_batch(self, mem_batch):
        if type(mem_batch) == str:
            mem_batch = load_and_split(mem_batch)

        print(type(mem_batch))
        print(type(mem_batch) is list[dict])

        return self.memory.add(mem_batch, user_id=USER_ID)

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
                self.add_mem_batch("PLACEHOLDER")
                self.export_checkpoint()
                # print(file_path)

if __name__ == '__main__':
    test = Mem0Base('test')

    transcript_path = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/test/test_raw/6_1000088056.txt"

    test.add_mem_batch(transcript_path)

    test.export_checkpoint()
    # test.process_all_raw()



