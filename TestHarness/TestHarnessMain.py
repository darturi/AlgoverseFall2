from pathlib import Path
from em_organism_dir.eval.util.gen_eval_util import gen_and_eval_hf
import shutil
from google.colab import drive
from AccessPoint.MemoryGPT_def import MemoryGPTInstance


# Take checkpoint folder as input
# Also need to pass the test file(s)

async def test_checkpoints(
        checkpoint_dir,
        base_model_instance,
        model_tok,
        save_dir=Path('em_organism_dir/data/responses'),
        question_files={'em_organism_dir/data/eval_questions/first_plot_questions.yaml':['aligned','coherent']},
        drive_file_path='/content/drive/My Drive/AlgoverseFallPersonal/MemoryWork/EvalResults/'
):
    if type(checkpoint_dir) is str:
        checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    # Iterate through checkpoint dir
    for file_path in checkpoint_dir.iterdir():

        # For each checkpoint instantiate a memory object
        instance = MemoryGPTInstance(file_path, base_model_instance, model_tok)

        # For each test we want to conduct run a test instance
        for test_file, metrics in question_files.items():
            suffix = test_file.split('/')[-1]
            SAVE_PATH = save_dir / f"{suffix[:-4]}.csv"

            print('Results will be saved to', SAVE_PATH)

            await gen_and_eval_hf(instance,
                                  str(SAVE_PATH),
                                  overwrite=True,
                                  question_file=test_file,
                                  n_per_question=5,  # Changed from 20 to 5
                                  new_tokens=600,
                                  temperature=1.0,
                                  top_p=1.0,
                                  metrics=metrics)

            f_name = f"TEST_colab.csv"

            drive_file_path = f"{drive_file_path}/{f_name}"

            shutil.copy(SAVE_PATH, drive_file_path)

        print(file_path)

    return

if __name__ == "__main__":
    test_checkpoints("/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data/0518-014/0518-014_checkpoints")