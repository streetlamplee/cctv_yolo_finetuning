import os
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent.parent
os.chdir(ROOT_DIR)
def main(folder:str):
    for i in range(1, 14):
        if not os.path.exists(os.path.join(ROOT_DIR, folder, str(i))):
            continue

        for file in os.listdir(os.path.join(ROOT_DIR, folder, str(i))):
            new_file_name = f"{i}_{file}"
            print(new_file_name)
            shutil.copy(os.path.join(ROOT_DIR, folder, str(i), file), os.path.join(ROOT_DIR, folder, new_file_name))

    print("done")

if __name__ == "__main__":
    main("1121_data")