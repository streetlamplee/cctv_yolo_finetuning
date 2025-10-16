import os
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent
os.chdir(ROOT_DIR)

for i in range(1, 13):
    for file in os.listdir(os.path.join(ROOT_DIR, str(i))):
        new_file_name = f"{i}_{file}"
        print(new_file_name)
        shutil.copy(os.path.join(ROOT_DIR, str(i), file), os.path.join(ROOT_DIR, new_file_name))

print("done")