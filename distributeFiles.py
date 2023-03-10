import os
import shutil

"""
Because Google Drive did not perform well when number of files in a directory is large,
for image and sentence vectors, we use a distributed system.
e.g. file "root/458989.pt" should be moved to "root/45/458989.pt"
e.g. file "root/1858989.pt" should be moved to "root/18/1858989.pt"
"""

# directory to make distribution
files = os.listdir("dataset/testdesc")

for file in files:
    if not os.path.exists("dataset/testdesc/"+file[:2]):
        os.makedirs("dataset/testdesc/"+file[:2])
    shutil.move("dataset/testdesc/"+file, "dataset/testdesc/"+file[:2]+"/"+file)