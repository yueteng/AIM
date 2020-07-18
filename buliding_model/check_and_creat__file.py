import os
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def check_folder_contents(file):
    for the_file in os.listdir(file):

        file_path = os.path.join(file, the_file)

        try:

            if os.path.isfile(file_path):

                os.unlink(file_path)
        except Exception as e:
            print(e)