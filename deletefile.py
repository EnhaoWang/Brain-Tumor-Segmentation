import os
import shutil
CUR_PATH = r'./test_dataset_output'
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    file_num = 0
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            if os.path.splitext(file_path)[1] == '.png':
                file_num = file_num + 1
                os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print("Number of files:", file_num)

del_file(CUR_PATH)
print('file deleting successful!')

