import os


def get_project_root():
    """
    获取工程的根目录
    :return: str
    """
    # 获取绝对路径
    current_file = os.path.abspath(__file__)
    # 获取当前文件所在的文件夹路径（去掉文件名，只剩目录）
    current_dir = os.path.dirname(current_file)
    # 向上再跳一级目录，得到项目根目录
    project_root = os.path.dirname(current_dir)
    # 返回项目根目录的绝对路径
    return project_root


def get_abs_path(relative_path: str):
    """
    传递相对路径，获取绝对路径
    :param relative_path:
    :return: str
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
