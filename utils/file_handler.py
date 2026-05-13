import hashlib
import os
from document import Document
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(file_path: str):
    """
    计算文件的 MD5 值（用于文件唯一标识、校验）
    :param file_path: 文件路径
    :return: MD5 十六进制字符串，失败返回 None
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"[get_file_md5_hex] 文件{file_path}不存在")
        return None

    # 检查是否为有效文件
    if not os.path.isfile(file_path):
        logger.error(f"[get_file_md5_hex] {file_path}不是文件")
        return None

    # 初始化 MD5 对象
    md5 = hashlib.md5()
    md5_size = 4096  # 每次读取 4KB 分块计算 MD5，避免大文件占内存

    try:
        # 二进制方式打开文件
        with open(file_path, "rb") as f:
            # 分块读取并更新 MD5
            while chunk := f.read(md5_size):
                md5.update(chunk)

            # 获取最终 MD5 字符串
            md5_hex = md5.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[get_file_md5_hex] 计算文件{file_path}失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """
    遍历文件夹，只返回指定后缀的文件路径列表
    :param path: 文件夹路径
    :param allowed_types: 允许的文件后缀，如 ('.pdf', '.txt')
    :return: 符合条件的文件路径列表
    """
    file_list = []

    # 检查路径是否为文件夹
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path}不是文件夹")
        return allowed_types

    # 遍历目录下所有文件
    for file in os.listdir(path):
        # 只保留指定后缀的文件
        if file.endswith(allowed_types):
            file_list.append(os.path.join(path, file))

    return file_list


def pdf_loader(file_path: str, password=None) -> list[Document]:
    """PDF 加载器，返回 LangChain Document 列表"""
    return PyPDFLoader(file_path, password).load()


def txt_loader(file_path: str) -> list[Document]:
    """TXT 加载器，返回 LangChain Document 列表"""
    return TextLoader(file_path).load()
