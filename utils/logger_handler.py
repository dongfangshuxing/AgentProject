import logging
import os
from datetime import datetime

from utils.path_tool import get_abs_path

# 获取日志存放根目录（项目根目录下的 logs 文件夹）
LOG_ROOT = get_abs_path("logs")

# 创建 logs 目录，存在则不创建
os.makedirs(LOG_ROOT, exist_ok=True)

# 全局统一的日志输出格式
DEFAULT_LOGGING_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(name: str = "agent", file_level: int = logging.DEBUG, console_level: int = logging.INFO,
               log_file=None) -> logging.Logger:
    """
    创建并配置日志器，支持控制台 + 文件双输出
    :param name: 日志器名称
    :param file_level: 文件日志级别
    :param console_level: 控制台日志级别
    :param log_file: 日志文件路径，不传则自动生成
    :return: 配置好的 logger 实例
    """
    # 创建日志器
    logger = logging.getLogger(name)
    # 设置全局最低日志级别
    logger.setLevel(logging.DEBUG)

    # 防止重复添加处理器
    if logger.handlers:
        return logger

    # ----------------------
    # 配置控制台输出
    # ----------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(console_handler)

    # 未传入日志文件路径时，自动按日期生成
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime("%Y%m%d")}.log")

    # ----------------------
    # 配置文件输出
    # ----------------------
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()