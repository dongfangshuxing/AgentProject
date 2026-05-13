from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompt():
    """加载主系统提示词文件"""
    try:
        # 从配置读取路径并转为绝对路径
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt] 在yml配置文件中未找到 key: main_prompt_path，错误信息：{str(e)}")
        raise e

    try:
        # 读取提示词文本内容
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"[load_system_prompt] 系统提示词文件不存在：{system_prompt_path}")
        raise
    except Exception as e:
        logger.error(f"[load_system_prompt] 读取提示词文件失败：{system_prompt_path}，错误：{str(e)}")
        raise


def load_rag_prompt():
    """加载RAG总结提示词文件"""
    try:
        # 从配置读取路径并转为绝对路径
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt] 在yml配置文件中未找到 key: rag_summarize_prompt_path，错误信息：{str(e)}")
        raise e

    try:
        # 读取提示词文本内容
        with open(rag_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"[load_rag_prompt] RAG提示词文件不存在：{rag_prompt_path}")
        raise
    except Exception as e:
        logger.error(f"[load_rag_prompt] 读取提示词文件失败：{rag_prompt_path}，错误：{str(e)}")
        raise


def load_report_prompt():
    """加载报告生成提示词文件"""
    try:
        # 从配置读取路径并转为绝对路径
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompt] 在yml配置文件中未找到 key: report_prompt_path，错误信息：{str(e)}")
        raise e

    try:
        # 读取提示词文本内容
        with open(report_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"[load_report_prompt] 报告提示词文件不存在：{report_prompt_path}")
        raise
    except Exception as e:
        logger.error(f"[load_report_prompt] 读取提示词文件失败：{report_prompt_path}，错误：{str(e)}")
        raise


if __name__ == '__main__':
    docx = load_report_prompt()
    print(docx)