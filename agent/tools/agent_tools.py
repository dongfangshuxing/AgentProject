# 模块功能：智能体Agent 内置工具集合
import os
import random
from langchain_core.tools import tool

# 业务自定义模块导入
from rag.rag_service import RagSummarizerService
from utils.config_handler import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 初始化RAG摘要服务实例，用于向量库文档检索总结
rag = RagSummarizerService()

# 合法用户ID池
user_ids = ["1001", "1002", "1003", "1004", "1005", 
            "1006", "1007", "1008", "1009", "1010", "1011", "1012"]

# 可用统计月份范围：2025全年12个月
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

# 全局缓存外部业务数据
# 数据结构：{用户ID: {月份: {特征,效率,耗材,对比评价}}}
external_data = {}


@tool(description="从向量存储中检索参考资料并总结返回")
def rag_summarize(query: str) -> str:
    """
    RAG检索工具
    :param query: 检索查询问题
    :return: 知识库检索后的摘要文本
    """
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以可读消息字符串返回")
def get_weather(city: str) -> str:
    """
    模拟天气接口，固定返回晴天模板数据
    :param city: 城市名称
    :return: 天气详情文本
    """
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="随机获取用户所在城市名称，纯字符串返回")
def get_user_location() -> str:
    """随机从预设城市列表中选取一个返回"""
    return random.choice(["深圳", "合肥", "杭州", "南京"])


@tool(description="随机获取合法用户ID，纯字符串返回")
def get_user_id() -> str:
    """从标准用户ID池中随机抽取一个用户ID"""
    return random.choice(user_ids)


@tool(description="随机获取2025年任意月份，纯字符串返回")
def get_current_month() -> str:
    """从全年月份列表随机返回一个年月标识"""
    return random.choice(month_arr)


def generate_external_data():
    """
    加载外部CSV业务数据到全局字典 external_data
    逻辑：
    1. 仅全局数据为空时才加载，避免重复IO读取文件
    2. 读取配置中配置的外部数据CSV路径
    3. 跳过表头，逐行解析、去除双引号、结构化存入嵌套字典
    """
    # 已有数据则不再重复加载
    if external_data:
        return

    # 从配置读取路径并转为绝对路径
    external_data_path = get_abs_path(agent_config["external_data_path"])

    # 校验文件是否存在
    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件不存在：{external_data_path}")

    # 读取并解析CSV数据
    with open(external_data_path, "r", encoding="utf-8") as f:
        # 跳过第一行表头，遍历后续数据行
        for line in f.readlines()[1:]:
            line = line.strip()
            if not line:
                continue

            # 按逗号分割六个字段
            arr: list[str] = line.split(",")
            if len(arr) < 6:
                continue

            # 去除字段自带双引号
            user_id: str = arr[0].replace('"', "")
            feature: str = arr[1].replace('"', "")
            efficiency: str = arr[2].replace('"', "")
            consumables: str = arr[3].replace('"', "")
            comparison: str = arr[4].replace('"', "")
            time: str = arr[5].replace('"', "")

            # 初始化用户层级字典
            if user_id not in external_data:
                external_data[user_id] = {}

            # 按 用户ID-月份 维度存入结构化数据
            external_data[user_id][time] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison
            }


@tool(description="根据用户ID和月份，查询扫地机器人使用记录，返回可读文本信息")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部业务数据查询工具
    :param user_id: 用户编号 如 1001
    :param month: 年月 如 2025-01
    :return: 格式化后的用户使用记录文本，无数据返回空字符串并打印警告日志
    """
    # 确保数据已加载到内存
    generate_external_data()

    try:
        # 根据用户ID+月份取出记录
        record = external_data[user_id][month]
        # 转为易读文本，适配Agent输出
        res_text = (
            f"【用户使用记录】\n"
            f"用户特征：{record['特征']}\n"
            f"清洁效率：{record['效率']}\n"
            f"耗材状态：{record['耗材']}\n"
            f"横向对比：{record['对比']}"
        )
        return res_text
    except KeyError:
        # 无匹配数据记录告警
        logger.warning(f"[fetch_external_data] 未检索到用户 {user_id} 在 {month} 的使用记录")
        return ""

@tool(description="无入参，无返回值。调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续的提示词切换提供上下文信息")
def fill_context_for_report():
    return  "fill_context_for_report已调用"