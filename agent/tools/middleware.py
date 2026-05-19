# ==========================================================
# 模块名称：agent_middleware.py
# 模块功能：LangGraph / LangChain 智能体中间件集合
# 核心能力：
#  1. 工具调用监控（日志+上下文注入）
#  2. 大模型调用前日志记录
#  3. 动态提示词切换（普通对话 / 报告生成）
# ==========================================================

from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_load import load_report_prompt, load_system_prompt


# -------------------
# 工具调用监控中间件
# -------------------
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    logger.info(f"[tool monitor] 执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor] 传入参数: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor] 工具: {request.tool_call['name']} 调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = result

        return result

    except Exception as e:
        logger.error(f"[tool monitor] 工具调用失败: {str(e)}")
        raise e


# -------------------
# 模型调用前日志（已修复空消息问题）
# -------------------
@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    messages = state.get("messages", [])
    msg_len = len(messages)

    logger.info(f"[log_before_model] 即将调用模型，消息数量：{msg_len}")

    if msg_len > 0:
        try:
            last = messages[-1]
            logger.debug(f"[log_before_model] 最后消息：{last.__class__.__name__} | {last.content.strip()}")
        except:
            logger.debug("[log_before_model] 消息解析失败")

    return None


# -------------------
# 动态提示词切换
# -------------------
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()
    return load_system_prompt()