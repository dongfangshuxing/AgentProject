# ==========================================================
# 模块名称：agent_middleware.py
# 模块功能：LangGraph / LangChain 智能体中间件集合
# 核心能力：
#  1. 工具调用监控（日志+上下文注入）
#  2. 大模型调用前日志记录
#  3. 动态提示词切换（普通对话 / 报告生成）
# 适用场景：Agent 流程增强、日志追踪、报告生成模式控制
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
# 作用：监听所有工具执行，打印日志，并在执行 fill_context_for_report 时保存结果到上下文
# -------------------
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,  # 工具调用请求对象
        handler: Callable[[ToolCallRequest], ToolMessage | Command]  # 工具执行处理器
) -> ToolMessage | Command:
    # 打印工具名称
    logger.info(f"[tool monitor] 执行工具: {request.tool_call['name']}]")
    # 打印传入参数
    logger.info(f"[tool monitor] 传入参数: {request.tool_call['args']}]")

    try:
        # 执行原始工具逻辑
        result = handler(request)

        # 工具执行成功日志
        logger.info(f"[tool monitor] 工具: {result.tool_call['name']}调用成功]")

        # 如果当前工具是【报告上下文填充工具】，将结果存入 runtime.context 供全局使用
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result

    # 工具执行异常捕获
    except Exception as e:
        logger.info(f"[tool monitor] 工具: {result.tool_call['name']}调用失败,原因：{str(e)}]")
        # 抛出异常，不阻断上层错误流程
        raise e


# -------------------
# 大模型调用前中间件
# 作用：在 LLM 推理前打印消息数量与最后一条消息内容，便于调试
# -------------------
@before_model
def log_before_model(
        state: AgentState,  # 智能体当前状态
        runtime: Runtime,  # 运行时环境
):
    msg_len = len(state["messages"])
    logger.info(f"[log_before_model]即将调用模型，带有{msg_len}条消息")

    #  只有消息不为空时才打印最后一条
    if msg_len > 0:
        last_msg = state["messages"][-1]
        try:
            msg_type = last_msg.__class__.__name__
            content = last_msg.content.strip()
            logger.debug(f"[log_before_model] {msg_type} ||| {content}")
        except Exception as e:
            logger.debug(f"[log_before_model] 读取消息失败: {str(e)}")

    # 不修改提示词，直接返回 None
    return None


# -------------------
# 动态提示词切换中间件
# 作用：根据 context 中是否有 report 标记，自动切换系统提示词
# -------------------
@dynamic_prompt  # 每次提示词生成前自动调用
def report_prompt_switch(request: ModelRequest):
    # 从运行时上下文中获取是否为【报告生成模式】
    is_report = request.runtime.context.get("report", False)

    # 如果是报告模式 → 加载报告专用提示词
    if is_report:
        return load_report_prompt()

    # 否则 → 加载常规系统提示词
    return load_system_prompt()