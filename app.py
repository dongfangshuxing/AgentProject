import streamlit as st
import time

from agent.react_agent import ReactAgent

# 页面标题设置：展示智能客服系统名称
st.title("智扫通机器人智能客服")
# 页面分割线
st.divider()

# 判断会话中是否存在 message 键，不存在则初始化空列表存储聊天记录
if "message" not in st.session_state:
    st.session_state["message"] = []

# 判断会话中是否存在 agent 键，不存在则创建智能体实例（全局单例）
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 遍历会话中的聊天记录列表，依次展示用户和助手的消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 创建聊天输入框，接收用户输入的问题/指令
prompt = st.chat_input()

# ===================== 处理用户输入逻辑 =====================
# 当用户输入内容不为空时执行
if prompt:
    # 在界面上展示用户发送的消息
    st.chat_message("user").write(prompt)
    # 将用户消息追加到聊天历史记录中
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 定义空列表，用于缓存AI流式返回的所有结果片段
    res_List = []
    # 加载动画效果，提示用户AI正在处理请求
    with st.spinner("智能客服思考中..."):
        # 调用智能体的流式执行方法，获取流式响应生成器
        res_stream = st.session_state["agent"].execute_stream(prompt)


        # 定义捕获函数：接收流式生成器，一边缓存完整数据，一边逐字输出实现打字机效果
        def capture(generator, cache_list):
            # 遍历AI返回的每一段响应内容（流式片段）
            for chuck in generator:
                # 将当前响应片段存入缓存列表，用于后续保存完整回答到历史记录
                cache_list.append(chuck)
                # 逐字遍历当前片段内容，实现打字机式的逐字显示效果
                for char in chuck:
                    # 控制逐字输出的速度，数值越小输出越快
                    time.sleep(0.01)
                    # 逐个返回字符，前端实时展示打字动画
                    yield char


        # 在界面上展示AI助手的流式回复
        st.chat_message("assistant").write_stream(capture(res_stream, res_List))
        # 将AI最终完整回复追加到聊天历史记录
        st.session_state["message"].append({"role": "assistant", "content": res_List[-1]})
