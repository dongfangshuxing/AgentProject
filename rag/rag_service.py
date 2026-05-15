from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_stores import VectorStoreService
from utils.prompt_load import load_rag_prompt


def print_prompt(prompt):
    """调试工具：打印完整拼接后的提示词，仅开发调试使用"""
    print("=" * 20, prompt.to_string(), "=" * 20)
    return prompt


class RagSummarizerService(object):
    """
    RAG 问答总结服务
    核心流程：用户问题 -> 向量库相似度检索 -> 规整拼接参考上下文 -> 加载自定义提示词 -> 调用大模型生成答案
    """
    def __init__(self):
        # 初始化向量存储服务
        self.vector_store = VectorStoreService()
        # 获取向量库检索器，用于召回相似知识库文档
        self.retriever = self.vector_store.get_retriever()
        # 从配置文件/文本文件加载自定义RAG提示词模版
        self.prompt_text = load_rag_prompt()
        # 初始化Prompt模板
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 注入全局单例大模型
        self.model = chat_model
        # 初始化链式执行流程
        self.chain = self._init_chain()

    def _init_chain(self):
        """初始化执行链路：提示模板 -> 打印提示词(调试) -> 大模型推理 -> 字符串解析输出"""
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retrieve_docs(self, query: str) -> list[Document]:
        """
        根据用户问题检索相关知识库文档
        :param query: 用户提问
        :return: 检索到的Document文档列表
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        完整RAG问答入口
        :param query: 用户问题
        :return: 大模型最终回答文本
        """
        # 1. 检索相关文档
        context_docx = self.retrieve_docs(query)

        # 无检索结果直接返回，避免空上下文传入模型
        if not context_docx:
            return "暂无相关信息"

        # 2. 规整拼接参考文档上下文，格式标准化、易被LLM理解
        context = ""
        counter = 0
        for doc in context_docx:
            counter += 1
            # 只取有用的source元数据，不拼接冗余metadata
            source = doc.metadata.get("source", "未知来源")
            context += f"【参考文档{counter}】\n内容：{doc.page_content}\n文件来源：{source}\n"
            context += "-" * 30 + "\n"

        # 3. 调用链路生成回答并返回
        return self.chain.invoke({
            "input": query,
            "context": context
        })


if __name__ == '__main__':
    # 单元测试入口
    summarize = RagSummarizerService()
    res = summarize.rag_summarize("小户型适合什么扫地机器人")
    print(res)