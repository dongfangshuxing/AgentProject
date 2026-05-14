# 抽象工厂模式：统一管理大模型、向量嵌入模型的创建
# 作用：解耦模型创建逻辑，方便后续切换不同厂商模型
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from utils.config_handler import rag_config


# 抽象基类：定义所有模型工厂必须实现的接口
class BaseModelFactory(ABC):

    # 抽象方法：所有子类必须实现，用于生成并返回模型实例
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


# 对话模型工厂：专门创建【聊天大模型】实例（如通义千问）
class ChatModelFactory(BaseModelFactory):
    # 实现抽象方法，生成并返回聊天模型
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        # 创建并返回通义千问聊天模型，使用配置文件中的模型名和API Key
        return ChatTongyi(model=rag_config["chat_model_name"], api_key=rag_config["api_key"])


# 向量嵌入模型工厂：专门创建【文本嵌入模型】实例（用于RAG向量化）
class EmbeddingsFactory(BaseModelFactory):
    # 实现抽象方法，生成并返回向量嵌入模型
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        # 创建并返回通义DashScope嵌入模型，使用配置中的模型名和API Key
        return DashScopeEmbeddings(model=rag_config["embedding_name"], dashscope_api_key=rag_config["api_key"])


# 全局单例：创建聊天模型实例（全局可直接使用）
chat_model = ChatModelFactory().generate()

# 全局单例：创建向量嵌入模型实例（全局可直接使用）
embedding_model = EmbeddingsFactory().generate()
