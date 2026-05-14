import os
from langchain_chroma import Chroma
from model.factory import embedding_model
from utils.config_handler import chroma_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from langchain_core.documents import Document


class VectorStoreService:
    """
    向量存储服务，负责文档加载、分片、向量化及持久化到 Chroma 数据库。
    支持基于 MD5 的去重加载，避免重复处理相同内容的文件。
    """

    def __init__(self):
        """
        初始化向量存储实例和文本分片器。
        """
        # 初始化 Chroma 向量数据库，配置集合名称、嵌入模型和持久化目录
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )
        # 配置递归字符文本分片器，控制分片大小、重叠长度和分隔符规则
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chuck_size"],
            chunk_overlap=chroma_config["chuck_overlap"],
            separators=chroma_config["chuck_separators"],
            length_function=len
        )

    def get_retriever(self):
        """
        获取检索器实例，用于基于向量相似度检索相关文档片段。

        Returns:
            BaseRetriever: 配置检索器，默认返回 top-k 个最相似结果。
        """
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def load_documents(self):
        """
        批量加载指定目录下的知识库文件，进行去重检查、文本分片和向量化存储。
        已处理过的文件（通过 MD5 校验）会自动跳过，避免重复入库。
        """

        def check_md5(md5_str: str) -> bool:
            """
            检查指定 MD5 值是否已存在于历史处理记录中。

            Args:
                md5_str: 文件的 MD5 哈希值。

            Returns:
                bool: True 表示该文件内容已处理过，False 表示未处理。
            """
            md5_store_path = get_abs_path(chroma_config["md5_hex_store"])
            # 若记录文件不存在，创建空文件并视为未处理
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False
            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() == md5_str:
                        return True
            return False

        def save_md5(md5_str: str) -> None:
            """
            将文件 MD5 值追加写入历史记录，标记为已处理。

            Args:
                md5_str: 文件的 MD5 哈希值。
            """
            md5_store_path = get_abs_path(chroma_config["md5_hex_store"])
            # 使用追加模式写入，保留已有记录
            with open(md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5_str + "\n")

        def get_file_document(read_str: str):
            """
            根据文件扩展名选择对应的加载器解析文档内容。

            Args:
                read_str: 文件的绝对路径。

            Returns:
                list: 解析后的 Document 对象列表，不支持的类型返回空列表。
            """
            if read_str.endswith("txt"):
                return text_loader(read_str)
            if read_str.endswith("pdf"):
                return pdf_loader(read_str)
            logger.warning(f"不支持的文件类型，跳过处理: {read_str}")
            return []

        # 扫描指定目录，过滤出允许类型的文件列表
        allow_file_path:list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )

        logger.info(f"扫描到待处理文件共 {len(allow_file_path)} 个，开始加载知识库...")

        for file_path in allow_file_path:
            md5_hex = get_file_md5_hex(file_path)

            # 去重检查：已处理的文件直接跳过
            if check_md5(md5_hex):
                logger.info(f"文件已存在知识库中，跳过加载: {file_path} (MD5: {md5_hex})")
                continue

            try:
                # 解析文件内容为 LangChain Document 列表
                documents:list[Document] = get_file_document(file_path)

                if not documents:
                    logger.warning(f"文件解析后无有效内容，跳过入库: {file_path}")
                    continue

                # 对文档进行递归分片，控制单段文本长度
                spliter_document = self.spliter.split_documents(documents)
                if not spliter_document:
                    logger.warning(f"文档分片后无有效内容，跳过入库: {file_path}")
                    continue

                # 将分片后的文档写入 Chroma 向量数据库
                self.vector_store.add_documents(spliter_document)
                # 记录 MD5，标记该文件已处理
                save_md5(md5_hex)

                logger.info(f"知识库加载成功: {file_path}，共写入 {len(spliter_document)} 个分片")

            except Exception as e:
                logger.error(f"知识库加载失败: {file_path}，异常类型: {type(e).__name__}，详情: {e}")
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_documents()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for doc in res:
        print(doc.page_content)
        print("*" * 20)
