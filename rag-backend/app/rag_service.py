from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI
from app.config import settings


class RAGService:
    def __init__(self):
        self.message_store: Dict[str, ChatMessageHistory] = {}
        self._init_models()
        self._build_pipeline()

    def _init_models(self):
        # 1. 基础 Embedding
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        # 2. Pinecone 检索器
        self.vector_store = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME, embedding=self.embeddings
        )
        self.base_retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": settings.RETRIEVAL_K}
        )

        # 先扩大相似度候选集，再用多语言 Cross-Encoder 精排，避免只保留
        # 少量局部片段导致章节类问题的上下文不完整。
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.RERANK_MODEL
        )
        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder, top_n=settings.RERANK_TOP_N
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.base_retriever
        )

        # 4. DeepSeek 实例
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com",
            temperature=0.0,
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]

    def _build_pipeline(self):
        # 意图重写器
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    "将以下中文问题重写为适合技术手册检索的高密度英文关键词组合，只返回关键词：\n{question}",
                ),
            ]
        )
        query_rewriter = rewrite_prompt | self.llm | StrOutputParser()

        # 检索逻辑
        def retrieve_step(chain_input: dict):
            history_slice = chain_input.get("chat_history", [])[-6:]
            rewritten_query = query_rewriter.invoke(
                {"chat_history": history_slice, "question": chain_input["question"]}
            ).strip()
            # 向量库和文档均为英文，使用英文改写结果进行召回和排序。
            return self.compression_retriever.invoke(rewritten_query)

        # 生成回答 Prompt
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是技术规范专家。必须严格依据以下参考资料回答问题，严禁编造：\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        core_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(retrieve_step) | RunnableLambda(format_docs)
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        self.pipeline = RunnableWithMessageHistory(
            core_chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def execute_query(self, question: str, session_id: str) -> str:
        return self.pipeline.invoke(
            {"question": question}, config={"configurable": {"session_id": session_id}}
        )


rag_service = RAGService()
