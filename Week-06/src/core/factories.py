from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from src.core.config import app_config

class GenericFactory:
    """مصنع مرن لإنشاء كائنات النماذج والتضمين بدون التقيّد بشركة معينة"""

    @staticmethod
    def get_llm() -> BaseChatModel:
        cfg = app_config.llm
        provider = cfg.provider.lower()
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=cfg.model_name, temperature=cfg.temperature)
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=cfg.model_name, temperature=cfg.temperature)
        else:
            raise ValueError(f"مزود التوليد غير مدعوم: {provider}")

    @staticmethod
    def get_embeddings() -> Embeddings:
        cfg = app_config.embedding
        provider = cfg.provider.lower()
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=cfg.model_name)
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=cfg.model_name)
        else:
            raise ValueError(f"مزود التضمين غير مدعوم: {provider}")