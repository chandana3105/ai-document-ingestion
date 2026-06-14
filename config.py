from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    documents_dir: str = "documents"
    vector_db_dir: str = "vector_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_chat_model: str = "gpt-4o-mini"
    claude_model: str = "claude-haiku-4-5"
    retrieval_k: int = 5
    llm_provider: str = "claude"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
