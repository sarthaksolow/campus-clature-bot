from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

def get_user_memory(user_id: str, redis_url: str = "redis://redis:6379"):
    message_history = RedisChatMessageHistory(
        session_id=user_id,
        url=redis_url
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        chat_memory=message_history
    )

    return memory
