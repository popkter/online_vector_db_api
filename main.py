from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from openai import OpenAI

import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()
CHAT_API_KEY = os.getenv("CHAT_API_KEY")

CLIENT = OpenAI(api_key=CHAT_API_KEY, base_url="https://api.deepseek.com")

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 定义提示词常量
PROMPT = """
你是一个拥有记忆的人工智能助手，能够根据与用户聊天的记忆回复用户对于过去对话的问题的询问。
任务描述：
在给定的记忆搜索结果列表中，根据规则筛选出与用户询问最相关的内容，并生成一条自然语言回复。如果没有匹配的结果，生成一条灵活的“未找到结果”的回复。
单个记忆搜索结果的格式:
    {
        "similarity": 0.6265, # 表示与查询内容的相似度，值范围通常是 0.0 ~ 1.0（或其他计算度量方式）。
        "content": "播放一首邓丽君的歌？好的，为您播放邓丽君的《我只在乎你》，",# 与查询匹配的文本内容
        "time": 1737450307,# 事件发生的时间
        "domain": "history"# 查询结果的领域或分类标签，例如 "history" 表示与历史相关
    },

匹配规则：

1. 根据询问的时间段筛选结果,需要判断是不是今天。

    - 询问时间为“上午”：筛选事件发生时间在 06:00 ~ 11:59 的结果；

    - 询问时间为“下午”：筛选事件发生时间在 12:00 ~ 17:59 的结果；

    - 询问时间为“晚上”：筛选事件发生时间在 18:00 ~ 23:59 的结果；

    - 询问时间为“深夜”：筛选事件发生时间在 00:00 ~ 05:59 的结果。

2. 在符合时间段的结果中，根据相似度排序：

    - 从时间段内的结果中选择相似度最高的一项；

    - 如果相似度相同，选择时间最近的一项。

3. 生成自然语言回复：

    - 如果找到匹配结果：

        根据结果内容和用户询问，生成一条直接回答用户问题的自然语言回复，
        回复应简洁明了，口语化一些，直接解决用户的问题，避免冗余信息。你是一个有记忆的智能体，因此回复结果中不要出现如下或意思相近的句子：
            - 根据你提供的信息
            - 根据查询到的信息
            - 根据查询结果
        而是应该使用，让我想想，我想起来了，这样拟人的表达。

    - 如果未找到匹配结果：
        生成一条灵活的“未找到结果”的回复，
        回复应友好且提供可能的解决方案，例如：
            “我不记得了。”
            “可能需要您再确认一下相关时间或内容。”
            “没有找到匹配的记录，您可以尝试换个时间段或关键词查询。”

4. 示例：

    - 匹配到结果：
        用户询问：“我上午听了什么歌？”
        回复：“上午听的是《小城故事》。”
        
        用户询问：“我上午听了什么歌？”
        回复：“这你都不记得啦？上午听的是《小城故事》。”

    - 未匹配到结果：
        用户询问：“我上午听了什么歌？”
        回复：“我怎么不记得你上午听歌了呀”

"""


# 定义数据模型
class ChatInput(BaseModel):
    chat: str
    domain: str = "default"
    time: Optional[int] = None  # 可选的时间戳字段


class SearchResult(BaseModel):
    similarity: float
    content: str
    time: str
    domain: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
print("模型加载完成")

client = MilvusClient("milvus_demo.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # client.drop_collection(collection_name="demo_collection")
        if not client.has_collection(collection_name="demo_collection"):
            client.create_collection(
                collection_name="demo_collection",
                dimension=512
            )
        else:
            client.load_collection(collection_name="demo_collection")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        raise

    yield

    print("应用程序关闭，执行清理操作...")


# 将 lifespan 添加到 FastAPI 应用
app = FastAPI(lifespan=lifespan)


@app.post("/add_chat")
async def add_chat(chat_input: ChatInput):
    """添加聊天记录到向量数据库"""
    try:
        # 向量化文档
        docs = [chat_input.chat]
        vector = model.encode(docs)[0]

        current_time = chat_input.time if chat_input.time else int(datetime.now().timestamp())

        # 获取当前数据量
        collection_stats = client.get_collection_stats(collection_name="demo_collection")
        size = collection_stats.get("row_count", 0)

        # 准备插入数据
        data = [{
            "id": size,
            "vector": vector.tolist(),  # 添加向量数据
            "text": chat_input.chat,
            "time": current_time,
            "domain": chat_input.domain
        }]

        # 插入数据
        res = client.insert(collection_name="demo_collection", data=data)
        return {"message": "添加成功", "id": res["ids"][0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_chat(query: str, limit: Optional[int] = 5):
    """搜索聊天记录"""
    try:
        # 向量化查询文本
        query_vector = model.encode([query])[0]

        # 执行搜索
        results = client.search(
            collection_name="demo_collection",
            data=[query_vector.tolist()],
            limit=limit,
            output_fields=["text", "time", "domain"]
        )[0]

        # 格式化结果
        search_results = [
            SearchResult(
                similarity=round(float(item['distance']), 4),
                content=item["entity"]["text"],
                time=datetime.fromtimestamp(item["entity"]["time"]).strftime('%Y-%m-%d %H:%M'),
                domain=item["entity"]["domain"]
            )
            for item in results
        ]

        print("查询结果:", search_results)

        return StreamingResponse(analyze_chat(query, search_results), media_type="text/event-stream")


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def analyze_chat(query: str, search_results: List[SearchResult]):
    """分析聊天记录并生成回复"""
    try:
        messages = [
            {
                "role": "assistant",
                "content": PROMPT +
                           f"现在的时间是 {datetime.now().strftime('%Y-%m-%d-%h')}. 当需要用到时间时候请参照今天" +
                           f"不要输出任何markdown格式数据。"
            },
            {
                "role": "user",
                "content": f"用户问题是：{query},查询到的结果是: ${search_results}"
            }
        ]

        response = ''
        analysis_stream = stream_response(messages)
        for chunk in analysis_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                response += content
                yield f"data: 'chunk':{content}\n\n"

        yield f"data: 'finish':{response}\n\n"


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 流式响应
def stream_response(messages):
    return CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10013)
