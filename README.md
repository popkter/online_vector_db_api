# 聊天记忆系统 API

这是一个基于 FastAPI 和 Milvus 向量数据库实现的聊天记忆系统，能够存储和检索聊天记录，并通过语义相似度进行智能搜索。

## 功能特点

- 支持聊天记录的存储和向量化
- 基于语义相似度的智能搜索
- 流式响应的聊天分析
- 支持多领域（domain）分类
- 时间戳记录和查询

## 技术栈

- FastAPI：Web 框架
- Milvus：向量数据库
- SentenceTransformer：文本向量化模型
- DeepSeek：AI 对话模型
- Uvicorn：ASGI 服务器

## 环境要求

- Python 3.9+
- 环境变量配置（.env 文件）：
  - CHAT_API_KEY：DeepSeek API 密钥

## 安装依赖

```bash
pip install fastapi uvicorn python-dotenv openai pymilvus sentence-transformers torch
```

## 启动服务

```bash
python main.py
```

服务将在 `http://0.0.0.0:10013` 启动

## API 接口

### 1. 添加聊天记录

http
POST /add_chat
请求体：

```json{
"chat": "聊天内容",
"domain": "default",
"time": 1737542100 // Unix 时间戳
```

### 2. 搜索聊天记录

http
GET /search?query=搜索关键词&limit=5

参数：

- query：搜索关键词
- limit：返回结果数量（默认5条）

## 系统架构

1. 聊天记录存储：
   - 使用 SentenceTransformer 将文本转换为向量
   - 将向量和元数据存储在 Milvus 数据库中

2. 搜索功能：
   - 将搜索查询转换为向量
   - 在 Milvus 中进行向量相似度搜索
   - 返回最相似的结果

3. 智能分析：
   - 使用 DeepSeek 模型分析搜索结果
   - 生成自然语言回复
   - 支持流式响应

## 注意事项

1. 首次运行时会自动创建 Milvus 集合
2. 需要确保 Milvus 服务已经启动
3. 环境变量 TOKENIZERS_PARALLELISM 设置为 false 以避免警告
4. 向量维度固定为 512

## 错误处理

系统包含完整的错误处理机制：

- 数据库操作异常
- API 调用异常
- 向量化处理异常