from pymilvus import MilvusClient
from pymilvus import model
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = MilvusClient("milvus_demo.db")

# 创建 Collections
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

# 用向量表示文本
embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "明天的日程安排是什么？明天上午 9 点有一个部门会议，下午 2 点需要和客户进行视频会议。",
    "附近有什么好吃的餐厅？为您找到附近评分较高的餐厅有：XX 中餐厅，主打川菜；XX 西餐厅，提供正宗牛排等，08:10",
    "给我讲个笑话？许仙给老婆买了一顶帽子，白娘子戴上之后就死了，因为那是顶鸭（压）舌（蛇）帽，08:20",
    "播放一首舒缓的纯音乐？好的，为您播放久石让的《天空之城》，08:30",
    "查询一下从北京到上海的高铁时刻表？从北京到上海的高铁，最早一班是上午 7 点，之后每隔一小时左右都有车次，具体您可以通过 12306 官网查询，08:40",
    "设置下午 3 点的提醒，记得取快递？已为您设置下午 3 点取快递的提醒，08:50",
    "今天股市行情如何？很抱歉，我暂不支持实时查询股市行情，您可以通过专业的财经 APP 查看，09:00",
    "翻译 “我喜欢旅行” 成英文？“我喜欢旅行” 英文翻译为 “I like traveling”，09:10",
    "帮我查一下历史上的今天发生了什么？历史上的今天（具体日期），可能发生了众多事件，例如在 XX 年的今天，XX 重要事件发生，若您想了解更详细内容，可以在搜索引擎中查询，09:20",
    "最近的电影院在哪里？通过定位，最近的电影院是 XX 影城，距离您约 2 公里，位于 XX 街道 XX 号，09:30",
    "打开手机里的计算器？已为您打开手机计算器，09:40",
    "讲一下李白的生平？李白，字太白，号青莲居士，又号 “谪仙人”，是唐代著名诗人。他一生渴望入仕，游历四方，留下许多经典诗作，其诗歌风格豪放飘逸且富有浪漫主义色彩，09:50",
    "明天天气如何？预计明天多云，气温在 15 到 25 摄氏度之间，10:00",
    "查询一下苹果的营养价值？苹果富含维生素 C、纤维素等营养成分，有助于促进消化、增强免疫力等，10:10",
    "播放一首邓丽君的歌？好的，为您播放邓丽君的《月亮代表我的心》，10:20",
    "怎么制作红烧肉？先准备好五花肉等食材，将肉切块焯水，锅中倒油，放入冰糖炒出糖色，加入肉块翻炒上色，再加入葱姜蒜、八角等调料炖煮至软烂即可，10:30",
    "我的手机还有多少电量？剩余电量47%，10:40",
    "推荐一部好看的科幻电影？为您推荐《星际穿越》，影片讲述了一组宇航员穿越虫洞的冒险故事，有着震撼的视觉效果和深刻的科学设定，10:50",
    "附近有公园吗？附近有 XX 公园，距离您约 1.5 公里，您可以前往休闲散步，11:00",
    "把手机亮度调亮一点？好的，亮度调到50了，11:10"
]

vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

# 插入数据
res = client.insert(collection_name="demo_collection", data=data)

print(res)

# 语义搜索
query_text = "明天天气如何？"  # 添加一个模拟的时间戳

query_vectors = embedding_fn.encode_queries([query_text])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print("results:", res)
