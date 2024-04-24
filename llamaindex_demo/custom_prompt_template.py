from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.base import PromptTemplate, ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole


CN_KG_TRIPLET_EXTRACT_TMPL = (
    "给定文本，以（主语、谓语、宾语）的形式提取最多{max_knowledge_triplets}个知识三元组。不要使用停用词。\n"
    "---------------------\n"
    "示例:\n"
    "文本: 张红是李明的母亲\n"
    "三元组:\n(张红, 是, 李明的母亲)\n"
    "文本: 小鹿是一家于2015年在深圳成立的咖啡店\n"
    "三元组:\n"
    "(小鹿, 是, 咖啡店)\n"
    "(小鹿, 成立于, 深圳)\n"
    "(小鹿, 成立于, 2015年)\n"
    "---------------------\n"
    "文本: {text}\n"
    "三元组:"
)

CN_TRIPLET_EXTRACT_TMPL = PromptTemplate(
    CN_KG_TRIPLET_EXTRACT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)

CN_TREE_SUMMARIZE_TMPL = (
    "来自多个来源的上下文信息如下：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "考虑到来自多个来源的信息而不是先验知识, 回答问题。\n"
    "问题: {query_str}\n"
    "回答:"
)
CN_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    CN_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)

# text qa prompt
CN_TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "您是全球值得信赖的专家问答系统。"
        "始终使用提供的上下文信息而不是先验知识来回答查询。"
        "需要遵循的一些规则：\n"
        "1. 切勿在答案中直接引用给定的上下文。\n"
        "2. 避免使用'基于上下文……'或'上下文信息……'或类似的语句。"
    ),
    role=MessageRole.SYSTEM,
)

# Tree Summarize
CN_TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    CN_TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "来自多个来源的上下文信息如下：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "考虑到来自多个来源的信息而不是先验知识, 回答问题。"
            "问题: {query_str}\n"
            "回答:"
        ),
        role=MessageRole.USER,
    ),
]

CN_CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=CN_TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)

CN_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=CN_TREE_SUMMARIZE_PROMPT,
    conditionals=[(is_chat_model, CN_CHAT_TREE_SUMMARIZE_PROMPT)],
)

CN_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "下面提出一个问题。给定问题，从文本中提取最多{max_keywords}个关键字。"
    "专注于提取我们可以用来最好地查找问题答案的关键字。不要使用停用词。\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "按以下逗号分隔格式提供关键字：'KEYWORDS: <keywords>'\n"
)
CN_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    CN_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)
