"""
金融文本分类（少样本学习）
调用本地Qwen2.5模型，利用Few-Shot上下文实现四分类：
新闻报道、财务报告、公司公告、分析师报告
"""
#1.导入所有需要的库
from ollama import chat
# 必须加这个！彩色打印用的
from rich import print

#2.定义金融文本的4个类别 + 每个类别的示例句子（Few-Shot学习素材）
class_examples = {
    "新闻报道": [
        "宁德时代今日发布新电池，能量密度提升20%",
        "贵州茅台上半年营收同比增长15%"
    ],
    "财务报告": [
        "公司2024年净利润10亿元，同比下降5%",
        "资产负债表显示货币资金充足"
    ],
    "公司公告": [
        "本公司拟收购全资子公司100%股权",
        "股东减持股份计划公告"
    ],
    "分析师报告": [
        "维持腾讯控股买入评级，目标价400港元",
        "行业调研：新能源汽车渗透率持续提升"
    ]
}

# 定义使用的模型名称（你本地的模型，正确）
MODEL_NAME = "qwen2.5:0.5b"

def init_prompts():
    # 角色定义
    # "role"(角色),一共有三种
    # "system"	系统 / 规则制定者（最顶级）
    # "user"	用户（你 / 提问的人）
    # "assistant"	助手（模型自己）
    # "content"：内容（规定：这个角色说了什么？）
    system_prompt = {
        "role":"system",
        "content":"你是专业的金融文本分类助手，只能从【新闻报道、财务报告、公司公告、分析师报告】中选择一个类别回复，禁止输出其他内容！"
    }
    # 存储对话历史
    messages = [system_prompt]

    # 构造少样本对话
    for label,examples in class_examples.items():
        for example in examples:
            # 统一提问格式
            messages.append({"role": "user","content":f"文本：{example}，请分类"})
            messages.append({"role": "assistant", "content":label})
    return messages

def inference(text:str,messages:list):
    # 复制上下文，不修改原数据
    chat_messages = messages.copy()
    # 统一提问格式！和学习时的句式一样
    chat_messages.append({"role":"user","content":f"文本：{text}，请分类"})
    try:
        # 调用本地Ollama模型
        response = chat(model=MODEL_NAME,messages=chat_messages)
        # 提取结果
        result = response["message"]["content"].strip()
        return result
    except Exception as e:
        return f"分类失败：{str(e)}"


if __name__ == "__main__":
    # 初始化上下文
    print("[yellow]正在初始化少样本学习上下文...[/yellow]")
    few_shot_messages = init_prompts()
    print("[green]初始化完成！模型已经学习完所有金融文本示例[/green]\n")

    # 测试句子
    test_texts = [
        "比亚迪发布2024年第三季度财报，营收增长25%",
        "本公司拟非公开发行股票募集资金",
        "给予宁德时代增持评级，目标价280元",
        "今日A股三大指数集体收涨"
    ]

    # 批量测试
    print("[blue]开始测试金融文本分类：[/blue]")
    for i, text in enumerate(test_texts, 1):
        res = inference(text, few_shot_messages)
        print(f"[{i}] 文本：{text}")
        print(f"[red]分类结果：{res}[/red]\n")