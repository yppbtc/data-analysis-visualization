"""
金融信息抽取（少样本学习 + Schema约束）
调用本地Qwen2.5模型，按预定义Schema抽取金融实体，
利用Few-Shot示例引导JSON格式化输出，正则清洗解析结果
"""
import json
import ollama
import re
# 定义不同实体下的具备属性（抽取字段规范）
schema = {
    '金融': ['日期', '股票名称', '开盘价', '收盘价', '成交量'],
}
# 信息抽取提示词模板（核心Prompt！）
IE_PATTERN = "{}\n\n提取上述句子中{}的实体,并按照JSON格式输出,上述句子中不存在的信息用['原文中未提及']来表示,多个值之间用','分隔｡"
# 提供一些例子供模型参考（Few-Shot学习素材）
ie_examples = {
    "金融":[
        {"content":'2023-01-10，股市震荡。股票古哥-D[EOOE]美股今⽇开盘价100美元，⼀度飙升⾄105美元，随后回落⾄98美元，最终以102美元收盘，成交量达到520000。',
        'answers': {
                '日期': ['2023-01-10'],
                '股票名称': ['古哥-D[EOOE]美股'],
                '开盘价': ['100美元'],
                '收盘价': ['102美元'],
                '成交量': ['520000'],
                }
        }
    ]
}

def init_prompts():
    # 初始化前置prompt，做InContext Learning
    ie_pre_history = [{"role": "system", "content": "你是一个信息抽取助手｡"}, ]

    # 双层循环：遍历类别 → 遍历示例
    for label, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            # 拼接schema字段
            properties_str = ', '.join(schema[label])
            schema_str_list = f'“{label}”({properties_str})'
            # 用模板生成提问
            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)
            # 添加【用户提问】
            ie_pre_history.append({"role": "user", "content": f'{sentence_with_prompt}'})
            # 添加【模型回答】（JSON格式标准答案）
            ie_pre_history.append(
                {"role": "assistant", "content": f"{json.dumps(example['answers'], ensure_ascii=False)}"})
    return {'ie_pre_history': ie_pre_history}

def clean_response(response: str):
    # 后处理模型输出，提取纯JSON
    if '```json' in response:
        # 正则提取```json```包裹的内容
        res = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if len(res) and res[0]:
            response = res[0]
        response.replace('､', ',')
    try:
        # 转成JSON字典
        return json.loads(response)
    except:
        return response


def inference(sentences: list, custom_settings: dict):
    for sentence in sentences:
        cls_res = "金融"  # 固定抽取金融类
        properties_str = ', '.join(schema[cls_res])
        schema_str_list = f'“{cls_res}”({properties_str})'
        # 拼接提示词
        sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)

        # 调用Ollama模型
        response = ollama.chat(
            model="qwen2.5:0.5b",  # 你换成自己的：qwen2.5:0.5b
            messages=[
                *custom_settings['ie_pre_history'],  # 少样本上下文
                {"role": "user", "content": sentence_with_ie_prompt}  # 待抽取句子
            ]
        )
        # 获取结果 + 清理格式
        res_content = response["message"]["content"]
        ie_res = clean_response(res_content)
        # 打印输出
        print(f'>>> sentence: {sentence}')
        print(f'>>> inference answer: {ie_res}')

if __name__ == "__main__":
    # 待抽取的测试句子
    sentences = [
        '2023-02-15,股票佰笃[BD]美股开盘价10美元,最终以13美元收盘,成交量460,000｡',
        '2023-04-05,股票盘古(0021)开盘价23元,最终以26美元收盘,成交量310,000｡',
    ]
    # 初始化少样本上下文
    custom_settings = init_prompts()
    # 开始推理
    inference(sentences, custom_settings)