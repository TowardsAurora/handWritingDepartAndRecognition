import torch

f = open('./500.txt', encoding='utf-8')
data = []
for line in f:
    data.append(line.strip())

character_map = {}
for i in range(500):
    character_map.setdefault(data[i // 25][i % 25],i)
print(character_map)


def encode(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将文本转换为数字序列
    numbers = [character_map[c] for c in text]
    # 转换为张量
    tensor = torch.tensor(numbers, dtype=torch.float)
    # 增加批量维度
    tensor = tensor.unsqueeze(0)
    # 转换为 GPU 张量
    tensor = tensor.to(device)
    return tensor