import pandas as pd

# 中文省份到英文的映射表
province_map = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "上海": "Shanghai",
    "重庆": "Chongqing",
    "河北": "Hebei",
    "山西": "Shanxi",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "海南": "Hainan",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "中国台湾": "Taiwan",
    "内蒙古": "Inner Mongolia",
    "广西": "Guangxi",
    "西藏": "Tibet",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "中国香港": "Hong Kong",
    "中国澳门": "Macau"
}

# 读取数据
df = pd.read_csv('ip.csv')  # 替换成你的文件路径

# 替换中文省份为英文
df['ip'] = df['ip'].map(province_map)

# 保存结果
df.to_csv('translated_data.csv', index=False)
print("✅ 地名已翻译完成，保存为 translated_data.csv")
