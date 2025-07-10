import pandas as pd
import os
import glob

# 设置你的CSV文件所在文件夹路径
folder_path = 'C:\\Users\\14049\\Desktop\\第五人格'  # 修改为你的文件夹路径

# 获取所有CSV文件路径
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 读取并合并所有CSV文件
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 保存合并后的文件
combined_df.to_csv(os.path.join(folder_path, 'merged_output.csv'), index=False)

print(f"成功合并了 {len(csv_files)} 个文件，输出文件为 'merged_output.csv'")
