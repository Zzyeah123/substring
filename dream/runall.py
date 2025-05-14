import subprocess
import itertools

# 固定参数部分，确保顺序和程序预期一致
base_command_template = (
    "python run.py --model DREAM --dname DBLP --substrings-num {substrings_num} "
    "--loss-coefficient {loss_coefficient} --complex-rate {complex_rate} "
    "--flag 1 --p-train 1.0 --p-val 0.1 --p-test 0.1 --seed 0 "
    "--l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 "
    "--max-epoch 100 --patience 5 --max-d 3 --max-char 200 "
    "--bs 32 --h-dim 512 --es 100 --clip-gr 10.0"
)

# 参数取值范围
substrings_num_values = range(1, 7)  # 从 1 到 7
loss_coefficient_values = [0.5, 1.0,1.5]
complex_rate_values = [1.0, 0.8,0.5,0.2,0.0]

# 遍历所有参数组合
for substrings_num, loss_coefficient, complex_rate in itertools.product(
    substrings_num_values, loss_coefficient_values, complex_rate_values
):
    # 根据模板填充参数，保证选项顺序
    full_command = base_command_template.format(
        substrings_num=substrings_num,
        loss_coefficient=loss_coefficient,
        complex_rate=complex_rate,
    )
    print(f"Running command: {full_command}")

    # 执行命令
    process = subprocess.run(full_command, shell=True)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
