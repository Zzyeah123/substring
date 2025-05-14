import subprocess
import itertools

# 固定参数部分，确保顺序和程序预期一致
base_command_template = (
    "python run.py --model DREAM --dname {dname} --substrings-num 5 "
    "--loss-coefficient 1.0 --complex-rate 0.5 "
    "--flag 1 --p-train {p_train} --p-val 0.1 --p-test 0.1 --seed 2 "
    "--l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 "
    "--max-epoch 100 --patience 5 --max-d 3 --max-char 200 "
    "--bs 32 --h-dim 512 --es 100 --clip-gr 10.0"
)

# 参数取值范围
dname_values = ["DBLP", "GENE"]
train_value=[0.01,0.09,0.18,0.35,0.7,1.0]

# 遍历所有参数组合
for dname,p_train in itertools.product(
    dname_values, train_value
):
    # 根据模板填充参数，保证选项顺序
    full_command = base_command_template.format(
        dname=dname,
        p_train=p_train
    )
    print(f"Running command: {full_command}")

    # 执行命令
    process = subprocess.run(full_command, shell=True)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
