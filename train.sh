label_rate=(0.5 1 2)

for A in "${label_rate[@]}"
do
          # 执行 Python 命令
      python run.py --label_rate $A
#      python run_m.py --label_rate $A
done