#!/bin/bash

# 获取所有运行Python进程的用户，去重并计数
count=$(ps aux | awk '
NR>1 { 
    # 提取可执行文件名（去掉路径）
    cmd = $11
    gsub(/.*\//, "", cmd)  # 去掉路径部分
    if (tolower(cmd) ~ /^python/) {
        print $1
    }
}' | sort | uniq | wc -l)

echo "$count"