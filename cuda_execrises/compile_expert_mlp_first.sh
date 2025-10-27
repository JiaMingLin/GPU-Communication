#!/bin/bash
# 編譯 expert_mlp_first.cu 至 output/expert_mlp_first

cd "$(dirname "$0")"

nvcc -std=c++17 expert_mlp_first.cu -o output/expert_mlp_first -O2 -arch=sm_75 --compiler-options -Wall

if [ $? -eq 0 ]; then
    echo "✓ 編譯成功！"
    echo "可執行文件: output/expert_mlp_first"
    echo ""
    echo "使用方式:"
    echo "  ./output/expert_mlp_first [num_experts] [num_tokens] [zero_expert_cnt] [ky] [kx] [d]"
    echo ""
    echo "預設參數: num_experts=6, num_tokens=40, zero_expert_cnt=2, ky=4, kx=3, d=16"
else
    echo "✗ 編譯失敗"
    exit 1
fi

