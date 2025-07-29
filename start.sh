#!/bin/bash

# GPU Communication 啟動腳本
# 用於執行 NXIL sharding 相關的腳本

echo "=== GPU Communication NXIL Sharding Launcher ==="
echo "開始執行 NXIL sharding 腳本..."

# 檢查 nxil_script.sh 是否存在
if [ -f "nxil_script.sh" ]; then
    echo "找到 nxil_script.sh，執行中..."
    chmod +x nxil_script.sh
    ./nxil_script.sh
elif [ -f "nxil_sharding.py" ]; then
    echo "找到 nxil_sharding.py，執行中..."
    python3 nxil_sharding.py
else
    echo "錯誤：找不到 nxil_script.sh 或 nxil_sharding.py"
    echo "請確認檔案存在於當前目錄中"
    exit 1
fi

echo "=== 執行完成 ==="
