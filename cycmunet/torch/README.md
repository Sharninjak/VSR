# PyTorch implementation of CycMuNet+

This is the PyTorch implementation of CycMuNet+.

## Installation

```bash
conda create -f environment.yml
```

## Contents

### `train.py`

Train the network.

### `test.py`

Test the network accuracy.

### `export_onnx.py`

Export trained network to ONNX format. 


cycmunet-main/torch/
├── README.md             # 项目说明文档
├── environment.yml       # 环境配置文件
├── model/                # 模型定义
│   ├── __init__.py       # 模型入口
│   ├── part.py           # 基础网络组件
│   ├── util.py           # 工具函数
│   └── cycmunet/         # CycMuNet特定模块
│       ├── __init__.py   # CycMuNet主类定义
│       ├── part.py       # CycMuNet特有的网络构建块
│       └── module.py     # 包含更多模块
├── cycmunet/             # 配置和运行相关代码
│   ├── model.py          # 模型参数定义
│   └── run.py            # 训练和测试参数定义
├── cycmunet_train.py     # 训练脚本
├── cycmunet_test.py      # 测试模型的脚本
└── cycmunet_export_onnx.py  # 导出ONNX模型的脚本