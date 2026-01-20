# OCR 功能安装说明

本插件使用 OCR（光学字符识别）技术来识别图片中的文字内容，以便更好地审核表情包和包含文字的图片。

## 安装步骤

### 1. 安装 Python 依赖

插件会自动安装以下 Python 依赖：
- Pillow >= 10.0.0
- pytesseract >= 0.3.10

### 2. 安装系统依赖

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-sim  # 中文简体支持
sudo apt-get install tesseract-ocr-chi-tra  # 中文繁体支持（可选）
```

#### CentOS/RHEL:
```bash
sudo yum install epel-release
sudo yum install tesseract
sudo yum install tesseract-langpack-chi-sim  # 中文简体支持
```

#### macOS:
```bash
brew install tesseract
brew install tesseract-lang  # 包含中文支持
```

#### Windows:
1. 下载 Tesseract-OCR 安装包：https://github.com/UB-Mannheim/tesseract/wiki
2. 安装时选择中文语言包
3. 将安装路径添加到系统环境变量 PATH

### 3. 验证安装

运行以下命令验证安装：
```bash
tesseract --version
```

如果看到版本信息，说明安装成功。

## 功能说明

OCR 功能会：
1. 自动下载图片
2. 识别图片中的文字内容
3. 对识别的文字进行违规关键词检查
4. 如果发现违规关键词，直接判定为违规

## 注意事项

- OCR 功能是可选的，如果安装失败不会影响其他功能
- OCR 主要用于识别表情包和包含文字的图片
- 系统会记录 OCR 识别的结果供调试使用