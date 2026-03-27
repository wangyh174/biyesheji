import os
import sys

# 设置 Hugging Face 国内镜像源，解决下载慢或网络不通的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("未安装 huggingface_hub，正在安装...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
    from huggingface_hub import snapshot_download

def main():
    repo_id = "runwayml/stable-diffusion-v1-5"
    local_dir = os.path.join(os.path.dirname(__file__), "models", "stable-diffusion-v1-5")
    
    print(f"开始从 hf-mirror.com 下载 {repo_id} 到本地目录: {local_dir}")
    print("由于模型较大 (大约4~5GB)，这可能需要一些时间。如果网络中断会自动重试...")
    
    import time
    while True:
        try:
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                ignore_patterns=["*.ckpt", "*.safetensors"],
            )
            break
        except Exception as e:
            print(f"网络连接中断: {e}")
            print("5秒后自动重试恢复下载...")
            time.sleep(5)
            
    print(f"\n下载完成！模型已保存在: {path}")
    print("接下来你在运行 01_generate.py 时，可以使用参数指定本地路径：")
    print(f"python scripts/01_generate.py --generator diffusers --model-path {local_dir}")

if __name__ == "__main__":
    main()
