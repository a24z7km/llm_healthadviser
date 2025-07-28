# ===================================================================
# ■ セル1：環境構築 兼 自動リセットセル (最終版)
# ===================================================================
#
# このセルを実行すると、最後に自動でランタイムが再起動します。
# 「ランタイムがクラッシュしました」という表示が出たら成功です。
#
# -------------------------------------------------------------------
import os

print("🔄 ステップ1/3: 既存の競合ライブラリを強制アンインストールします...")
!pip uninstall -y numpy pandas datascience thinc opencv-python opencv-contrib-python opencv-python-headless torch torchvision torchaudio
!pip uninstall -y transformers peft trl accelerate datasets bitsandbytes huggingface_hub pypdf langchain langchain-community

print("\n🚀 ステップ2/3: 必要なライブラリをバージョン指定でインストールします...")
# numpyを他のライブラリより先にインストールすることで競合を回避
!pip install -q "numpy==1.26.4"
!pip install -q \
  "pandas==2.2.2" \
  "datasets==2.19.2" \
  "langchain==0.2.1" \
  "langchain-community==0.2.1" \
  "transformers==4.41.2" \
  "peft==0.11.1" \
  "trl==0.8.6" \
  "accelerate==0.30.1" \
  "bitsandbytes==0.43.1" \
  "huggingface_hub>=0.22.2" \
  "torch==2.3.0" \
  "pypdf==4.2.0" \
  --extra-index-url https://download.pytorch.org/whl/cu121

print("\n💥 ステップ3/3: ライブラリを正しく読み込むため、ランタイムを自動的に再起動します...")
print("「ランタイムがクラッシュしました」と表示されたら、このセルの処理は正常に完了です。次のセルに進んでください。")

# ランタイムを強制的に再起動して、インストールしたライブラリをクリーンな状態でロードさせる
os.kill(os.getpid(), 9)
