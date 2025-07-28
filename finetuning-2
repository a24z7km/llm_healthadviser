# ===================================================================
# ■ セル2：ミニプロダクトセル (ファイル読込・APIキー修正版)
# ===================================================================
#
# ColabにアップロードしたPDFを読み込み、データセット生成から
# ファインチューニングまでを一貫して行います。
#
# -------------------------------------------------------------------

import os
import gc
import torch
import json
from datasets import load_dataset
from google.colab import userdata
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. 初期設定とHugging Faceへのログイン ---
print("🔑 Hugging Faceへのログインを開始します...")

try:
    # ユーザー指定のシークレット名 'TEST2' を使用
    hf_token = userdata.get('TEST2')
    if not hf_token:
        raise ValueError("Hugging Faceのトークンが設定されていません。")
    login(token=hf_token)
    print("✅ Hugging Faceへのログインが完了しました。")
except Exception as e:
    print(f"❌ セットアップ中にエラー: {e}")
    print("Colabのシークレットに 'TEST2' が正しく設定されているか確認してください。")
    raise e

# --- 2. モデルとリポジトリの設定 ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 必ずあなたのHugging Faceユーザー名とモデル名に変更してください ★
# ★ (例: "your-hf-username/My-Sleep-Advisor-v1")          ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
NEW_MODEL_REPO_ID = "a24z7km/Llama3-Health-Advisor-v5" # ← ここを書き換える

FINETUNE_TARGET_MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
DATA_GENERATION_MODEL_ID = "elyza/Llama-3-ELYZA-JP-8B"
OUTPUT_DIR = "./results"


# --- 3. ローカルPDFファイルからの学習データセット自動生成 ---
print("\nローカルPDFファイルから学習データを自動生成します...")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 必ずColabにアップロードしたご自身のPDFファイル名に変更してください ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
PDF_FILE_PATH = "健康日本21(第三次)推進のための説明資料①.pdf" # ← ここを書き換える

if not os.path.exists(PDF_FILE_PATH):
    raise FileNotFoundError(f"ファイルが見つかりません: '{PDF_FILE_PATH}'。ColabにPDFファイルをアップロードしたか、ファイル名が正しいか確認してください。")

print(f"📚 '{PDF_FILE_PATH}' を読み込んでいます...")
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)
print(f"✅ テキストを {len(split_docs)}個のチャンクに分割しました。")

# 3-2. データ生成用LLMのロード
print(f"\n🤖 データ生成用のLLM ({DATA_GENERATION_MODEL_ID}) をロードしています...")
quantization_config_gen = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# 正しいトークン変数 'hf_token' を使用
generator_tokenizer = AutoTokenizer.from_pretrained(DATA_GENERATION_MODEL_ID, token=hf_token)
generator_model = AutoModelForCausalLM.from_pretrained(
    DATA_GENERATION_MODEL_ID,
    quantization_config=quantization_config_gen,
    device_map="auto",
    token=hf_token
)

pipe = pipeline("text-generation", model=generator_model, tokenizer=generator_tokenizer, torch_dtype=torch.bfloat16)

# 3-3. ローカルLLMによるQ&Aペアの生成
print("\n🤖 ローカルLLMを使ってQ&Aペアを生成します...（時間がかかります）")
qa_generation_prompt_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
あなたは、提供されたコンテキスト情報のみに基づいて、質の高いQ&Aペアを1つだけ作成する専門家です。一般的な知識やコンテキストにない情報は絶対に追加せず、JSON形式で出力してください。
JSON形式:
{ "input": "生成した質問", "output": "生成した回答" }<|eot_id|>
<|start_header_id|>user<|end_header_id|>
コンテキスト情報:
{context}
上記のコンテキストのみを使い、実践的で具体的な健康に関するQ&AペアをJSON形式で作成してください。<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
qa_pairs = []
for i, doc in enumerate(split_docs):
    print(f"  - チャンク {i+1}/{len(split_docs)} を処理中...")
    prompt = qa_generation_prompt_template.format(context=doc.page_content)
    try:
        terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=pipe.tokenizer.eos_token_id)
        result_text = outputs[0]['generated_text'][len(prompt):].strip()
        json_part = result_text[result_text.find('{'):result_text.rfind('}')+1]
        json.loads(json_part)
        qa_pairs.append(json_part)
    except Exception as e:
        print(f"    ⚠️ チャンク {i+1} の処理中にエラーまたは無効なJSONが生成されました: {e}")
        continue

# 3-4. データセットファイルの保存
DATASET_FILE = "generated_sleep_dataset.jsonl"
print(f"\n💾 生成したQ&Aペアを '{DATASET_FILE}' に保存しています...")
with open(DATASET_FILE, "w", encoding="utf-8") as f:
    for pair_json_str in qa_pairs:
        f.write(pair_json_str + "\n")
print("✅ データセットの自動生成が完了しました。")


# 3-5. メモリ解放
print("\n🧹 データ生成用モデルのメモリを解放します...")
del pipe, generator_model, generator_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("✅ メモリを解放しました。")

# --- 4. ファインチューニングの準備 ---
print(f"\n✨ ファインチューニング対象モデル ({FINETUNE_TARGET_MODEL_ID}) の準備を開始します。")

# 4-1. データセットの準備と整形
print("🥣 ファインチューニング用にデータセットを準備しています...")
def format_dataset_phi3(example):
    if not example.get("input") or not example.get("output"): return None
    return {"text": f"<|user|>\n{example['input']}<|end|>\n<|assistant|>\n{example['output']}<|end|>"}

dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
formatted_dataset = dataset.filter(lambda example: example.get("input") and example.get("output"))
formatted_dataset = formatted_dataset.map(format_dataset_phi3)
print(f"✅ データセットの準備と整形が完了しました。学習データ数: {len(formatted_dataset)}")

# 4-2. モデルとトークナイザーの準備
print("📦 ベースモデルとトークナイザーをロードしています...")
quantization_config_ft = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(FINETUNE_TARGET_MODEL_ID, quantization_config=quantization_config_ft, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(FINETUNE_TARGET_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
print("✅ モデルとトークナイザーのロードが完了しました。")

# --- 5. ファインチューニングの実行 ---
print("\n🔥 ファインチューニングを開始します...")
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'])
training_args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4, num_train_epochs=3, logging_steps=10, report_to="none", save_strategy="no")
trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=formatted_dataset, peft_config=peft_config, dataset_text_field="text", max_seq_length=2048)
trainer.train()
print("✅ ファインチューニングが完了しました。")

# --- 6. モデルのアップロード ---
print(f"\n🚀 学習済みモデルをリポジトリ '{NEW_MODEL_REPO_ID}' にアップロードしています...")
try:
    # 正しいトークン変数 'hf_token' を使用
    trainer.model.push_to_hub(NEW_MODEL_REPO_ID, private=True, token=hf_token)
    trainer.tokenizer.push_to_hub(NEW_MODEL_REPO_ID, private=True, token=hf_token)
    print(f"✅✅✅ アップロード完了！ https://huggingface.co/{NEW_MODEL_REPO_ID} を確認してください。")
except Exception as e:
    print(f"❌ アップロード中にエラーが発生しました: {e}")

# --- 7. 推論テスト ---
print("\n🧪 ファインチューニング後のモデルで推論をテストします...")
prompt = "最近、仕事のプレッシャーで夜中に目が覚めてしまいます。どうしたら良いでしょうか？"
chat = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
output_ids = model.generate(input_ids.to(model.device), max_new_tokens=300, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7, top_p=0.9)
response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

print("\n【質問】")
print(prompt)
print("\n【モデルの回答】")
print(response)

print("\n🎉🎉🎉 すべてのプロセスが完了しました！ 🎉🎉🎉")
