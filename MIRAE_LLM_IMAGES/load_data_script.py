# 1. 외부망 모델 및 튜닝 데이터셋 저장 스크립트 
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 1-1. 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 1-2. 저장 경로 (경로 바꿔주세요)
save_path = "/model/"

# 1-3. 모델과 토크나이저 저장
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# 1-4. 데이터셋 로드 (데이터셋 설정)
ds = load_dataset("argilla/llama-2-banking-fine-tune")

# 1-5. 데이터셋을 로컬에 저장 (경로 바꿔주세요)
local_dataset_path = "/dataset/"
dataset.save_to_disk(local_dataset_path)
