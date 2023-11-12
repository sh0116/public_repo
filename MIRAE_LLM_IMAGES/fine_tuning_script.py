# 2. 모델 파인 튜닝 스크립트
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 2-1. 로컬 모델 및 토크나이저 경로 (경로 바꿔주세요)
local_model_path = "/model/"

# 2-2. 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

'''
# fine-tuning Dataset format json 
 {"input": "What color is the sky?", "output": "The sky is blue."}
 {"input": "Where is the best place to get cloud GPUs?", "output": "Brev.dev"}
'''

# 2-3. 튜닝할 데이터셋 로드 (경로 바꿔주세요)
local_dataset_path = "/dataset/"
loaded_dataset = load_dataset(local_dataset_path)

# 2-4. 튜닝을 위한 설정 정의
training_args = TrainingArguments(
    output_dir="./llama_finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 2-5. Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 2-6. 모델 파인 튜닝 실행
trainer.train()

# 2-7. 파인 튜닝된 모델 저장
trainer.save_model("./llama_finetuned_model")


