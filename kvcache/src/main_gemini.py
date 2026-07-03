import hashlib
from llama_cpp import Llama

class ExternalKVCacheManager:
    def __init__(self):
        # 實務上這裡可以換成 Redis 或 Diskcache
        self.cache_store = {}

    def _generate_key(self, prompt: str) -> str:
        """使用 SHA-256 將 Prompt 轉為唯一的快取 Key"""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def get_cache(self, prompt: str) -> bytes:
        key = self._generate_key(prompt)
        return self.cache_store.get(key, None)

    def set_cache(self, prompt: str, state_data: bytes):
        key = self._generate_key(prompt)
        self.cache_store[key] = state_data
        print(f"✅ 成功將 KV 狀態寫入外部快取。Key: {key[:8]}...")

# 1. 初始化模型（必須啟用核心參數）
# 注意：為了讓快取發揮作用，通常需要固定 seed 或處理好序列長度
llm = Llama(model_path="./your-model.gguf", n_ctx=2048)
cache_mgr = ExternalKVCacheManager()

# 定義一個常見的 System Prompt（我們想快取這個部分的 KV）
system_prompt = "你是一個專業的資安專家，請用繁體中文回答所有問題。"
user_prompt = "什麼是 SQL Injection？"
full_prompt = f"{system_prompt}\nUser: {user_prompt}\nAssistant:"

# 2. 檢查外部快取
cached_state = cache_mgr.get_cache(system_prompt)

if cached_state:
    print("🎯 快取命中！正在載入外部 KV 快取狀態...")
    llm.load_state(cached_state)
else:
    print("❌ 快取未命中。開始初次推論並建立快取...")
    # 先單獨對 System Prompt 進行評估（Evaluate）以生成 KV Cache
    llm.eval(llm.tokenize(system_prompt.encode('utf-8')))
    
    # 將 System Prompt 生成的 KV 狀態匯出並存入外部快取
    system_state = llm.save_state()
    cache_mgr.set_cache(system_prompt, system_state)

# 3. 繼續進行後續的 Token 生成
output = llm(full_prompt, max_tokens=100)
print(output["choices"][0]["text"])