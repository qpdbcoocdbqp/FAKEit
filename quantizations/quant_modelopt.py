from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader


model_path = "Qwen/Qwen3-0.6B"

for quantization in ["modelopt_fp8", "modelopt_fp4"]:
    model_config = ModelConfig(
        model_path=model_path,
        quantization=quantization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    load_config = LoadConfig(
        modelopt_export_path=f"./quantizations/models--{quantization}",
        modelopt_checkpoint_save_path="./checkpoint.pth",
    )
    device_config = DeviceConfig(device="cuda")
    model_loader = get_model_loader(load_config, model_config)

    quantized_model = model_loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )
