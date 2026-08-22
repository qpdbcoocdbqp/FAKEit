import json, time, urllib.request, wave

body = json.dumps({
    "model": "audio8/tts-0.6b",
    "input": "どの私も私よ 壊れかけの日々さえも 掛け替えない記憶の総てよ 私を私たらしめてよ",
    "references": [
        {
            "audio_path": "/mnt/c/Users/siao/iloveit/FAKEit/docs/tts/resource/reference.wav",
            "text": "突然轉錯帳可能是某個系統整個當掉結果回頭一查才發現寫這段的是AI而唯一該把關的人從頭到尾沒看過他那這個鍋到底該誰扛"
        }
    ],
    "response_format": "wav",
    "temperature": 0.2,
}).encode()

req = urllib.request.Request(
    "http://localhost:8010/v1/audio/speech",
    data=body,
    headers={"Content-Type": "application/json"},
)

t0 = time.perf_counter()
with urllib.request.urlopen(req) as resp:
    wav_bytes = resp.read()
elapsed = time.perf_counter() - t0

output_path = "output.wav"
with open(output_path, "wb") as f:
    f.write(wav_bytes)

with wave.open(output_path) as wf:
    audio_duration = wf.getnframes() / wf.getframerate()

rtf = elapsed / audio_duration
print(f"generation : {elapsed:.3f}s")
print(f"audio      : {audio_duration:.3f}s")
print(f"RTF        : {rtf:.4f}  ({'faster' if rtf < 1 else 'slower'} than real-time)")
