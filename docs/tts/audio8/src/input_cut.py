from pydub import AudioSegment


audio = AudioSegment.from_mp3("./docs/tts/pey6r-116tp.mp3")
clip = audio[13_000:32_500]  # ms
clip.export("./docs/tts/reference.mp3", format="mp3")
