python tools/weights_for_g3flow/download_from_huggingface.py
cd RoboTwin_Benchmark
mv ../tools/weights_for_g3flow/robotwin_assets.zip ./
unzip robotwin_assets.zip
rm robotwin_assets.zip
cd ..