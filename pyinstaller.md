mkdir pyinstaller
cd pyinstaller
pip install pyinstaller
pyinstaller ../main.py
cp -r ./dist/main/* ./build/main/

./build/main/main --config_path="/home/wenzihao/Desktop/WandCalibration/config/config.yaml"