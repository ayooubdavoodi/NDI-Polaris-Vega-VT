


################ Polaris Vega VT ######################

# our system IP is 169.254.7.240


ping 169.254.7.143


cd ~/Polaris_Vega_VT/Combined_API_Sample_C++_v1.9.7/CombinedAPIsample
make clean || true
make

cd ~/Polaris_Vega_VT/Combined_API_Sample_C++_v1.9.7/CombinedAPIsample
./bin/linux/capisample 169.254.7.143 --tools="$(pwd)/sroms/Kia_phantom_marker.rom" 



cd ~/Polaris_Vega_VT/Combined_API_Sample_C++_v1.9.7/CombinedAPIsample/build/linux
LD_LIBRARY_PATH="$(pwd)" ./ardemo 169.254.7.143 "$(pwd)/../../sroms/" "Kia_phantom_marker.rom" 554


pip install scikit-surgerynditracker opencv-python numpy



python3 src/vega_rtsp_overlay.py
