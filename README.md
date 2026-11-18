# Deployment of a Neural Network on the Crazyflie STM32 Processor

## Hardware Requirements

* Crazyflie 2.x
* Crazyradio dongle
* USB-A to Micro-USB cable
* Laptop

## Software Requirements

* Conda / Python Venv
* Git
* [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html)
* Linux / WSL

---

## Setup Before the Workshop

This workshop requires a Python environment on your local machine and Linux / Wsl to build the Crazyflie firmware.

In addition to Docker and Python, this workshop will investigate the use of **ST Edge AI**, a tool developed by STMicroelectronics that enables the easy integration of Neural Networks into ST hardware. In particular, ST Edge AI can take *ONNX* models exported from common Deep Learning frameworks as input and convert them into optimized C code. The tool provides exported code with weights and operations, while the core functionalities are implemented in a runtime library from ST.

To execute the code for this workshop, some initial setup is necessary.

> **Note for Windows Users:** If you are working on a Windows machine, please follow the **Windows Troubleshooting** instructions at the bottom of this document *first* to ensure that your Crazyflie dongle is recognized by Windows, as the OS may default to the wrong drivers for the Crazyflie and Crazyradio.

Once the Crazyradio is available, follow these steps:

1. Clone the workshop repository:

```bash
git clone --recursive https://github.com/FiloSamo/lesson_ann_crazyflie.git
git submodule update --init --recursive
cd lesson_ann_crazyflie
```

2. Create and activate a Conda environment, then install the required packages:

```bash
conda create -n crazy python=3.14
conda activate crazy
conda install pip
pip install -r requirements.txt

```

- alternative:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Install the packages required to build the crazyflie firmware
```bash
sudo apt update
sudo apt-get install make gcc-arm-none-eabi \
    gcc \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libgl1 \
    libfontconfig1 \
    libxkbcommon-x11-0
```

4. Install STEdgeAI-core *(Instructions for installing ST Edge AI manually can be found [HERE](https://www.st.com/en/development-tools/stedgeai-core.html#section-get-software-table).
```bash
chmod +x stedgeai-linux-onlineinstaller
sudo ./stedgeai-linux-onlineinstaller -t /opt/ST/STEdgeAI --accept-licenses --accept-messages --confirm-command --verbose install stedgeai0400.stm32mcu
```
    - Export the environment variables:
```bash
echo 'export PATH="$PATH:/opt/ST/STEdgeAI/4.0/Utilities/linux"' >> ~/.bashrc
echo 'export STEDGEAI_CORE_DIR="/opt/ST/STEdgeAI"' >> ~/.bashrc
source ~/.bashrc
```

---

## Workshop - How to Compile and Flash Our Project

Once you have trained an ANN for MNIST classification and exported it to the *ONNX* format, it is time to convert it into optimized C code to run on the Crazyflie's Cortex-M4.

1. To generate the C files required to run the Neural Network within our application, we need to use the `stedgeai` tool, passing our *ONNX* file as an argument. This command generates the optimized code necessary to integrate the ANN into the `app` application for the Crazyflie.

```bash
stedgeai generate --model ./output/model.onnx --output ./st_ai_ws --optimization balanced --target stm32 --no-inputs-allocation --no-outputs-allocation

```

2. Move the generated output to the `app/src` directory:

```bash
mv ./st_ai_ws/ ./app/src/

```

3. Next, copy the Cortex-M4 runtime library into the `st_ai_ws` folder:

```bash
cp ${STEDGEAI_CORE_DIR}/4.0/Middlewares/ST/AI/Lib/GCC/ARMCortexM4/NetworkRuntime1201_CM4_GCC.a ./app/src/st_ai_ws/NetworkRuntime1201_CM4_GCC.a

```

4. Create a Kbuild file inside the *st_ai_ws* folder to include the generated files in the compilation process:

```bash
echo "obj-y += network.o network_data.o" > ./app/src/st_ai_ws/Kbuild

```
5. With this initial setup complete, we are ready to compile the firmware:

```bash
cd app
make cf2_defconfig
make -j$(nproc)

```

6.  Navigate to the *app* directory on your local machine:

```bash
cd app

```

7. Flash the Crazyflie firmware. With the Crazyflie turned off, press and hold the power button for three seconds until the blue LEDs light up to enter bootloader mode. Then, run the following command:

```bash
cfloader flash build/cf2.bin stm32-fw
```

8. To run the inference, execute the *inference_server.py* script. The server will send MNIST test samples to the Crazyflie and wait for its predictions.

```bash
cd ..
python inference_server.py
```

---

## Troubleshooting

### Windows

To communicate with the Crazyflie, the laptop must recognize the Crazyradio dongle. On native Linux, the dongle should work without issues. However, if you are using Windows (or WSL on Windows), you need to check which drivers the system is assigning to the Crazyradio. Bitcraze's products typically use the *libusb* driver. To verify the driver in use, follow these instructions:

1. Connect the Crazyradio.
2. Go to **Control Panel > Device Manager**.
3. Check if the Crazyradio device is listed under *libusb-win32 devices*.
4. If it is not, download Zadig from [http://zadig.akeo.ie/](http://zadig.akeo.ie/).
5. Select the Crazyradio device and the *libusb-win32* driver, then install the drivers. *(Note: If you cannot see the Crazyradio, go to **Options > List All Devices** in Zadig).*

For reference, see the official documentation: [USB on Windows (Bitcraze)](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/)

---

## Authors

![GIF](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGl1OGVqY2hzcHM5ZXB4bzgyM3M4b2psaG81MHNwOWl5NW45NTg0aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QXwtfadqo7wbfmT46H/giphy.gif)

* **Filippo Samorì** - filippo.samori@unibo.it
