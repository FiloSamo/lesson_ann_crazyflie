# stedgeai - Crazyflie Application

This project is a Crazyflie application that uses the ST Edge AI API to run inference with a Neural Network (NN) on an STM32 microcontroller.

## Description

The application communicates with a PC via the Crazyflie app channel. Its main purpose is to receive input data, perform the inference, and return the classification result to the PC.

The NN model used is a simple fully connected neural network with one hidden layer, trained on the MNIST dataset (handwritten digit recognition).

## Documentation

The documentation can be found in the [ST Edge AI API documentation](https://stedgeai-dc.st.com/assets/embedded-docs/embedded_client_stai_api.html).

## Detailed Operation

1. **Data Reception**: The application waits for input data sent by the PC through the app channel. Due to packet size limits, the data is sent in small chunks of 7 floating-point values (float) per packet.
2. **Synchronization**: Each packet contains a `packet_id` to keep track of the correct order of the received data. If a desynchronization is detected (e.g., a lost or out-of-order packet), the entire receiving batch is discarded to avoid incorrect inferences.
3. **Inference**: Once a full sample (MNIST image) has been received, the application starts the inference.
4. **Result**: The final inference result corresponds to the index of the output neuron that reached the highest value.
5. **Sending Response**: The classification result (an integer representing the predicted digit) is sent back to the PC via a packet.
6. **Performance**: The application measures and prints on the debug console the time taken to complete the entire inference (in microseconds/milliseconds).

## Configuration and Build

The application integrates into the Crazyflie build system using Kbuild.

The default parameters of the application are defined in `app-config`:
* The application is enabled by default (`CONFIG_APP_ENABLE=y`)
* Task priority: 1
* Allocated stack size: 350

The `Makefile` compiles the application by including the necessary STedgeAI libraries and relies on the main Crazyflie firmware Makefile.
