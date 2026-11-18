/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
  ****************************************************************************                   
 * 
 *     ______     ______     ______     __         ______     ______    
 *    /\  ___\   /\  ___\   /\  ___\   /\ \       /\  __ \   /\  == \   
 *    \ \  __\   \ \ \____  \ \___  \  \ \ \____  \ \  __ \  \ \  __<   
 *     \ \_____\  \ \_____\  \/\_____\  \ \_____\  \ \_\ \_\  \ \_____\ 
 *      \/_____/   \/_____/   \/_____/   \/_____/   \/_/\/_/   \/_____/ 
 *                                                                                                         
 * @author: Filippo Samorì
 * @date: 2026-06-18
 * @description: Example of a Crazyflie application using the ST Edge AI API to run inference with a ANN on the STM32.
 *              The application receives the input data for the ANN from a PC through the app channel, runs the inference and sends the result back to the PC.
 *              The ANN model is a simple fully connected with one hidden layer, trained on the MNIST dataset. 
 *              The input data is sent in chucks of 7 floats for every packet,
 *              and the packet_id is used to keep track of the order of the received data and to detect possible desyncs.
 *              The application also measures the time taken for the inference and prints it on the console.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "app.h"
#include "app_channel.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "CNSL"
#include "debug.h"

#include "st_ai_ws/network.h"

#define SENDED_DATA_SIZE 7

/* Global byte buffer to save instantiated C-model network context */
STAI_ALIGNED(STAI_NETWORK_CONTEXT_ALIGNMENT)
static stai_network network_context[STAI_NETWORK_CONTEXT_SIZE] = {0};

/* Global c-array to handle the activations buffer */
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t activations[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];

/* Array to store the data of the input tensor */
STAI_ALIGNED(STAI_NETWORK_IN_1_ALIGNMENT)
static float in_data[STAI_NETWORK_IN_1_SIZE];
/* or static uint8_t in_data[STAI_NETWORK_IN_1_SIZE_BYTES]; */

/* c-array to store the data of the output tensor */
STAI_ALIGNED(STAI_NETWORK_OUT_1_ALIGNMENT)
static float out_data[STAI_NETWORK_OUT_1_SIZE];
/* static uint8_t out_data[STAI_NETWORK_OUT_1_SIZE_BYTES]; */

struct mnistPacketRX {
  float data[SENDED_DATA_SIZE];
  uint8_t packet_id;
} __attribute__((packed));

struct mnistPacketTX {
  int result;
} __attribute__((packed));

/* Auxiliary functions */

int maxIndex(float* arr, int dim) {
    if (dim <= 0) {
        return -1; 
    }
    int indx_max = 0;

    for (int i = 1; i < dim; i++) {
        if (arr[i] > arr[indx_max]) {
            indx_max = i; 
        }
    }

    return indx_max;
}

// STedgeAI C-model API functions

/* 
 * Bootstrap
 */

int aiInit(void) {
  stai_return_code ret_code;
  
  /* Initialize runtime library */
  ret_code = stai_runtime_init();
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  /* Initialize network model context */
  ret_code = stai_network_init(network_context);
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  /* Set network activations buffers */
  const stai_ptr acts[] = { activations };
  ret_code = stai_network_set_activations(network_context, acts, STAI_NETWORK_ACTIVATIONS_NUM);
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  return 0;
}

int aiDeinit(void) {
  stai_return_code ret_code;

  /* Deinitialize network model context */
  ret_code = stai_network_deinit(network_context);
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  /* Deinitialize runtime library */
  ret_code = stai_runtime_deinit();
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  return 0;
}

/* 
 * Run inference
 */
int aiRun(void *in_data, void *out_data) {
  stai_return_code ret_code;

  /* Set network input/output buffers */
  stai_ptr inputs_ptr[] = { in_data };
  ret_code = stai_network_set_inputs(network_context, inputs_ptr, STAI_NETWORK_IN_NUM);
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };

  stai_ptr outputs_ptr[] = { out_data };  
  ret_code = stai_network_set_outputs(network_context, outputs_ptr, STAI_NETWORK_OUT_NUM);
  if (ret_code != STAI_SUCCESS) { DEBUG_PRINT("STAI Error: %d\n", ret_code); };


  /* 2 - Perform the inference */
  ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
  if (ret_code != STAI_SUCCESS) {
      ret_code = stai_network_get_error(network_context);
      DEBUG_PRINT("STAI Error: %d\n", ret_code);
  };
  
  return 0;
}

void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  aiInit();

  static struct mnistPacketRX rxPacket;
  static struct mnistPacketTX txPacket;

  int counter = 0;
  int expected_packet_id = 0;
  while(1) {

    /* Communication */
    if (appchannelReceiveDataPacket(&rxPacket, sizeof(rxPacket), APPCHANNEL_WAIT_FOREVER)) {
      // receive the MNIST data in chucks of 7 floats for every packet
      // packet_id is used to keep track of the order of the received data and to detect possible desyncs.

      if (rxPacket.packet_id == 0) {
        counter = 0;
        expected_packet_id = 0;
      }

      if (rxPacket.packet_id != expected_packet_id) {
        // if the packet id is not the expected one, it means that we have a desync and we need to reset the counter and the expected packet id
        DEBUG_PRINT("Desync! Desired %d, received %d. Reject ...\n", expected_packet_id, rxPacket.packet_id);
        counter = 0; 
        expected_packet_id = 0;
        continue;
      }

      for (int i = 0; i < SENDED_DATA_SIZE; i++) {
        in_data[counter] = rxPacket.data[i];
        counter++;
      }

      expected_packet_id++;
    }
    /* End Communication */

    if (counter >= STAI_NETWORK_IN_1_SIZE) {
      // once a full input sample has been received, we can run the inference and send the result back to the PC

      counter = 0;
      expected_packet_id = 0;
      txPacket.result = 0;
      DEBUG_PRINT("Inference!\n");
      const uint64_t begin = usecTimestamp();

      aiRun(in_data, out_data);

      txPacket.result = maxIndex(out_data, STAI_NETWORK_OUT_1_SIZE);
      const uint64_t end = usecTimestamp();
        
      const int elapsed = end - begin;
      DEBUG_PRINT("Pred time: %d us\n", elapsed);

      appchannelSendDataPacketBlock(&txPacket, sizeof(txPacket));
    }

  }

  aiDeinit();
}
