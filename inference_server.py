# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.ù
#
#########################################################################
#     ______     ______     ______     __         ______     ______    
#    /\  ___\   /\  ___\   /\  ___\   /\ \       /\  __ \   /\  == \   
#    \ \  __\   \ \ \____  \ \___  \  \ \ \____  \ \  __ \  \ \  __<   
#     \ \_____\  \ \_____\  \/\_____\  \ \_____\  \ \_\ \_\  \ \_____\ 
#      \/_____/   \/_____/   \/_____/   \/_____/   \/_/\/_/   \/_____/ 
#                                                                      
# Based on the example from the Crazyflie python library
# Adapted to send MNIST data to the Crazyflie for inference
# Author: Filippo Samorì
# Date: 2026-06-18
# Description: The server sends MNIST images to the Crazyflie over the app channel, and receives the inference results back. 

import logging
import time
from threading import Thread

import struct

import cflib
from cflib.crazyflie import Crazyflie
from utils.utils import load_datasets
import numpy as np

logging.basicConfig(level=logging.ERROR)

DATA_PATH = "./data"
URI = "radio://0/50/2M/E7E7E7E7E8"

def console_received(data):
        '''
        Callback function that is called when a console message is received from the Crazyflie.
        '''
        print(f"CONSOLE: {data}")

class MnistToByte:
    '''
        Utility class to convert MNIST dataset images to byte format for sending over the app channel.
    '''
    def __init__(self, mnist_dataset):
        if mnist_dataset is None:
            raise ValueError("mnist_dataset cannot be None")
        
        self.mnist_dataset = mnist_dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''        
            Get the next image from the MNIST dataset, convert it to byte format, and return it along with the label.
        '''
        if self.index >= len(self.mnist_dataset):
            raise StopIteration
        image_tensor, label = self.mnist_dataset[self.index]
        flattened_image = image_tensor.numpy().flatten()
        byte_image = flattened_image.tobytes()
        self.index += 1
        return byte_image, label
        

class AppchannelTest:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
        self.is_running = True
        self._cf = Crazyflie()

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        self._cf.appchannel.packet_received.add_callback(self._app_packet_received)

        # Enable to see the console output from the Crazyflie
        self._cf.console.receivedChar.add_callback(console_received)
        
        self._cf.open_link(link_uri)
        self._received = False

        print('Connecting to %s' % link_uri)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        print("Generating thread to test app channel...")
        Thread(target=self._test_appchannel).start()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        self.is_running = False
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        self.is_running = False
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        self.is_running = False
        print('Disconnected from %s' % link_uri)

    def _app_packet_received(self, data):
        (result, ) = struct.unpack("<i", data)
        self._received = True
        print(f"Received result: {result}")
        print()

    def close_link(self):
        self._cf.close_link()

    def _test_appchannel(self):
        '''
            Test the app channel by sending MNIST images to the Crazyflie and receiving inference results back.
            This function is run in a separate thread.
        '''
        print("Testing app channel...")
        _ , test_data = load_datasets(DATA_PATH)
        test_data = MnistToByte(test_data)
        for inp, label in test_data:
            print(f"Sending input for label: {label}")
            # slplit the byte data into packets of 7 floats (28 bytes) to send over the app channel
            split = [inp[i:i+28] for i in range(0, len(inp), 28)]
            # Initialize the packet ID for the current MNIST image
            packet_id = 0

            for packet in split:
                payload = packet + struct.pack("<B", packet_id)
                self._cf.appchannel.send_packet(payload)
                packet_id += 1
                time.sleep(0.1)
            while not self._received:
                time.sleep(0.1)
            self._received = False

        self._cf.close_link()


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    # Connect to the Crazyflie and run the app channel test
    le = AppchannelTest(URI)
    try:
        while le.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        le.close_link() 
        print("\nManual interruption (Ctrl+C). Closing...")

