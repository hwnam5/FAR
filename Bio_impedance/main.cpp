/*******************************************************************************
 * Copyright (C) 2016 Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *******************************************************************************
 */
#include "mbed.h"
#include "max32630fthr.h"
#include "USBSerial.h"
#include "RpcServer.h"
#include "StringInOut.h"
#include "Peripherals.h"
#include "MAX30001.h"
#include "DataLoggingService.h"
#include "PushButton.h"
#include "USBSerial.h"
#include "Streaming.h"
#include "SDFileSystem.h"
#include "version.h"


//Init PMIC on FTHR board and set logic thresholds to 3.3V
MAX32630FTHR pegasus(MAX32630FTHR::VIO_3V3);

SDFileSystem sd(P0_5, P0_6, P0_4, P0_7, "sd");  // mosi, miso, sclk, cs

///
/// wire Interfaces
///
/// Define with Maxim VID and a Maxim assigned PID, set to version 0x0001 and non-blocking
USBSerial usbSerial(0x0b6a, 0x7531, 0x0001, false);


//SD card insertion detection pin
DigitalIn SDDetect(P2_2, PullUp);


/// DigitalOut for CS
DigitalOut cs(P5_6);
/// SPI Master 2 with SPI0_SS for use with MAX30001
SPI spi(SPI2_MOSI, SPI2_MISO, SPI2_SCK); // used by MAX30001
/// SPI Master 1
QuadSpiInterface quadSpiInterface(SPI1_MOSI, SPI1_MISO, SPI1_SCK,
                                  SPI1_SS); // used by S25FS512
///Debug port
Serial debug(USBTX, USBRX);

///
/// Devices
///

/// External Flash
S25FS512 s25fs512(&quadSpiInterface);
/// ECG device
MAX30001 max30001(&spi, &cs);
InterruptIn max30001_InterruptB(P5_5);
InterruptIn max30001_Interrupt2B(P5_4);


/// HSP platform LED
HspLed hspLed(LED_RED);
/// Packet TimeStamp Timer, set for 1uS
Timer timestampTimer;
/// HSP Platform push button
PushButton pushButton(SW1);

int main() {
  
  //boost baudrate so we can get messages while running gui
  debug.baud(115200);
  
  // local input state of the RPC
  int inputState;
  // RPC request buffer
  char request[128];
  // RPC reply buffer
  char reply[128];

  // display start banner
  debug.printf("Maxim Integrated mbed hSensor %d.%d.%d %02d/%02d/%02d\n", 
    VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, 
    VERSION_MONTH, VERSION_DAY, VERSION_SHORT_YEAR);
  fflush(stdout);

  // turn on red led
  debug.printf("Init HSPLED...\n");
  fflush(stdout);
  hspLed.on();

  // set NVIC priorities for GPIO to prevent priority inversion
  debug.printf("Init NVIC Priorities...\n");
  fflush(stdout);
  NVIC_SetPriority(GPIO_P0_IRQn, 5);
  NVIC_SetPriority(GPIO_P1_IRQn, 5);
  NVIC_SetPriority(GPIO_P2_IRQn, 5);
  NVIC_SetPriority(GPIO_P3_IRQn, 5);
  NVIC_SetPriority(GPIO_P4_IRQn, 5);
  NVIC_SetPriority(GPIO_P5_IRQn, 5);
  NVIC_SetPriority(GPIO_P6_IRQn, 5);
  // used by the MAX30001
  NVIC_SetPriority(SPIM2_IRQn, 0);

  // Be able to statically reference these devices anywhere in the application
  Peripherals::setS25FS512(&s25fs512);
  Peripherals::setUSBSerial(&usbSerial);
  Peripherals::setTimestampTimer(&timestampTimer);
  Peripherals::setHspLed(&hspLed);
  Peripherals::setPushButton(&pushButton);
  Peripherals::setMAX30001(&max30001);
  Peripherals::setSdFS(&sd);
  Peripherals::setSDDetect(&SDDetect);

  // init the S25FS256 external flash device
  debug.printf("Init S25FS512...\n");
  fflush(stdout);
  s25fs512.init();

  // start blinking led1
  debug.printf("Init HSPLED Blink...\n");
  fflush(stdout);
  hspLed.blink(1000);

  //
  // MAX30001
  //
  debug.printf("Init MAX30001 callbacks, interrupts...\n");
  fflush(stdout);
  max30001_InterruptB.disable_irq();
  max30001_Interrupt2B.disable_irq();
  max30001_InterruptB.mode(PullUp);
  max30001_InterruptB.fall(&MAX30001Mid_IntB_Handler);
  max30001_Interrupt2B.mode(PullUp);
  max30001_Interrupt2B.fall(&MAX30001Mid_Int2B_Handler);
  max30001_InterruptB.enable_irq();
  max30001_Interrupt2B.enable_irq();
  MAX30001_AllowInterrupts(1);
  max30001.max30001_sw_rst(); // Do a software reset of the MAX30001
  max30001.max30001_INT_assignment(MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,   MAX30001::MAX30001_NO_INT,  //  en_enint_loc,      en_eovf_loc,   en_fstint_loc,
                                     MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_NO_INT,  //  en_dcloffint_loc,  en_bint_loc,   en_bovf_loc,
                                     MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_NO_INT,  //  en_bover_loc,      en_bundr_loc,  en_bcgmon_loc,
                                     MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,   MAX30001::MAX30001_NO_INT,  //  en_pint_loc,       en_povf_loc,   en_pedge_loc,
                                     MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,  //  en_lonint_loc,     en_rrint_loc,  en_samp_loc,
                                     MAX30001::MAX30001_INT_ODNR, MAX30001::MAX30001_INT_ODNR);                            //  intb_Type,         int2b_Type)
  max30001.onDataAvailable(&StreamPacketUint32);

  // initialize the RPC server
  debug.printf("Init RPC Server...\n");
  fflush(stdout);
  RPC_init();
  // initialize the logging service
  debug.printf("Init LoggingService...\n");
  fflush(stdout);
  LoggingService_Init();
  // initialize the SD disk
  sd.disk_initialize();

  // start main loop
  debug.printf("Start main loop...\n");
  fflush(stdout);

  // Initialize BIOZ impedance measurement
  
  while (1) 
  {
    // get a RPC string if one is available
    inputState = getLine(request, sizeof(request));
    // if a string has been captured, process string
    if (inputState == GETLINE_DONE) 
    {
      //Send request to debug port
      debug.printf(request);
      // process the RPC string
      RPC_call(request, reply);
      //Send reply to debug port
      debug.printf(reply);
      // output the reply
      putStr(reply);
    }
    
    // process any logging or streaming requests
    LoggingService_ServiceRoutine();
  }
}

