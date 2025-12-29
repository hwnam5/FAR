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
 
 // 이진수 문자열 생성 헬퍼 함수 (4자리)
 void uint8_to_bin_str(uint8_t value, char* bin_str) {
   bin_str[0] = ((value >> 3) & 1) ? '1' : '0';
   bin_str[1] = ((value >> 2) & 1) ? '1' : '0';
   bin_str[2] = ((value >> 1) & 1) ? '1' : '0';
   bin_str[3] = (value & 1) ? '1' : '0';
   bin_str[4] = '\0';
 }
 
 // 데이터가 들어오면 자동으로 실행될 함수
 void dumpBioZData(uint32_t id, uint32_t *buffer, uint32_t length) {
   // 들어온 데이터가 BioZ 데이터인지 확인 (ID: 0x33)
   if (id == MAX30001_DATA_BIOZ) {
       for (int i = 0; i < length; i++) {
           // BioZ 데이터는 24비트(또는 20비트) 정수형태로 들어옵니다.
           // 보기 좋게 10진수와 16진수로 같이 출력합니다.
           debug.printf("BioZ: %u (0x%06X)\r\n", buffer[i], buffer[i]);
       }
   }
 }
 
 // [추가됨] 측정 시작 함수 (BioZ 설정 및 동기화)
 void startMeasurement(uint8_t fcgen_frequency) {
   debug.printf("측정 시작 (Start Measurement)...\n");
   
   // BIOZ 초기화 및 시작
   int bioz_init_result = max30001.max30001_BIOZ_InitStart(
     0b1, // bioz 채널 활성화
     0b0, 0b0, // bip, bin 입력 스위치 닫기 (측정 모드)
     0b00, 0b00, // 캘리브레이션 신호 없음
     0b00, // Unchopped, LPF 포함 (추천 모드)
     7,    // FIFO Threshold (8개 쌓이면 인터럽트)
     0b0,  // data rate : 64sps
     0b010, // Analog HPF: 800Hz
     0b0,  // 외부 저항 바이어스 Off
     0b10, // Gain: 40 V/V
     0b00, // Digital HPF Off
     0b00, // Digital LPF Off
     fcgen_frequency, // 입력받은 자극 주파수 사용
     0b0,  // Current Generator Monitor Off
     0b111, // 전류 세기: 96uA (최대)
     0b0,  // Phase Offset Off
     0b0   // Power Mode: Low Power
   );
 
   if (bioz_init_result != 0) {
     debug.printf("BIOZ 초기화 실패 에러코드: %d\n", bioz_init_result);
   } else {
     // 동기화 (타이밍 리셋 및 실제 시작)
     max30001.max30001_synch();
     debug.printf("BioZ 측정 시작됨! (데이터가 출력됩니다)\n");
   }
 }
 
 // [추가됨] 측정 정지 함수 (BioZ 기능 Off)
 void stopMeasurement() {
   debug.printf("측정 정지 (Stop Measurement)...\n");
   // BioZ 기능만 끔 (전력 절약)
   max30001.max30001_Stop_BIOZ();
   debug.printf("BioZ 측정 중지됨.\n");
 }
 
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
 
   hspLed.on();
 
   // set NVIC priorities
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
   s25fs512.init();
   hspLed.blink(1000);
 
   //
   // MAX30001 Setup
   //
   debug.printf("Init MAX30001 callbacks, interrupts...\n");
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
 
   // [중요] 인터럽트 설정: BioZ 인터럽트(BINT)를 INT_2B 핀으로 설정
   max30001.max30001_INT_assignment(MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,   MAX30001::MAX30001_NO_INT,  
                                      MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_NO_INT,  // en_bint_loc -> INT_2B
                                      MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_NO_INT,  
                                      MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,   MAX30001::MAX30001_NO_INT,  
                                      MAX30001::MAX30001_INT_2B,   MAX30001::MAX30001_INT_B,    MAX30001::MAX30001_NO_INT,  
                                      MAX30001::MAX30001_INT_ODNR, MAX30001::MAX30001_INT_ODNR);                            
   
   // 데이터 수신 콜백 등록 (우리가 만든 dumpBioZData 함수 연결)
   max30001.onDataAvailable(&dumpBioZData);
 
   // initialize RPC & Logging
   RPC_init();
   LoggingService_Init();
   sd.disk_initialize();
 
   // --------------------------------------------------------------------------
   // [1] 사용자 입력 (자극 주파수 설정)
   // --------------------------------------------------------------------------
   uint8_t fcgen_frequency = 0b0010; // 기본값
   debug.printf("\n=========================================\n");
   debug.printf("BIOZ 자극 주파수를 입력하세요 (0-15, 기본값: 2): ");
   fflush(stdout);
   
   char input_buffer[10];
   int input_freq = -1;
   char bin_str[5];
   if (getLine(input_buffer, sizeof(input_buffer)) == GETLINE_DONE) {
     input_freq = atoi(input_buffer);
     if (input_freq >= 0 && input_freq <= 15) {
       fcgen_frequency = (uint8_t)input_freq;
     }
   }
   uint8_to_bin_str(fcgen_frequency, bin_str);
   debug.printf("설정된 주파수: %d (0b%s)\n", fcgen_frequency, bin_str);
 
   // --------------------------------------------------------------------------
   // [2] 필수 초기화 (Rbias & FMSTR) - BioZ 사용 전 반드시 1회 실행
   // --------------------------------------------------------------------------
   debug.printf("기본 시스템 초기화 (Rbias & FMSTR)...\n");
   max30001.max30001_Rbias_FMSTR_Init(0b01, 0b10, 0b1, 0b1, 0b00);
 
   // 초기 자동 시작 (원하시면 주석 처리하세요)
   startMeasurement(fcgen_frequency);
 
   debug.printf("=========================================\n");
   debug.printf("명령어를 입력하세요:\n");
   debug.printf(" - 'start': 측정 시작\n");
   debug.printf(" - 'stop' : 측정 정지\n");
   debug.printf("=========================================\n");
   fflush(stdout);
 
   // --------------------------------------------------------------------------
   // [3] 메인 루프 (명령어 처리)
   // --------------------------------------------------------------------------
   while (1) 
   {
     // 사용자 입력 대기
     inputState = getLine(request, sizeof(request));
     
     if (inputState == GETLINE_DONE) 
     {
       // 입력된 명령어 확인
       if (strncmp(request, "start", 5) == 0) 
       {
          // 'start' 입력 시 -> 측정 시작
          startMeasurement(fcgen_frequency);
       }
       else if (strncmp(request, "stop", 4) == 0) 
       {
          // 'stop' 입력 시 -> 측정 정지
          stopMeasurement();
       }
       else 
       {
          // 그 외 명령어는 기존 RPC 처리
          RPC_call(request, reply);
          debug.printf(reply);
       }
     }
     
     // 로깅 서비스 처리
     LoggingService_ServiceRoutine();
   }
 }