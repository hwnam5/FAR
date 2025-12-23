//
//  MotionManager.swift
//  IMUStreamer
//
//  Created by hwnam on 12/23/25.
//

import Foundation
import CoreMotion
import WatchConnectivity

// 데이터 구조체 정의
struct MotionData {
    let timestamp: Double
    let accX: Double, accY: Double, accZ: Double
    let gyroX: Double, gyroY: Double, gyroZ: Double
}

class MotionManager: ObservableObject {
    private var motionManager = CMMotionManager()
    private var motionDataArray: [MotionData] = [] // 데이터 임시 저장 배열
    
    @Published var accX: Double = 0.0
    @Published var accY: Double = 0.0
    @Published var accZ: Double = 0.0
    @Published var gyroX: Double = 0.0
    @Published var gyroY: Double = 0.0
    @Published var gyroZ: Double = 0.0
    
    let updateInterval = 0.02 // 50Hz로 상향 (데이터 수집용으로 적합)

    func startUpdates() {
        motionDataArray.removeAll() // 시작 시 이전 데이터 삭제
        
        if motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable {
            motionManager.accelerometerUpdateInterval = updateInterval
            motionManager.gyroUpdateInterval = updateInterval
            
            // 데이터 수집 시작
            motionManager.startAccelerometerUpdates()
            motionManager.startGyroUpdates()
            
            // 타이머를 통해 동기화된 데이터 수집
            Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] timer in
                guard let self = self else { return }
                if !self.motionManager.isAccelerometerActive { timer.invalidate() }
                
                if let accData = self.motionManager.accelerometerData,
                   let gyroData = self.motionManager.gyroData {
                    
                    let data = MotionData(
                        timestamp: Date().timeIntervalSince1970,
                        accX: accData.acceleration.x, accY: accData.acceleration.y, accZ: accData.acceleration.z,
                        gyroX: gyroData.rotationRate.x, gyroY: gyroData.rotationRate.y, gyroZ: gyroData.rotationRate.z
                    )
                    
                    self.motionDataArray.append(data)
                    
                    // UI 업데이트
                    DispatchQueue.main.async {
                        self.accX = data.accX; self.accY = data.accY; self.accZ = data.accZ
                        self.gyroX = data.gyroX; self.gyroY = data.gyroY; self.gyroZ = data.gyroZ
                    }
                }
            }
        }
    }

    func stopUpdates(activityName: String) {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        
        saveAndSendCSV(activityName: activityName)
    }

    private func saveAndSendCSV(activityName: String) {
        guard !motionDataArray.isEmpty else { return }
        
        // 1. 파일 이름 생성 (운동종류_날짜시간)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmm"
        let dateString = formatter.string(from: Date())
        let fileName = "\(activityName)_\(dateString).csv"
        
        // 2. CSV 내용 작성
        var csvString = "timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ\n"
        for data in motionDataArray {
            let row = "\(data.timestamp),\(data.accX),\(data.accY),\(data.accZ),\(data.gyroX),\(data.gyroY),\(data.gyroZ)\n"
            csvString.append(row)
        }
        
        // 3. 파일로 저장
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        do {
            try csvString.write(to: path, atomically: true, encoding: .utf8)
            
            // 4. 아이폰으로 파일 전송
            if WCSession.default.activationState == .activated {
                WCSession.default.transferFile(path, metadata: nil)
                print("파일 전송 시작: \(fileName)")
            }
        } catch {
            print("파일 저장 실패: \(error)")
        }
    }
}
