import Foundation
import CoreMotion
import WatchConnectivity

// 데이터 구조체
struct MotionData {
    let timestamp: Double
    let accX: Double, accY: Double, accZ: Double
    let gyroX: Double, gyroY: Double, gyroZ: Double
}

class MotionManager: ObservableObject {
    private let motionManager = CMMotionManager()
    private let sensorQueue = OperationQueue() // 백그라운드 작업 큐
    
    // UI 표시용
    @Published var accX: Double = 0.0
    @Published var accY: Double = 0.0
    @Published var accZ: Double = 0.0
    @Published var gyroX: Double = 0.0
    @Published var gyroY: Double = 0.0
    @Published var gyroZ: Double = 0.0
    
    // 저장용 배열
    private var motionDataArray: [MotionData] = []
    
    // 50Hz (0.02초)
    private let updateInterval = 0.02
    
    func startUpdates() {
        print("------- [DeviceMotion] 통합 센서 수집 시작 -------")
        motionDataArray.removeAll()
        
        // [핵심 변경] Acc, Gyro를 따로 부르지 않고 'DeviceMotion' 하나로 통일
        // 이게 훨씬 안정적이고 자이로 누락이 없습니다.
        guard motionManager.isDeviceMotionAvailable else {
            print("🚨 [치명적 오류] DeviceMotion을 사용할 수 없습니다. (재부팅 필요)")
            return
        }
        
        motionManager.deviceMotionUpdateInterval = updateInterval
        
        motionManager.startDeviceMotionUpdates(to: sensorQueue) { [weak self] (data, error) in
            guard let self = self, let data = data else {
                if let error = error { print("🚨 센서 에러: \(error)") }
                return
            }
            
            // 1. 데이터 추출
            // DeviceMotion은 '중력'과 '사용자 움직임'을 분리해서 줍니다.
            // 기존처럼 Raw 데이터를 원하시면 gravity + userAcceleration을 더하면 됩니다.
            // 여기서는 직관적인 'userAcceleration(순수 움직임)'과 'rotationRate(자이로)'를 씁니다.
            
            // (만약 중력까지 포함된 쌩 날것의 가속도가 필요하면: data.gravity.x + data.userAcceleration.x)
            let currentAccX = data.gravity.x + data.userAcceleration.x
            let currentAccY = data.gravity.y + data.userAcceleration.y
            let currentAccZ = data.gravity.z + data.userAcceleration.z
            
            let currentGyroX = data.rotationRate.x
            let currentGyroY = data.rotationRate.y
            let currentGyroZ = data.rotationRate.z
            
            // 2. UI 업데이트 (메인 쓰레드)
            DispatchQueue.main.async {
                self.accX = currentAccX
                self.accY = currentAccY
                self.accZ = currentAccZ
                
                self.gyroX = currentGyroX
                self.gyroY = currentGyroY
                self.gyroZ = currentGyroZ
            }
            
            // 3. 데이터 저장 (여기서 바로 저장 - DeviceMotion은 동기화가 잘 되어 있음)
            let motionData = MotionData(
                timestamp: Date().timeIntervalSince1970,
                accX: currentAccX, accY: currentAccY, accZ: currentAccZ,
                gyroX: currentGyroX, gyroY: currentGyroY, gyroZ: currentGyroZ
            )
            self.motionDataArray.append(motionData)
        }
    }

    func stopUpdates(activityName: String) {
        print("------- 센서 수집 종료 (데이터 개수: \(motionDataArray.count)) -------")
        motionManager.stopDeviceMotionUpdates() // 통합 중지
        
        saveAndSendCSV(activityName: activityName)
    }

    private func saveAndSendCSV(activityName: String) {
        guard !motionDataArray.isEmpty else {
            print("⚠️ 저장된 데이터가 없습니다.")
            return
        }
        
        var csvString = "timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ\n"
        for data in motionDataArray {
            let row = String(format: "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                             data.timestamp,
                             data.accX, data.accY, data.accZ,
                             data.gyroX, data.gyroY, data.gyroZ)
            csvString.append(row)
        }
        
        let fileName = "\(activityName)_\(Int(Date().timeIntervalSince1970)).csv"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        do {
            try csvString.write(to: tempURL, atomically: true, encoding: .utf8)
            if WCSession.default.activationState == .activated {
                WCSession.default.transferFile(tempURL, metadata: nil)
                print("🚀 파일 전송 시작: \(fileName)")
            }
        } catch {
            print("파일 저장 실패: \(error)")
        }
    }
} 
