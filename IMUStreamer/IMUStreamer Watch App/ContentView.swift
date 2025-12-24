import SwiftUI

struct ContentView: View {
    @StateObject var connectivity = ConnectivityManager.shared
    @StateObject var motion = MotionManager()
    @State var isRecording = false
    
    var body: some View {
        // 1. 운동이 선택되지 않았을 때의 화면
        if connectivity.selectedActivity == "선택 안됨" {
            VStack {
                Image(systemName: "iphone.radiowaves.left.and.right")
                    .font(.largeTitle)
                    .foregroundColor(.blue)
                Text("아이폰에서\n운동을 선택하세요")
                    .multilineTextAlignment(.center)
                    .padding()
            }
        } 
        // 2. 운동이 선택되었을 때 나타나는 수집 화면
        else {
            ScrollView {
                VStack(spacing: 8) {
                    Text("현재 운동")
                        .font(.caption2)
                    Text(connectivity.selectedActivity)
                        .font(.headline)
                        .foregroundColor(.yellow)
                    
                    Divider()
                    
                    // 수치 표시창
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Acc").bold()
                            Text("X:\(motion.accX, specifier: "%.3f")")
                            Text("Y:\(motion.accY, specifier: "%.3f")")
                            Text("Z:\(motion.accZ, specifier: "%.3f")")
                        }
                        Spacer()
                        VStack(alignment: .leading) {
                            Text("Gyro").bold()
                            Text("X:\(motion.gyroX, specifier: "%.3f")")
                            Text("Y:\(motion.gyroY, specifier: "%.3f")")
                            Text("Z:\(motion.gyroZ, specifier: "%.3f")")
                        }
                    }
                    .font(.system(size: 12, design: .monospaced))
                    
                    Divider()
                    
                    if !isRecording {
                        Button("시작") {
                            isRecording = true
                            motion.startUpdates()
                        }
                        .tint(.green)
                    } else {
                        Button("종료") {
                            isRecording = false
                            motion.stopUpdates(activityName: connectivity.selectedActivity)
                            
                            // 전송 후 다시 대기 화면으로 돌아가고 싶다면 주석 해제
                            // connectivity.selectedActivity = "선택 안됨"
                        }
                        .tint(.red)
                    }
                }
            }
        }
    }
}
