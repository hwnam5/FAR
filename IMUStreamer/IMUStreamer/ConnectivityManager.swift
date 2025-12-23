import Foundation
import WatchConnectivity
import Combine

class ConnectivityManager: NSObject, WCSessionDelegate, ObservableObject {
    // 싱글톤 인스턴스 (앱 전체에서 공유)
    static let shared = ConnectivityManager()
    
    // 1. 현재 선택된 운동 이름 (워치/아이폰 공통)
    @Published var selectedActivity: String = "선택 안됨"
    
    // 2. 파일 수신 시 아이폰 UI를 띄우기 위한 변수들
    @Published var receivedFileURL: URL?
    @Published var isShowingShareSheet: Bool = false
    
    override init() {
        super.init()
        // WCSession 지원 여부 확인 및 활성화
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }
    
    // MARK: - [보내기] 아이폰 -> 워치 (운동 종류 변경)
    func sendActivityChange(_ activity: String) {
        let session = WCSession.default
        
        // 방법 1: 앱이 켜져 있을 때 즉시 전송
        if session.isReachable {
            session.sendMessage(["activity": activity], replyHandler: nil)
        }
        
        // 방법 2: 백그라운드 상태를 대비해 컨텍스트 업데이트
        do {
            try session.updateApplicationContext(["activity": activity])
        } catch {
            print("데이터 전송 실패: \(error.localizedDescription)")
        }
    }
    
    // MARK: - [받기] 메시지 수신 (운동 이름 동기화)
    
    // 1. sendMessage로 받았을 때
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        DispatchQueue.main.async {
            if let activity = message["activity"] as? String {
                self.selectedActivity = activity
            }
        }
    }
    
    // 2. updateApplicationContext로 받았을 때
    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        DispatchQueue.main.async {
            if let activity = applicationContext["activity"] as? String {
                self.selectedActivity = activity
            }
        }
    }
    
    // MARK: - [받기] 파일 수신 (워치 -> 아이폰 CSV 도착)
    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        let fileManager = FileManager.default
        
        // 저장할 경로: 아이폰의 문서(Documents) 폴더
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationURL = documentsURL.appendingPathComponent(file.fileURL.lastPathComponent)
        
        do {
            // 이미 같은 이름의 파일이 있으면 삭제
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            
            // 임시 저장소에 있는 파일을 문서 폴더로 이동
            try fileManager.moveItem(at: file.fileURL, to: destinationURL)
            print("파일 저장 완료: \(destinationURL.lastPathComponent)")
            
            // UI 업데이트: 메인 스레드에서 공유 창(Share Sheet) 띄우기 신호 보냄
            DispatchQueue.main.async {
                self.receivedFileURL = destinationURL
                self.isShowingShareSheet = true
            }
            
        } catch {
            print("파일 이동 중 에러 발생: \(error.localizedDescription)")
        }
    }
    
    // MARK: - WCSession 필수 델리게이트 (연결 상태 관리)
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if let error = error {
            print("WCSession 활성화 실패: \(error.localizedDescription)")
        } else {
            print("WCSession 활성화됨: \(activationState.rawValue)")
        }
    }
    
    #if os(iOS)
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) {
        // 세션이 비활성화되면 다시 활성화 시도
        WCSession.default.activate()
    }
    #endif
}